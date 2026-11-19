//! The new runtime's side: the case vocabulary on `metal2vulkan-rt`'s public API.
//!
//! The vocabulary is ported one family at a time, as the old runtime's side was. A call this side
//! does not answer yet answers [`Failure::unavailable`] with its name, so a case that needs it
//! ends refused, its row naming the call: nothing is skipped, and nothing is answered for it. The
//! rows it refuses are held to their own ledger, `known-gaps-rt.tsv`
//! ([`crate::known_gaps::rt_ledger`]). The old runtime's side ([`crate::runtime_side`]) is
//! untouched by this one. A translator refusal that names capabilities the device offers is the
//! runtime's refusal and not the venue's lack: the runtime's floor leaves them off
//! (`RtDevice::failure`).
//!
//! The runtime places a buffer where its consumer says, in a block the consumer took; this side
//! takes a block of the buffer's own length for each buffer, Shared, as a case's buffers are, and
//! the runtime zeroes it as `newBufferWithLength:options:` does. It grants itself every texture and
//! sampler index once the device is made ([`RtDevice::open`]), and gives each texture a shader
//! reaches the next texture index and each constexpr sampler the index the first state that
//! numbered its state words took, as Metal gives an equal one the same ID.
//!
//! The runtime announces no degrade notice: where a kernel's answers may differ from Metal's, its
//! state carries static precision notes ([`metal2vulkan_rt::ComputePipelineState::precision_notes`],
//! [`metal2vulkan_rt::RenderPipelineState::precision_notes`]). This side announces the notes of
//! each state its case dispatched or drew with, by the codes the suite's allowances name
//! (`metal2vulkan-degrade-<kind>`).

use crate::case::{
    Answer, Bindings, Failure, Handle, SamplerSpec, ShaderSource, Side, StageInRegion, StageInput,
    TextureSpec, Venue,
};
use crate::command_buffer_facts::CommandBufferFacts;
use crate::device::Venues;
use crate::empty_command_buffers::{
    Alone, Behind, EmptyCommandBuffers, ARRIVAL, EMPTY, FILL, FILL_BYTES,
};
use crate::event_orders::Status;
use crate::transcript::{Negative, Value};
use ash::vk;
use metal2vulkan::device::{ApiVersion, DeviceDescription};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_rt as rt;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

mod compute;
mod render;
mod work;

/// How long a wait for a command buffer waits before the side records that it never returned:
/// the old runtime's side's bound.
const WAIT: Duration = Duration::from_secs(60);

/// The bytes the command-buffer facts' one blit fills, as the other sides fill.
const TIMED_BYTES: usize = 256;

/// Why the command-buffer facts' queue label and getters are not this side's to answer.
const THE_CONSUMERS: &str = "a queue's label and a command buffer's device and commandQueue \
                             never reach the GPU: metal2vulkan-rt leaves them to its consumer";

/// The command buffers a queue of this side's holds at once: `newCommandQueue`'s.
const QUEUE_COMMAND_BUFFERS: u32 = 64;

/// The new runtime's device a replay runs on, with the venue it reports and the indices it hands
/// out.
pub struct RtDevice {
    device: rt::Device,
    venue: Venue,
    indices: RefCell<Indices>,
    /// What the physical device offers, as the translator's description reads it: what decides
    /// whether a refusal that names capabilities is the venue's lack or the runtime's
    /// ([`RtDevice::failure`]).
    offered: DeviceDescription,
}

impl RtDevice {
    /// The first physical device of the kind `venues` names that opens, in the order Metal's
    /// default device would take it (discrete, integrated, virtual; a CPU implementation for
    /// [`Venues::Cpu`]), with every texture and sampler index granted to this side.
    pub fn open(venues: Venues) -> Result<RtDevice, String> {
        let instance = rt::Instance::new(&[]).map_err(|error| error.to_string())?;
        let (_, raw) = instance.raw();
        let mut candidates: Vec<(u8, rt::PhysicalDevice)> = (instance.physical_devices())
            .map_err(|error| error.to_string())?
            .into_iter()
            .filter_map(|physical| {
                // SAFETY: the physical device is the instance's, which is live for the call.
                let kind =
                    unsafe { raw.get_physical_device_properties(physical.raw()) }.device_type;
                let rank = match (venues, kind) {
                    (Venues::Gpu, vk::PhysicalDeviceType::DISCRETE_GPU) => 0,
                    (Venues::Gpu, vk::PhysicalDeviceType::INTEGRATED_GPU) => 1,
                    (Venues::Gpu, vk::PhysicalDeviceType::VIRTUAL_GPU) => 2,
                    (Venues::Cpu, vk::PhysicalDeviceType::CPU) => 0,
                    _ => return None,
                };
                Some((rank, physical))
            })
            .collect();
        candidates.sort_by_key(|(rank, _)| *rank);
        let mut refused = Vec::new();
        for (_, physical) in candidates {
            let venue = venue(&instance, &physical);
            // SAFETY: the physical device is the instance's, which the runtime made at its
            // floor's version.
            let offered = unsafe {
                DeviceDescription::query(
                    raw,
                    physical.raw(),
                    ApiVersion::from_packed(rt::floor::VERSION),
                )
            }
            .map_err(|error| format!("{}: {error}", venue.device))?;
            match rt::Device::new(&physical, rt::DeviceOptions::default()) {
                Ok(device) => {
                    for class in [rt::IndexClass::Texture, rt::IndexClass::Sampler] {
                        let range = class.range();
                        (device.grant_indices(class, range.start, range.end - range.start))
                            .map_err(|error| error.to_string())?;
                    }
                    return Ok(RtDevice {
                        device,
                        venue,
                        indices: RefCell::default(),
                        offered,
                    });
                }
                Err(why) => refused.push(format!("{}: {why}", venue.device)),
            }
        }
        Err(format!(
            "no {venues:?} device opens for metal2vulkan-rt: {}",
            refused.join("; ")
        ))
    }

    /// The venue the device reports.
    pub fn venue(&self) -> &Venue {
        &self.venue
    }

    /// The runtime's device.
    pub fn device(&self) -> &rt::Device {
        &self.device
    }

    /// The failure a typed error of the runtime is on this device. A translator refusal that names
    /// capabilities is the venue's lack only where the device offers none of its alternatives;
    /// where it offers one, the capability is one `metal2vulkan-rt`'s floor leaves off, a typed
    /// refusal of the runtime's until a stream uses it, and the failure is the runtime's refusal.
    pub(crate) fn failure(&self, error: &rt::Error) -> Failure {
        let mut failure = Failure::of_rt(error);
        if let rt::Error::Translation(metal2vulkan::error::Error::Unsupported(unsupported)) = error
        {
            if offers_one(&self.offered, unsupported) {
                failure.lacks = None;
            }
        }
        failure
    }

    /// [`RtDevice::failure`] of a translator error, where the side calls the translator's library
    /// itself.
    pub(crate) fn translation_failure(&self, error: &metal2vulkan::error::Error) -> Failure {
        let mut failure = Failure::of_translation(error);
        if let metal2vulkan::error::Error::Unsupported(unsupported) = error {
            if offers_one(&self.offered, unsupported) {
                failure.lacks = None;
            }
        }
        failure
    }

    /// The next index of `class`, round the class's range: an object made there takes the index
    /// from an earlier one, which a case, making far fewer objects than the range holds, does not
    /// keep that long. A sampler index a constexpr sampler took is passed over.
    fn next_index(&self, class: rt::IndexClass) -> u32 {
        let mut indices = self.indices.borrow_mut();
        let range = class.range();
        let mut index = *indices.last(class);
        loop {
            index = match index + 1 {
                next if range.contains(&next) => next,
                _ => range.start,
            };
            if class != rt::IndexClass::Sampler || !indices.constexprs.values().any(|&i| i == index)
            {
                *indices.last(class) = index;
                return index;
            }
        }
    }

    /// The next texture index ([`RtDevice::next_index`]) for a texture of `usage` where a shader
    /// reaches it (shader read or write, or no bit, which the runtime takes as every usage), and
    /// `None` where none does, as the runtime's texture makers take an index.
    fn texture_index(&self, usage: rt::TextureUsage) -> Option<u32> {
        let reached = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
        (usage.is_empty() || usage.intersects(reached))
            .then(|| self.next_index(rt::IndexClass::Texture))
    }

    /// The indices of a pipeline state's numbered samplers: a constexpr sampler takes the index
    /// its state words took first; a read's or a write's sampler takes nothing at its index.
    fn sampler_ids(&self, samplers: &[rt::NumberedSampler]) -> Vec<u32> {
        samplers
            .iter()
            .map(|sampler| match sampler {
                rt::NumberedSampler::Constexpr(words) => {
                    let taken = self.indices.borrow().constexprs.get(words).copied();
                    taken.unwrap_or_else(|| {
                        let index = self.next_index(rt::IndexClass::Sampler);
                        self.indices.borrow_mut().constexprs.insert(*words, index);
                        index
                    })
                }
                _ => 0,
            })
            .collect()
    }
}

/// Whether `offered` holds every capability of one of the alternatives `unsupported` names: its
/// features and extensions. A property, limit, format or version is the device's own answer,
/// which device creation does not choose, and a refusal of another kind names no alternative.
pub(crate) fn offers_one(
    offered: &DeviceDescription,
    unsupported: &metal2vulkan::error::Unsupported,
) -> bool {
    use metal2vulkan::error::{Capability, Needs};
    let Needs::AnyOf(alternatives) = &unsupported.needs else {
        return false;
    };
    alternatives.iter().any(|all| {
        all.iter().all(|capability| match capability {
            Capability::Feature(feature) => offered.features.contains(feature),
            Capability::Extension(extension) => offered.extensions.contains(extension),
            _ => false,
        })
    })
}

/// The texture and sampler indices the side hands out: the last index of each class taken, and
/// each constexpr sampler's index by its state words.
#[derive(Default)]
struct Indices {
    texture: u32,
    sampler: u32,
    constexprs: HashMap<[u64; 2], u32>,
}

impl Indices {
    /// The last index of `class` taken.
    fn last(&mut self, class: rt::IndexClass) -> &mut u32 {
        match class {
            rt::IndexClass::Sampler => &mut self.sampler,
            _ => &mut self.texture,
        }
    }
}

/// The venue of `physical`, a device of `instance`: its name, the runtime and the driver, and the
/// API version as Vulkan reports them, and the host's vector instruction set for a CPU
/// implementation.
fn venue(instance: &rt::Instance, physical: &rt::PhysicalDevice) -> Venue {
    let (_, raw) = instance.raw();
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    // SAFETY: the physical device is the instance's, which is live for the call.
    unsafe { raw.get_physical_device_properties2(physical.raw(), &mut properties) };
    let properties = properties.properties;
    let text = |chars: &[std::ffi::c_char]| {
        let bytes: Vec<u8> = chars
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    Venue {
        device: text(&properties.device_name),
        driver: format!(
            "metal2vulkan-rt on {} {}",
            text(&driver.driver_name),
            text(&driver.driver_info)
        ),
        api: format!(
            "{}.{}",
            vk::api_version_major(properties.api_version),
            vk::api_version_minor(properties.api_version)
        ),
        lossless_compression: false,
        host: (properties.device_type == vk::PhysicalDeviceType::CPU)
            .then(crate::case::host_vector_isa),
    }
}

/// A heap as the runtime's consumer makes one: a block of the heap's size and storage, and the
/// range of it the heap is. The runtime keeps no heap object; what Metal's `MTLHeap` answers of
/// itself is the consumer's ([`heap_is_the_consumers`]).
struct RtHeap {
    block: rt::Block,
    range: rt::HeapRange,
    /// Whether the heap is a placement heap, whose buffers its calls place at an offset.
    placement: bool,
}

/// The consumer's own, for `MTLHeap`'s calls and getters and an automatic heap's suballocation:
/// the runtime keeps no heap object, and its README names the heap as the consumer's.
fn heap_is_the_consumers() -> Failure {
    Failure::named(
        "TheConsumers",
        "metal2vulkan-rt keeps no heap object: its consumer places resources in the blocks it takes \
         and implements MTLHeap's calls and getters",
    )
}

/// The `MTLPipelineOption` bits the consumer's own reason names
/// ([`ConsumerObject::PipelineReflection`](crate::known_gaps::ConsumerObject::PipelineReflection)):
/// `MTLPipelineOptionBindingInfo` (`MTLPipelineOptionArgumentInfo`'s bit),
/// `MTLPipelineOptionBufferTypeInfo` and `MTLPipelineOptionFailOnBinaryArchiveMiss`.
const NAMED_OPTIONS: u64 = (1 << 0) | (1 << 1) | (1 << 2);

/// The consumer's own, for a pipeline creation under `options` that asks for reflection or takes
/// options: the runtime builds no pipeline reflection and acts on no `MTLPipelineOption`, and its
/// consumer builds reflection from the library's AIR, as the runtime's README says. A word with a
/// bit outside the named options is `call`'s, unavailable, so an option the reason does not name is
/// never taken for the consumer's.
pub(crate) fn reflection_is_the_consumers(call: &str, options: u64) -> Failure {
    if options & !NAMED_OPTIONS != 0 {
        return Failure::unavailable(call);
    }
    Failure::named(
        "TheConsumers",
        "metal2vulkan-rt builds no pipeline reflection and acts on no MTLPipelineOption \
         (FailOnBinaryArchiveMiss, ArgumentInfo, BufferTypeInfo, BindingInfo): its consumer builds \
         reflection from the library's AIR",
    )
}

/// The new runtime's side of one case.
pub struct RtSide<'a> {
    on: &'a RtDevice,
    queue: Option<rt::CommandQueue>,
    libraries: Vec<rt::Library>,
    pipelines: Vec<rt::ComputePipelineState>,
    /// Each buffer, with the block it lies in, which it alone takes.
    buffers: Vec<(rt::Buffer, rt::Block)>,
    textures: Vec<rt::Texture>,
    command_buffers: Vec<rt::CommandBuffer>,
    samplers: Vec<rt::SamplerState>,
    render_pipelines: Vec<rt::RenderPipelineState>,
    depth_stencils: Vec<rt::DepthStencilState>,
    /// Each heap the case made, as its consumer makes one ([`Side::heap`]).
    heaps: Vec<RtHeap>,
    /// The pipelines the case dispatched, whose precision notes [`Side::announced`] answers.
    dispatched: BTreeSet<usize>,
    /// The render pipeline states a pass the case encoded set, whose precision notes
    /// [`Side::announced`] answers too.
    drawn: BTreeSet<usize>,
}

impl<'a> RtSide<'a> {
    /// A side of one case on `on`.
    pub fn new(on: &'a RtDevice) -> Self {
        Self {
            on,
            queue: None,
            libraries: Vec::new(),
            pipelines: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            command_buffers: Vec::new(),
            samplers: Vec::new(),
            render_pipelines: Vec::new(),
            depth_stencils: Vec::new(),
            heaps: Vec::new(),
            dispatched: BTreeSet::new(),
            drawn: BTreeSet::new(),
        }
    }

    /// The case's queue, made at its first use.
    fn queue(&mut self) -> Answer<&rt::CommandQueue> {
        if self.queue.is_none() {
            let queue = (self.on.device.new_command_queue(QUEUE_COMMAND_BUFFERS))
                .map_err(|error| Failure::of_rt(&error))?;
            self.queue = Some(queue);
        }
        Ok(self.queue.as_ref().expect("made above"))
    }

    fn held(&self, handle: Handle) -> Answer<&rt::Buffer> {
        (self.buffers.get(handle.0 as usize))
            .map(|(buffer, _)| buffer)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    }

    fn held_heap(&self, handle: Handle) -> Answer<&RtHeap> {
        (self.heaps.get(handle.0 as usize))
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("heap {}", handle.0)))
    }

    fn held_texture(&self, handle: Handle) -> Answer<&rt::Texture> {
        (self.textures.get(handle.0 as usize))
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    }

    fn held_library(&self, handle: Handle) -> Answer<&rt::Library> {
        (self.libraries.get(handle.0 as usize))
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", handle.0)))
    }

    /// A compute command buffer of `pipeline` with `bindings` at index 0 up, `dispatch` encoded
    /// into it, committed.
    fn encode(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        dispatch: impl FnOnce(&mut rt::ComputeCommandEncoder) -> Result<(), rt::Error>,
    ) -> Answer<Handle> {
        let at = pipeline.0 as usize;
        let state = (self.pipelines.get(at))
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {at}")))?
            .clone();
        let buffers = (bindings.buffers.iter())
            .map(|&handle| self.held(handle).cloned())
            .collect::<Answer<Vec<rt::Buffer>>>()?;
        let textures = (bindings.textures.iter())
            .map(|&handle| self.held_texture(handle).cloned())
            .collect::<Answer<Vec<rt::Texture>>>()?;
        let samplers = (bindings.samplers.iter())
            .map(|&handle| {
                (self.samplers.get(handle.0 as usize).cloned())
                    .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
            })
            .collect::<Answer<Vec<rt::SamplerState>>>()?;
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let command_buffer = self.queue()?.command_buffer().map_err(of_rt)?;
        let mut encoder =
            (command_buffer.compute_command_encoder(rt::DispatchType::Serial)).map_err(of_rt)?;
        encoder.set_compute_pipeline_state(&state).map_err(of_rt)?;
        for (index, buffer) in buffers.iter().enumerate() {
            (encoder.set_buffer(Some(buffer), 0, index as u32)).map_err(of_rt)?;
        }
        for (index, texture) in textures.iter().enumerate() {
            (encoder.set_texture(Some(texture), index as u32)).map_err(of_rt)?;
        }
        for (index, sampler) in samplers.iter().enumerate() {
            (encoder.set_sampler_state(Some(sampler), index as u32)).map_err(of_rt)?;
        }
        dispatch(&mut encoder).map_err(of_rt)?;
        encoder.end_encoding();
        command_buffer.commit().map_err(of_rt)?;
        self.dispatched.insert(at);
        Ok(keep(&mut self.command_buffers, command_buffer))
    }
}

/// `value` kept in `kept`, by its handle.
fn keep<T>(kept: &mut Vec<T>, value: T) -> Handle {
    kept.push(value);
    Handle(u32::try_from(kept.len() - 1).expect("fewer than 2^32 objects"))
}

/// Host nanoseconds as Metal's seconds.
fn seconds(nanos: u64) -> f64 {
    nanos as f64 / 1e9
}

/// A command buffer's kernel times, in seconds.
fn kernel_times(command_buffer: &rt::CommandBuffer) -> [f64; 2] {
    [
        seconds(command_buffer.kernel_start_time()),
        seconds(command_buffer.kernel_end_time()),
    ]
}

/// A command buffer's GPU times, in seconds.
fn gpu_times(command_buffer: &rt::CommandBuffer) -> [f64; 2] {
    [
        seconds(command_buffer.gpu_start_time()),
        seconds(command_buffer.gpu_end_time()),
    ]
}

/// The status word once `command_buffer`'s wait returned, or `None` where it did not within
/// `bound`; the refusal's text where the runtime refused the wait. The runtime's wait has no
/// bound, so it runs on a thread of its own, which a wait that never returns leaves behind.
fn waited_for(command_buffer: &rt::CommandBuffer, bound: Duration) -> Option<String> {
    let (sent, received) = mpsc::channel();
    let waiting = command_buffer.clone();
    std::thread::spawn(move || {
        let word = match waiting.wait_until_completed() {
            Ok(()) => format!("{:?}", waiting.status()),
            Err(error) => error.to_string(),
        };
        let _ = sent.send(word);
    });
    received.recv_timeout(bound).ok()
}

/// The runtime's descriptor of a query, each field as the query states it, a zero included, so
/// that what Metal refuses of it the runtime is asked too.
fn queried(query: &crate::texture_query::HeapTextureQuery) -> rt::TextureDescriptor {
    let [width, height, depth] = query.size;
    let mut d = rt::TextureDescriptor::new(query.texture_type, query.format, width, height);
    d.depth = depth;
    d.mipmap_level_count = query.levels;
    d.sample_count = query.samples;
    d.array_length = query.array_length;
    d.usage = query.usage;
    d.storage_mode = work::storage_mode(query.storage);
    d
}

/// What a texture the runtime made answers of itself, read of its descriptor: each `MTLTexture`
/// getter the runtime's descriptor states, and at `cpuCacheMode`, `hazardTrackingMode` and
/// `allowGPUOptimizedContents`, which it does not, the consumer's own.
fn texture_facts(texture: &rt::Texture) -> crate::buffer_texture::BufferTextureFacts {
    let d = texture.descriptor();
    let consumers = || {
        Failure::named(
            "TheConsumers",
            "metal2vulkan-rt's TextureDescriptor states no cpuCacheMode, hazardTrackingMode or \
             allowGPUOptimizedContents: its consumer implements them",
        )
        .recorded()
    };
    crate::buffer_texture::BufferTextureFacts {
        texture_type: u64::from(d.texture_type.as_raw()),
        pixel_format: u64::from(d.pixel_format.as_raw()),
        width: d.width.into(),
        height: d.height.into(),
        depth: d.depth.into(),
        mipmap_level_count: d.mipmap_level_count.into(),
        array_length: d.array_length.into(),
        sample_count: d.sample_count.into(),
        usage: u64::from(d.usage.as_raw()),
        // `MTLStorageMode`'s raw values.
        storage_mode: match d.storage_mode {
            rt::StorageMode::Shared => 0,
            rt::StorageMode::Managed => 1,
            rt::StorageMode::Private => 2,
            rt::StorageMode::Memoryless => 3,
        },
        cpu_cache_mode: consumers(),
        hazard_tracking_mode: consumers(),
        gpu_optimized: consumers(),
    }
}

/// [`waited_for`] within [`ARRIVAL`], as a status.
fn waited(command_buffer: &rt::CommandBuffer) -> Status {
    waited_for(command_buffer, ARRIVAL).map_or(Status::NeverSettled, Status::read)
}

impl Side for RtSide<'_> {
    fn venue(&self) -> Venue {
        self.on.venue.clone()
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        let library = rt::Library::from_air(source.bitcode)
            .map_err(|error| self.on.translation_failure(&error))?;
        Ok(keep(&mut self.libraries, library))
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let function = (self.held_library(library)?.function(entry))
            .map_err(|error| self.on.translation_failure(&error))?;
        let descriptor = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(
            rt::FunctionDescriptor::new(function),
        ));
        let on = self.on;
        let state = (on.device)
            .new_compute_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| on.failure(&error))?;
        Ok(keep(&mut self.pipelines, state))
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        let handle = self.zeroed_buffer(bytes.len())?;
        let contents = (self.held(handle)?.contents())
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
        // SAFETY: the buffer was just made with at least `bytes.len()` bytes, the host reaches
        // them, and no command uses it yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        Ok(handle)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        let length = length.max(1) as u64;
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let block = (self.on.device.take_block(rt::MemoryClass::Shared, length)).map_err(of_rt)?;
        let buffer = (self.on.device.new_buffer(block.id, 0, length, Some(0))).map_err(of_rt)?;
        Ok(keep(&mut self.buffers, (buffer, block)))
    }

    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle> {
        let mut descriptor =
            rt::TextureDescriptor::new(rt::TextureType::Type2D, format, size[0], size[1]);
        descriptor.usage = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
        descriptor.storage_mode = rt::StorageMode::Shared;
        let index = self.on.next_index(rt::IndexClass::Texture);
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let texture = (self.on.device.new_texture(&descriptor, Some(index))).map_err(of_rt)?;
        let region = rt::Region {
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
        };
        (texture.replace_region(region, 0, 0, bytes, row_pitch.into(), 0)).map_err(of_rt)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let texture = self.held_texture(texture)?;
        let mut bytes = vec![0u8; row_pitch as usize * size[1] as usize];
        let region = rt::Region {
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
        };
        (texture.get_bytes(&mut bytes, row_pitch.into(), 0, region, 0, 0))
            .map_err(|error| Failure::of_rt(&error))?;
        Ok(bytes)
    }

    fn read_texture_over(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        before: &[u8],
    ) -> Answer<Vec<u8>> {
        let wanted = row_pitch as usize * size[1] as usize;
        if before.len() != wanted {
            return Err(Failure::named(
                "TooFewBytes",
                format!(
                    "{} bytes before the read, and it reads {wanted}",
                    before.len()
                ),
            ));
        }
        let texture = self.held_texture(texture)?;
        let mut bytes = before.to_vec();
        let region = rt::Region {
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
        };
        (texture.get_bytes(&mut bytes, row_pitch.into(), 0, region, 0, 0))
            .map_err(|error| Failure::of_rt(&error))?;
        Ok(bytes)
    }

    /// `newTextureWithDescriptor:` of the query, kept, and what it answers of itself
    /// (`texture_facts`).
    fn queried_texture_facts(
        &mut self,
        query: &crate::texture_query::HeapTextureQuery,
    ) -> Answer<Option<crate::buffer_texture::BufferTextureFacts>> {
        let descriptor = queried(query);
        let index = self.on.texture_index(descriptor.usage);
        let texture = (self.on.device.new_texture(&descriptor, index))
            .map_err(|error| Failure::of_rt(&error))?;
        let facts = texture_facts(&texture);
        keep(&mut self.textures, texture);
        Ok(Some(facts))
    }

    /// `newTextureWithDescriptor:offset:bytesPerRow:` on a buffer of the request's length and
    /// storage, in blocks of its own, and what the texture answers of itself
    /// (`texture_facts`). The texture is kept; it holds the buffer, and the buffer its blocks'
    /// memory, as the runtime's `Buffer::new_texture` says.
    fn buffer_texture(
        &mut self,
        request: &crate::buffer_texture::BufferTextureRequest,
    ) -> Answer<crate::buffer_texture::BufferTextureFacts> {
        let (buffer, _blocks) =
            work::buffer_of(self.on, request.buffer_length, request.buffer_storage)?;
        let descriptor = queried(&request.descriptor);
        let index = self.on.texture_index(descriptor.usage);
        let made = buffer.new_texture(&descriptor, index, request.offset, request.bytes_per_row);
        let texture = made.map_err(|error| Failure::of_rt(&error))?;
        let facts = texture_facts(&texture);
        keep(&mut self.textures, texture);
        Ok(facts)
    }

    fn heap_texture_size_and_align(
        &mut self,
        query: &crate::texture_query::HeapTextureQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let sized = (self.on.device.heap_texture_size_and_align(&queried(query)))
            .map_err(|error| Failure::of_rt(&error))?;
        Ok(crate::texture_query::SizeAndAlign {
            size: sized.size,
            align: sized.align,
        })
    }

    /// Whether `newTextureWithDescriptor:` of the query makes a texture, at the next texture index
    /// where a shader reaches it. The runtime answers no texture Metal answers nil for: what it
    /// does not make is its typed error, recorded as the failure it is.
    fn makes_texture(&mut self, query: &crate::texture_query::HeapTextureQuery) -> Answer<bool> {
        let descriptor = queried(query);
        let index = self.on.texture_index(descriptor.usage);
        (self.on.device.new_texture(&descriptor, index))
            .map(|_| true)
            .map_err(|error| Failure::of_rt(&error))
    }

    /// `newHeapWithDescriptor:` as the runtime's consumer makes a heap: a block of the heap's size
    /// and storage. The descriptor's CPU cache and hazard tracking modes are the consumer's own,
    /// which the runtime's blocks state neither of.
    fn heap(&mut self, spec: &crate::heap_query::HeapSpec) -> Answer<Handle> {
        if let Some(why) = spec.is_well_formed() {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let (block, range) = work::heap_of(self.on, spec.size, spec.storage)?;
        let placement = matches!(spec.heap_type, metal2vulkan_runtime::HeapType::Placement);
        Ok(keep(
            &mut self.heaps,
            RtHeap {
                block,
                range,
                placement,
            },
        ))
    }

    /// A placement heap's `newBufferWithLength:options:offset:`: the device's buffer at that
    /// offset of the heap's block, its bytes not zeroed, as a heap's are not. An automatic heap's
    /// suballocation, and a call without an offset, are the consumer's own.
    fn heap_buffer(
        &mut self,
        heap: Handle,
        spec: &crate::heap_query::HeapBufferSpec,
    ) -> Answer<Handle> {
        let held = self.held_heap(heap)?;
        let Some(offset) = spec.offset.filter(|_| held.placement) else {
            return Err(heap_is_the_consumers());
        };
        let at = held.range.offset + offset;
        let buffer = (self
            .on
            .device
            .new_buffer(held.range.block, at, spec.length, None))
        .map_err(|error| Failure::of_rt(&error))?;
        let block = held.block.clone();
        Ok(keep(&mut self.buffers, (buffer, block)))
    }

    fn heap_own_facts(&mut self, _heap: Handle) -> Answer<crate::heap_query::HeapOwnFacts> {
        Err(heap_is_the_consumers())
    }

    fn heap_max_available_size(&mut self, _heap: Handle, _alignment: u64) -> Answer<u64> {
        Err(heap_is_the_consumers())
    }

    fn set_heap_label(&mut self, _heap: Handle, _label: &str) -> Answer<()> {
        Err(heap_is_the_consumers())
    }

    fn heap_purgeable_state(
        &mut self,
        _heap: Handle,
        _state: metal2vulkan_runtime::PurgeableState,
    ) -> Answer<u64> {
        Err(heap_is_the_consumers())
    }

    /// The consumer's own: the runtime's `RenderPassDescriptor` is a value its consumer fills, so
    /// what Metal's descriptor object answers of its fields after its setters is the consumer's to
    /// keep, as the runtime's README says.
    fn pass_descriptor_facts(
        &mut self,
        _spec: &crate::pass_descriptors::PassSpec,
    ) -> Answer<crate::pass_descriptors::PassDescriptorFacts> {
        Err(Failure::named(
            "TheConsumers",
            "metal2vulkan-rt's RenderPassDescriptor is a value its consumer fills: what Metal's \
             MTLRenderPassDescriptor answers of its fields is the consumer's",
        ))
    }

    /// The consumer's own: a compute pipeline's bindings are its reflection's
    /// (`reflection_is_the_consumers`).
    fn compute_pipeline_bindings(
        &mut self,
        _library: Handle,
        _entry: &str,
    ) -> Answer<Vec<crate::function_bindings::BindingFacts>> {
        Err(reflection_is_the_consumers("compute_pipeline_bindings", 0))
    }

    /// The consumer's own (`reflection_is_the_consumers`).
    fn compute_pipeline_under_options(
        &mut self,
        _library: Handle,
        _entry: &str,
        options: u64,
    ) -> Answer<crate::pipeline_options::OptionOutcome> {
        Err(reflection_is_the_consumers(
            "compute_pipeline_under_options",
            options,
        ))
    }

    /// The consumer's own (`reflection_is_the_consumers`).
    fn render_pipeline_under_options(
        &mut self,
        _library: Handle,
        _vertex: &str,
        _fragment: &str,
        _attachment: PixelFormat,
        options: u64,
    ) -> Answer<crate::pipeline_options::OptionOutcome> {
        Err(reflection_is_the_consumers(
            "render_pipeline_under_options",
            options,
        ))
    }

    /// The consumer's own (`reflection_is_the_consumers`).
    fn compute_pipeline_reflection_under(
        &mut self,
        _library: Handle,
        _entry: &str,
        options: u64,
    ) -> Answer<crate::pipeline_option_reflection::UnderOptions> {
        Err(reflection_is_the_consumers(
            "compute_pipeline_reflection_under",
            options,
        ))
    }

    /// The consumer's own for a compute pipeline's two creation paths, which compare the
    /// reflection each delivers under its options (`reflection_is_the_consumers`); a render,
    /// tile or mesh pipeline's are unavailable.
    fn pipeline_both_paths(
        &mut self,
        _library: Handle,
        form: &crate::pipeline_handlers::Form<'_>,
    ) -> Answer<crate::pipeline_handlers::BothPaths> {
        use crate::pipeline_handlers::Form;
        const CALL: &str = "pipeline_both_paths";
        Err(match *form {
            Form::ComputeFunction { options, .. } => {
                reflection_is_the_consumers(CALL, options.unwrap_or(0))
            }
            Form::ComputeDescriptor { options, .. } => reflection_is_the_consumers(CALL, options),
            Form::Render { .. } | Form::Tile { .. } | Form::Mesh { .. } => {
                Failure::unavailable(CALL)
            }
        })
    }

    fn minimum_linear_texture_alignment(&mut self, format: PixelFormat) -> Answer<u64> {
        (self.on.device.minimum_linear_texture_alignment(format))
            .map_err(|error| Failure::of_rt(&error))
    }

    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, |encoder| {
            encoder.dispatch_threadgroups(threadgroups, threads)
        })
    }

    fn wait(&mut self, command_buffer: Handle) -> Answer<Value> {
        let held = (self.command_buffers.get(command_buffer.0 as usize)).ok_or_else(|| {
            Failure::named(
                "NoSuchHandle",
                format!("command buffer {}", command_buffer.0),
            )
        })?;
        Ok(match waited_for(held, WAIT) {
            Some(word) => Value::unit_variant(word),
            None => Value::Nothing(Negative::NeverReturned),
        })
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        let buffer = self.held(buffer)?;
        let past = (offset.checked_add(length as u64)).is_none_or(|end| end > buffer.length());
        if past {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{length} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        let contents = (buffer.contents())
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
        // SAFETY: `[offset, offset + length)` is inside the buffer's bytes, the host reaches
        // them, and the caller waited for the command buffer that wrote them.
        let bytes =
            unsafe { std::slice::from_raw_parts(contents.as_ptr().add(offset as usize), length) };
        Ok(bytes.to_vec())
    }

    /// The codes of the precision notes of every state the case dispatched or drew with, once.
    fn announced(&mut self) -> Vec<String> {
        let dispatched = (self.dispatched.iter())
            .filter_map(|&at| self.pipelines.get(at))
            .flat_map(rt::ComputePipelineState::precision_notes);
        let drawn = (self.drawn.iter())
            .filter_map(|&at| self.render_pipelines.get(at))
            .flat_map(rt::RenderPipelineState::precision_notes);
        let codes: BTreeSet<&'static str> = (dispatched.chain(drawn))
            .map(|note| metal2vulkan_runtime::Degradation::from_precision_note(note).code())
            .collect();
        self.dispatched.clear();
        self.drawn.clear();
        codes.into_iter().map(str::to_owned).collect()
    }

    /// The runtime's own answer to the descriptor, which states the stage input as the old
    /// runtime's side does.
    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle> {
        let function = (self.held_library(library)?.function(entry))
            .map_err(|error| self.on.translation_failure(&error))?;
        let mut descriptor = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(
            rt::FunctionDescriptor::new(function),
        ));
        descriptor.stage_input = Some(crate::runtime_side::stage_input_descriptor(stage_input));
        let on = self.on;
        let state = (on.device)
            .new_compute_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| on.failure(&error))?;
        Ok(keep(&mut self.pipelines, state))
    }

    /// One level of Shared storage in the role's usage, at the next texture index where a shader
    /// reaches it.
    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        let mut descriptor = rt::TextureDescriptor::new(
            rt::TextureType::Type2D,
            spec.format,
            spec.size[0],
            spec.size[1],
        );
        descriptor.usage = crate::runtime_side::usage_of(spec.role);
        descriptor.storage_mode = rt::StorageMode::Shared;
        let reached = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
        let index = (descriptor.usage.intersects(reached))
            .then(|| self.on.next_index(rt::IndexClass::Texture));
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let texture = (self.on.device.new_texture(&descriptor, index)).map_err(of_rt)?;
        if !bytes.is_empty() {
            let region = rt::Region {
                origin: [0, 0, 0],
                size: [spec.size[0], spec.size[1], 1],
            };
            (texture.replace_region(region, 0, 0, bytes, spec.row_pitch.into(), 0))
                .map_err(of_rt)?;
        }
        Ok(keep(&mut self.textures, texture))
    }

    /// The call the request names: the device's `newTextureWithDescriptor:`, the buffer's
    /// `newTextureWithDescriptor:offset:bytesPerRow:` or the parent's view, at the next texture
    /// index where a shader reaches the texture, then its uploads in order.
    ///
    /// The runtime's descriptor has no `allowGPUOptimizedContents`: each texture is in the
    /// device's own layout, which the flag allows, and Metal's texels are the same with it off, so
    /// the request's flag asks the runtime nothing. A view's swizzle other than each channel's own
    /// is `newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:`, which the runtime
    /// does not have; the view of the default swizzle is the call without it.
    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        use crate::texture::Over;
        let format = PixelFormat::try_from(request.format).map_err(|problem| {
            Failure::named(
                "NoSuchPixelFormat",
                format!("{}: {problem:?}", request.format),
            )
        })?;
        let [width, height, depth] = request.size;
        let mut descriptor =
            rt::TextureDescriptor::new(request.texture_type, format, width, height);
        descriptor.depth = depth.max(1);
        descriptor.mipmap_level_count = request.levels.max(1);
        descriptor.array_length = request.slices.max(1);
        descriptor.usage = request.usage;
        descriptor.storage_mode = match request.private {
            true => rt::StorageMode::Private,
            false => rt::StorageMode::Shared,
        };
        descriptor.swizzle = request.swizzle;
        let index = |usage| self.on.texture_index(usage);
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let texture = match request.over {
            None => self
                .on
                .device
                .new_texture(&descriptor, index(request.usage)),
            Some(Over::Buffer {
                buffer,
                offset,
                bytes_per_row,
            }) => (self.held(buffer)?).new_texture(
                &descriptor,
                index(request.usage),
                offset,
                bytes_per_row,
            ),
            Some(Over::View { .. }) if request.swizzle != rt::TextureSwizzleChannels::DEFAULT => {
                return Err(Failure::unavailable(
                    "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:",
                ));
            }
            Some(Over::View { texture }) => {
                let parent = self.held_texture(texture)?;
                parent.new_texture_view(
                    format,
                    request.texture_type,
                    0..request.levels.max(1),
                    0..request.slices.max(1),
                    index(parent.descriptor().usage),
                )
            }
        }
        .map_err(of_rt)?;
        for upload in request.uploads {
            let [x, y, z] = upload.origin;
            let [w, h, d] = upload.size;
            let region = rt::Region {
                origin: [x, y, z],
                size: [w, h, d.max(1)],
            };
            (texture.replace_region(
                region,
                upload.level,
                upload.slice,
                upload.bytes,
                upload.row_pitch.into(),
                upload.image_pitch.into(),
            ))
            .map_err(of_rt)?;
        }
        Ok(keep(&mut self.textures, texture))
    }

    /// The program on the runtime, as the old runtime's side runs it (`rt_side::work`).
    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        let on = self.on;
        work::run(on, self.queue()?, work)
    }

    /// At the next sampler index. Every state the runtime makes has a resource ID, so
    /// `supportArgumentBuffers` asks for nothing more.
    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let mut descriptor = rt::SamplerDescriptor::new();
        descriptor.min_filter = spec.min_filter;
        descriptor.mag_filter = spec.mag_filter;
        descriptor.mip_filter = spec.mip_filter;
        descriptor.s_address_mode = spec.s_address_mode;
        descriptor.t_address_mode = spec.t_address_mode;
        descriptor.r_address_mode = spec.r_address_mode;
        descriptor.border_color = spec.border_color;
        descriptor.normalized_coordinates = spec.normalized_coordinates;
        descriptor.lod_min_clamp = spec.lod_min_clamp;
        descriptor.lod_max_clamp = spec.lod_max_clamp;
        descriptor.lod_average = spec.lod_average;
        descriptor.max_anisotropy = spec.max_anisotropy;
        descriptor.compare_function = spec.compare_function;
        descriptor.reduction_mode = spec.reduction_mode;
        descriptor.lod_bias = spec.lod_bias;
        let index = self.on.next_index(rt::IndexClass::Sampler);
        let sampler = (self.on.device.new_sampler_state(&descriptor, index))
            .map_err(|error| Failure::of_rt(&error))?;
        Ok(keep(&mut self.samplers, sampler))
    }

    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        if !stage_in.is_empty() {
            return Err(Failure::unavailable("set_stage_in_region"));
        }
        self.encode(pipeline, bindings, |encoder| {
            encoder.dispatch_threads(grid, threads)
        })
    }

    fn target(&mut self, spec: &crate::render::TargetSpec) -> Answer<Handle> {
        self.make_target(spec)
    }

    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &crate::render::RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        self.make_render_pipeline(library, spec)
    }

    fn depth_stencil_state(&mut self, spec: &rt::DepthStencilDescriptor) -> Answer<Handle> {
        let state = self.on.device.new_depth_stencil_state(spec);
        Ok(keep(&mut self.depth_stencils, state))
    }

    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        let region = rt::Region {
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
        };
        (self
            .held_texture(texture)?
            .replace_region(region, 0, 0, bytes, row_pitch.into(), 0))
        .map_err(|error| Failure::of_rt(&error))
    }

    fn render(&mut self, work: &[crate::render::RenderWork]) -> Answer<crate::case::PassOutcome> {
        self.run_render(work)
    }

    fn compute_pass(
        &mut self,
        pass: &crate::compute_pass::ComputePass<'_>,
    ) -> Answer<crate::compute_pass::ComputePassOutcome> {
        self.run_compute_pass(pass)
    }

    /// The facts the runtime holds: the status, the four times and the work's completion. The
    /// queue's label and the command buffer's `device` and `commandQueue` are the consumer's own
    /// bookkeeping ([`CommandBufferFacts::the_consumers`]), and the resource state encoder is the
    /// sparse family's, which the runtime does not make.
    fn command_buffer_facts(&mut self) -> Answer<CommandBufferFacts> {
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let scratch = self.buffer(&[0u8; TIMED_BYTES])?;
        let target = self.held(scratch)?.clone();
        let queue = self.queue()?.clone();
        let held = queue.command_buffer().map_err(of_rt)?;
        let kernel_before = kernel_times(&held);
        let gpu_before = gpu_times(&held);
        // One blit of a few bytes of the side's own buffer: enough work that the GPU ran, and no
        // shader, draw or dispatch at all.
        let mut blit = held.blit_command_encoder().map_err(of_rt)?;
        (blit.fill_buffer(&target, 0..TIMED_BYTES as u64, 0xa5)).map_err(of_rt)?;
        blit.end_encoding();
        held.commit().map_err(of_rt)?;
        let settled = |command_buffer: &rt::CommandBuffer| {
            waited_for(command_buffer, WAIT)
                .map_or(Value::Nothing(Negative::NeverReturned), Value::unit_variant)
        };
        let status = settled(&held);
        let (kernel_after, gpu_after) = (kernel_times(&held), gpu_times(&held));
        // The command buffer the encoder would be opened on, committed with nothing in it.
        let state_buffer = queue.command_buffer().map_err(of_rt)?;
        state_buffer.commit().map_err(of_rt)?;
        let state_encoder_status = settled(&state_buffer);
        let facts = CommandBufferFacts::new(
            false,
            [false; 2],
            kernel_before,
            kernel_after,
            Ok((gpu_before, gpu_after)),
            false,
            status,
            state_encoder_status,
        );
        Ok(facts
            .the_consumers(THE_CONSUMERS)
            .without_resource_state_encoder(Failure::unavailable("resource_state_command_encoder")))
    }

    fn empty_command_buffers(&mut self) -> Answer<EmptyCommandBuffers> {
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let order: Arc<Mutex<Vec<String>>> = Arc::default();
        // A handler that sends the status its command buffer reads inside it, and records `name`
        // in `order` where one is given.
        let sender = |name: Option<&'static str>| {
            let (sent, received) = mpsc::channel();
            let order = Arc::clone(&order);
            let handler = move |own: &rt::CommandBuffer| {
                if let (Some(name), Ok(mut order)) = (name, order.lock()) {
                    order.push(name.to_owned());
                }
                let _ = sent.send(format!("{:?}", own.status()));
            };
            (handler, received)
        };
        let settled = |received: &mpsc::Receiver<String>| {
            (received.recv_timeout(ARRIVAL)).map_or(Status::NeverSettled, Status::read)
        };
        let queue = (self.on.device.new_command_queue(QUEUE_COMMAND_BUFFERS)).map_err(of_rt)?;
        // Alone on an idle queue.
        let alone = queue.command_buffer().map_err(of_rt)?;
        let (scheduled, in_scheduled) = sender(None);
        let (completed, in_completed) = sender(None);
        alone.add_scheduled_handler(scheduled).map_err(of_rt)?;
        alone.add_completed_handler(completed).map_err(of_rt)?;
        let kernel_before = kernel_times(&alone);
        let gpu_before = gpu_times(&alone);
        alone.commit().map_err(of_rt)?;
        let after_the_wait = waited(&alone);
        let statuses = [
            settled(&in_scheduled),
            settled(&in_completed),
            after_the_wait,
        ];
        let alone = Alone::new(
            statuses,
            (kernel_before, kernel_times(&alone)),
            Ok((gpu_before, gpu_times(&alone))),
        );
        // Behind a fill of the same queue.
        let block = (self
            .on
            .device
            .take_block(rt::MemoryClass::Private, FILL_BYTES))
        .map_err(of_rt)?;
        let target = (self.on.device.new_buffer(block.id, 0, FILL_BYTES, None)).map_err(of_rt)?;
        let fill = queue.command_buffer().map_err(of_rt)?;
        let mut blit = fill.blit_command_encoder().map_err(of_rt)?;
        (blit.fill_buffer(&target, 0..FILL_BYTES, 0x5a)).map_err(of_rt)?;
        blit.end_encoding();
        let (filled, _) = sender(Some(FILL));
        fill.add_completed_handler(filled).map_err(of_rt)?;
        let empty = queue.command_buffer().map_err(of_rt)?;
        // The empty one's completed handler records itself and reads the fill's status.
        let (fills_status, fill_read) = mpsc::channel();
        let (in_order, read_fill) = (Arc::clone(&order), fill.clone());
        (empty.add_completed_handler(move |_: &rt::CommandBuffer| {
            if let Ok(mut order) = in_order.lock() {
                order.push(EMPTY.to_owned());
            }
            let _ = fills_status.send(format!("{:?}", read_fill.status()));
        }))
        .map_err(of_rt)?;
        fill.commit().map_err(of_rt)?;
        empty.commit().map_err(of_rt)?;
        let after_the_wait = waited(&empty);
        let fills_status = settled(&fill_read);
        // Settled before the times are read: its handler may still be running.
        let _ = waited(&fill);
        let handler_order = order.lock().map(|order| order.clone()).unwrap_or_default();
        let behind = Behind::new(
            handler_order,
            [fills_status, after_the_wait],
            (
                seconds(empty.kernel_start_time()),
                seconds(fill.kernel_end_time()),
            ),
            Ok((gpu_times(&fill), gpu_times(&empty))),
        );
        Ok(EmptyCommandBuffers { alone, behind })
    }
}
