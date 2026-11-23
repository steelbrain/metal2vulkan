//! The runtime's side: the case vocabulary on `metal2vulkan-runtime`'s public API.
//!
//! A call this side has no answer for returns [`Failure::unavailable`] with its name, so a case
//! that needs it ends refused, its row naming the call: nothing is skipped, and nothing is answered
//! for it. The rows it refuses are held to the ledger, `known-gaps-rt.tsv`
//! ([`crate::known_gaps::ledger`]). A translator refusal that names capabilities the device offers
//! is the runtime's refusal and not the venue's lack: the runtime's floor leaves them off
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
//! state carries static precision notes ([`metal2vulkan_runtime::ComputePipelineState::precision_notes`],
//! [`metal2vulkan_runtime::RenderPipelineState::precision_notes`]), and so do a texture of a
//! block-compressed format ([`metal2vulkan_runtime::Texture::precision_note`]) and a sampler state that
//! reads an opaque-black border ([`metal2vulkan_runtime::SamplerState::precision_note`]). This side
//! announces the notes of each state its case dispatched or drew with, of each texture and
//! sampler state it made, and of each upload into a 32-bit float depth (`upload_note`), by the
//! codes the suite's allowances name (`metal2vulkan-degrade-<kind>`).

use crate::case::{
    Answer, ArgumentSlot, Bindings, Failure, Handle, SamplerSpec, ShaderSource, Side,
    StageInRegion, StageInput, TextureSpec, Venue,
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
use metal2vulkan_runtime as rt;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

mod compute;
mod descriptors;
mod events;
mod fences;
pub(crate) mod fields;
mod functions;
mod heap_sizes;
mod render;
mod state_facts;
mod states;
mod tables;
mod work;

/// How long a wait for a command buffer waits before the side records that it never returned.
const WAIT: Duration = Duration::from_secs(60);

/// The bytes the command-buffer facts' one blit fills, as the other sides fill.
const TIMED_BYTES: usize = 256;

/// The bytes the command buffer of a handler's own wait fills.
const HANDLER_FILL: usize = 64;

/// Why the command-buffer facts' queue label and getters are not this side's to answer.
const THE_CONSUMERS: &str = "a queue's label and a command buffer's device and commandQueue \
                             never reach the GPU: metal2vulkan-runtime leaves them to its consumer";

/// The command buffers a queue of this side's holds at once: `newCommandQueue`'s.
const QUEUE_COMMAND_BUFFERS: u32 = 64;

/// The runtime's device a replay runs on, with the venue it reports and the indices it hands
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
            "no {venues:?} device opens for metal2vulkan-runtime: {}",
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
    /// where it offers one, the capability is one `metal2vulkan-runtime`'s floor leaves off, a typed
    /// refusal of the runtime's until a stream uses it, and the failure is the runtime's refusal.
    pub(crate) fn failure(&self, error: &rt::Error) -> Failure {
        let mut failure = Failure::of_rt(error);
        if let rt::Error::Translation {
            error: metal2vulkan::error::Error::Unsupported(unsupported),
            ..
        } = error
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

    /// The next texture index ([`RtDevice::next_index`]) for a texture of `pf` and `usage` where a
    /// shader reaches it (shader read or write, no bit, which the runtime takes as every usage, or
    /// a colour render target, which a fragment function reads through its attachment), and
    /// `None` where none does, as the runtime's texture makers take an index.
    fn texture_index(&self, pf: rt::PixelFormat, usage: rt::TextureUsage) -> Option<u32> {
        let reached = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
        let color_target = usage.contains(rt::TextureUsage::RENDER_TARGET)
            && !metal2vulkan::metal::format::is_depth_stencil(pf);
        (usage.is_empty() || usage.intersects(reached) || color_target)
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
            "metal2vulkan-runtime on {} {}",
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
        "metal2vulkan-runtime keeps no heap object: its consumer places resources in the blocks it takes \
         and implements MTLHeap's calls and getters",
    )
}

/// The consumer's own, for a pipeline creation that asks for reflection or takes options, under
/// any option word
/// ([`ConsumerObject::PipelineReflection`](crate::known_gaps::ConsumerObject::PipelineReflection)):
/// the runtime builds no pipeline reflection and acts on no `MTLPipelineOption`, and its consumer
/// builds reflection from the library's AIR, as the runtime's README says. A word with a bit Metal
/// defines no option for is the consumer's too: Metal reflects under it as under
/// `MTLPipelineOptionBindingInfo` (the `pipeline-options-*-undefined-bit-*` recordings).
pub(crate) fn reflection_is_the_consumers() -> Failure {
    Failure::named(
        "TheConsumers",
        "metal2vulkan-runtime builds no pipeline reflection and acts on no MTLPipelineOption \
         (FailOnBinaryArchiveMiss, ArgumentInfo, BufferTypeInfo, BindingInfo): its consumer builds \
         reflection from the library's AIR",
    )
}

/// The runtime's side of one case.
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
    /// The static notes of what a pass the case encoded set beside its pipeline states: a depth
    /// bias other than 0 (`RenderCommandEncoder::DEPTH_BIAS_NOTE`), a draw of lines or a line
    /// strip and the `Lines` fill mode (`RenderCommandEncoder::LINE_NOTE`), and each
    /// depth-stencil state's (`DepthStencilState::precision_note`); and of each texture and
    /// sampler state the case made
    /// (`Texture::precision_note`, `SamplerState::precision_note`), which [`Side::announced`]
    /// answers too.
    encoder_notes: BTreeSet<rt::PrecisionNote>,
    /// Each argument buffer the case made ([`Side::argument_buffer`]), with the textures its
    /// resource IDs name, which a dispatch that binds it makes resident.
    argument_residents: Vec<(Handle, Vec<Handle>)>,
    /// Each surface a texture of the case lies over ([`Side::surface_texture`]).
    surfaces: Vec<rt::Surface>,
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
            encoder_notes: BTreeSet::new(),
            argument_residents: Vec::new(),
            surfaces: Vec::new(),
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

    /// Keep `texture` as the case's, with the static note it carries (`Texture::precision_note`:
    /// the device's decode of a block-compressed format), which [`Side::announced`] answers.
    fn keep_texture(&mut self, texture: rt::Texture) -> Handle {
        self.encoder_notes.extend(texture.precision_note());
        keep(&mut self.textures, texture)
    }

    /// `replace_region` of `bytes` into `texture`, with its static note where it reaches a 32-bit
    /// float depth ([`upload_note`]), which [`Side::announced`] answers.
    fn replace(
        &mut self,
        texture: &rt::Texture,
        (region, level, slice): (rt::Region, u32, u32),
        bytes: &[u8],
        pitches: (u64, u64),
    ) -> Answer<()> {
        (texture.replace_region(region, level, slice, bytes.into(), pitches.0, pitches.1))
            .map_err(|error| Failure::of_rt(&error))?;
        let note = upload_note(texture, rt::BlitOption::None);
        self.encoder_notes.extend(note);
        Ok(())
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
        // The textures the bound argument buffers name: Metal keeps a resource a kernel reaches
        // through an argument buffer resident only where the encoder was told so.
        let resident = (self.argument_residents.iter())
            .filter(|(buffer, _)| bindings.buffers.contains(buffer))
            .flat_map(|(_, textures)| textures.iter())
            .map(|&handle| self.held_texture(handle).cloned())
            .collect::<Answer<Vec<rt::Texture>>>()?;
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
        for texture in &resident {
            (encoder.use_resource(rt::Resource::Texture(texture), rt::ResourceUsage::READ))
                .map_err(of_rt)?;
        }
        dispatch(&mut encoder).map_err(of_rt)?;
        encoder.end_encoding();
        command_buffer.commit().map_err(of_rt)?;
        self.dispatched.insert(at);
        Ok(keep(&mut self.command_buffers, command_buffer))
    }
}

/// The static note of an upload into `texture` of `option`, where it reaches a 32-bit float depth:
/// `Depth32Float`'s, or `Depth32Float_Stencil8`'s under its depth option
/// ([`rt::Texture::FLOAT_DEPTH_UPLOAD_NOTE`]).
fn upload_note(texture: &rt::Texture, option: rt::BlitOption) -> Option<rt::PrecisionNote> {
    let pixel_format = texture.descriptor().pixel_format;
    let depth = pixel_format == PixelFormat::Depth32Float
        || (pixel_format == PixelFormat::Depth32FloatStencil8
            && option == rt::BlitOption::DepthFromDepthStencil);
    depth.then_some(rt::Texture::FLOAT_DEPTH_UPLOAD_NOTE)
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
            "metal2vulkan-runtime's TextureDescriptor states no cpuCacheMode, hazardTrackingMode or \
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
        storage_mode: storage_mode(d.storage_mode),
        cpu_cache_mode: consumers(),
        hazard_tracking_mode: consumers(),
        gpu_optimized: consumers(),
    }
}

/// `MTLStorageMode`'s raw value of `mode`.
fn storage_mode(mode: rt::StorageMode) -> u64 {
    match mode {
        rt::StorageMode::Shared => 0,
        rt::StorageMode::Managed => 1,
        rt::StorageMode::Private => 2,
        rt::StorageMode::Memoryless => 3,
    }
}

/// The consumer's own, for `MTLResource`'s getters but `storageMode`: the runtime keeps no label,
/// device, CPU cache or hazard tracking mode, option word or heap on a buffer or texture, and its
/// consumer keeps what it made each with and where it placed it.
fn resource_is_the_consumers() -> Failure {
    Failure::named(
        "TheConsumers",
        "metal2vulkan-runtime keeps no label, device, cpuCacheMode, hazardTrackingMode, resourceOptions, \
         heap, heapOffset or isAliasable of a resource: its consumer keeps what it made each with \
         and where it placed it",
    )
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

    fn compute_pipeline_with_constants(
        &mut self,
        library: Handle,
        entry: &str,
        constants: &[crate::function_facts::ConstantValue],
    ) -> Answer<Handle> {
        let function = (self.held_library(library)?.function(entry))
            .map_err(|error| self.on.translation_failure(&error))?;
        let mut function = rt::FunctionDescriptor::new(function);
        function.constants = crate::function_facts::constant_values(constants);
        let descriptor = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(function));
        let on = self.on;
        let state = (on.device)
            .new_compute_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| on.failure(&error))?;
        Ok(keep(&mut self.pipelines, state))
    }

    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let state = self.held_pipeline(pipeline)?;
        Ok(u64::from(state.static_threadgroup_memory_length()))
    }

    fn compute_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<crate::pipeline_facts::ComputeStateFacts> {
        self.compute_state_facts(state, imageblock)
    }

    fn render_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<crate::pipeline_facts::RenderStateFacts> {
        self.render_state_facts(state, imageblock)
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

    /// `newBufferWithBytesNoCopy:length:options:deallocator:` of writable pages: the runtime's
    /// import of them (`Device::import_host_memory`), whose release unmaps them once the runtime
    /// frees the memory, and a buffer over the whole block, its bytes the pages' own. Metal's
    /// private read-only bit (`MTLResourceOptions` bit 24) has no counterpart in the runtime, so
    /// that form answers its call unavailable.
    fn host_buffer(&mut self, spec: &crate::host_pages::HostBufferSpec) -> Answer<Handle> {
        if let Some(why) = spec.unmappable() {
            return Err(Failure::named("UnmappableHostBuffer", why));
        }
        if spec.read_only {
            return Err(Failure::unavailable(
                "newBufferWithBytesNoCopy:length:options:deallocator: with MTLResourceOptions bit 24",
            ));
        }
        let length = usize::try_from(spec.bytes)
            .map_err(|_| Failure::named("UnmappableHostBuffer", "more bytes than a usize"))?;
        let pages = crate::host_pages::HostPages::of(length, spec.pattern)
            .ok_or_else(|| Failure::named("MmapFailed", format!("{length} bytes")))?;
        let pointer = pages.pointer();
        let release: Box<dyn FnOnce() + Send> = Box::new(move || drop(pages));
        // SAFETY: the pages are this side's own mapping of `length` readable and writable bytes.
        // The release owns them and unmaps them only once the runtime freed the import, or at
        // once where the import fails, and nothing maps them anew meanwhile.
        let imported =
            unsafe { (self.on.device).import_host_memory(pointer, spec.bytes, Some(release)) };
        let (block, _) = imported.map_err(|error| self.on.failure(&error))?;
        let buffer = (self.on.device.new_buffer(block.id, 0, spec.bytes, None))
            .map_err(|error| self.on.failure(&error))?;
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
        self.replace(&texture, (region, 0, 0), bytes, (row_pitch.into(), 0))?;
        Ok(self.keep_texture(texture))
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
        let index = (self.on).texture_index(descriptor.pixel_format, descriptor.usage);
        let texture = (self.on.device.new_texture(&descriptor, index))
            .map_err(|error| Failure::of_rt(&error))?;
        let facts = texture_facts(&texture);
        self.keep_texture(texture);
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
        let index = (self.on).texture_index(descriptor.pixel_format, descriptor.usage);
        let made = buffer.new_texture(&descriptor, index, request.offset, request.bytes_per_row);
        let texture = made.map_err(|error| Failure::of_rt(&error))?;
        let facts = texture_facts(&texture);
        self.keep_texture(texture);
        Ok(facts)
    }

    /// The runtime's answer, held to the property the side's `heap_sizes` states before it is
    /// answered: the numbers are the device's own, which the cases do not compare with Metal's on
    /// this side.
    fn heap_texture_size_and_align(
        &mut self,
        query: &crate::texture_query::HeapTextureQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let descriptor = queried(query);
        let sized = (self.on.device.heap_texture_size_and_align(&descriptor))
            .map_err(|error| Failure::of_rt(&error))?;
        let queue = self.queue()?.clone();
        heap_sizes::hold(self.on, &queue, query, &descriptor, sized)?;
        Ok(crate::texture_query::SizeAndAlign {
            size: sized.size,
            align: sized.align,
        })
    }

    /// The runtime's own answer, which reads none of the query's option bits.
    fn heap_buffer_size_and_align(
        &mut self,
        query: &crate::device_queries::HeapBufferQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let sized = (self.on.device).heap_buffer_size_and_align(query.length, query.options);
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
        let index = (self.on).texture_index(descriptor.pixel_format, descriptor.usage);
        let texture = (self.on.device.new_texture(&descriptor, index))
            .map_err(|error| Failure::of_rt(&error))?;
        self.encoder_notes.extend(texture.precision_note());
        Ok(true)
    }

    /// `newHeapWithDescriptor:` as the runtime's consumer makes a heap: a block of the heap's size
    /// and storage. The descriptor's CPU cache and hazard tracking modes are the consumer's own,
    /// which the runtime's blocks state neither of.
    fn heap(&mut self, spec: &crate::heap_query::HeapSpec) -> Answer<Handle> {
        if let Some(why) = spec.is_well_formed() {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let (block, range) = work::heap_of(self.on, spec.size, spec.storage)?;
        let placement = matches!(spec.heap_type, crate::values::HeapType::Placement);
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

    /// The consumer's own but for `storageMode`, which [`Side::resource_storage_mode`] answers.
    fn resource_facts(
        &mut self,
        _of: crate::heap_query::QueriedResource,
        _heap: Option<Handle>,
    ) -> Answer<crate::heap_query::ResourceFacts> {
        Err(resource_is_the_consumers())
    }

    /// A texture's or view's storage mode as the runtime made it, and a buffer's as the class of
    /// the block it lies in.
    fn resource_storage_mode(&mut self, of: crate::heap_query::QueriedResource) -> Answer<u64> {
        use crate::heap_query::QueriedResource;
        Ok(match of {
            QueriedResource::Texture(handle) => {
                storage_mode(self.held_texture(handle)?.descriptor().storage_mode)
            }
            QueriedResource::Buffer(handle) => {
                let (_, block) = (self.buffers.get(handle.0 as usize)).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("buffer {}", handle.0))
                })?;
                match block.class {
                    rt::MemoryClass::Shared => storage_mode(rt::StorageMode::Shared),
                    rt::MemoryClass::Private => storage_mode(rt::StorageMode::Private),
                }
            }
        })
    }

    /// A heap's `newTextureWithDescriptor:`, which places the texture where the heap's own
    /// suballocation puts it: the consumer's own, as an automatic heap's buffer is. The runtime's
    /// `Device::new_heap_texture` binds a texture at the offset the consumer chose.
    fn heap_texture(
        &mut self,
        heap: Handle,
        _spec: &crate::sparse_query::SparseTextureSpec,
    ) -> Answer<Handle> {
        self.held_heap(heap)?;
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
        _state: crate::values::PurgeableState,
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
            "metal2vulkan-runtime's RenderPassDescriptor is a value its consumer fills: what Metal's \
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
        Err(reflection_is_the_consumers())
    }

    /// The consumer's own (`reflection_is_the_consumers`).
    fn compute_pipeline_under_options(
        &mut self,
        _library: Handle,
        _entry: &str,
        _options: u64,
    ) -> Answer<crate::pipeline_options::OptionOutcome> {
        Err(reflection_is_the_consumers())
    }

    /// The consumer's own (`reflection_is_the_consumers`).
    fn render_pipeline_under_options(
        &mut self,
        _library: Handle,
        _vertex: &str,
        _fragment: &str,
        _attachment: PixelFormat,
        _options: u64,
    ) -> Answer<crate::pipeline_options::OptionOutcome> {
        Err(reflection_is_the_consumers())
    }

    /// The consumer's own (`reflection_is_the_consumers`).
    fn compute_pipeline_reflection_under(
        &mut self,
        _library: Handle,
        _entry: &str,
        _options: u64,
    ) -> Answer<crate::pipeline_option_reflection::UnderOptions> {
        Err(reflection_is_the_consumers())
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
        Err(match *form {
            Form::ComputeFunction { .. } | Form::ComputeDescriptor { .. } => {
                reflection_is_the_consumers()
            }
            Form::Render { .. } | Form::Tile { .. } | Form::Mesh { .. } => {
                Failure::unavailable("pipeline_both_paths")
            }
        })
    }

    fn read_write_texture_tier(&mut self) -> Answer<metal2vulkan::metal::ReadWriteTextureTier> {
        Ok(self.on.device.read_write_texture_support())
    }

    fn argument_buffers_tier(&mut self) -> Answer<metal2vulkan::metal::ArgumentBuffersTier> {
        Ok(self.on.device.argument_buffers_support())
    }

    fn minimum_linear_texture_alignment(&mut self, format: PixelFormat) -> Answer<u64> {
        (self.on.device.minimum_linear_texture_alignment(format))
            .map_err(|error| Failure::of_rt(&error))
    }

    /// The runtime's two view creators: `newTextureViewWithPixelFormat:` is the four-argument call
    /// over the whole texture the request names, and the swizzle's two selectors, whose
    /// `MTLTextureViewDescriptor` holds exactly the five values, are its swizzled call. A view a
    /// shader reaches takes the next texture index, as the texture it is made of did.
    fn texture_view(
        &mut self,
        of: Handle,
        request: &crate::texture_views::ViewRequest,
    ) -> Answer<Handle> {
        use crate::texture_views::ViewCall;
        let texture = self.held_texture(of)?;
        let range = |first: u64, count: u64, what: &str| -> Answer<std::ops::Range<u32>> {
            let first = work::narrow(first, what)?;
            Ok(first..first + work::narrow(count, what)?)
        };
        let levels = range(request.first_level, request.level_count, "a level")?;
        let slices = range(request.first_slice, request.slice_count, "a slice")?;
        let (format, texture_type) = (request.format, request.texture_type);
        let index = self.on.texture_index(format, texture.descriptor().usage);
        let view = match request.call {
            ViewCall::PixelFormat | ViewCall::LevelsSlices => {
                texture.new_texture_view(format, texture_type, levels, slices, index)
            }
            ViewCall::Swizzled | ViewCall::Descriptor => texture.new_texture_view_with_swizzle(
                format,
                texture_type,
                levels,
                slices,
                request.swizzle,
                index,
            ),
        }
        .map_err(|error| Failure::of_rt(&error))?;
        Ok(self.keep_texture(view))
    }

    /// The view's shape and swizzle, read of its descriptor; its parent's facts are the
    /// consumer's own
    /// ([`ConsumerObject::TextureViewParents`](crate::known_gaps::ConsumerObject::TextureViewParents)):
    /// the runtime's view keeps the texture whose image it reads and no getter, and makes no
    /// shareable texture.
    fn texture_view_facts(
        &mut self,
        texture: Handle,
        parent: Option<Handle>,
    ) -> Answer<crate::texture_views::ViewFacts> {
        let _ = parent;
        let d = self.held_texture(texture)?.descriptor();
        Ok(crate::texture_views::ViewFacts {
            parent: Err(Failure::named(
                "TheConsumers",
                "metal2vulkan-runtime keeps no parentTexture, parentRelativeLevel, \
                 parentRelativeSlice or shareable: its consumer keeps the texture it made a view \
                 of and what it shared",
            )),
            swizzle: crate::texture_views::swizzle_raw(d.swizzle),
            texture_type: u64::from(d.texture_type.as_raw()),
            pixel_format: u64::from(d.pixel_format.as_raw()),
            mipmap_level_count: u64::from(d.mipmap_level_count),
            array_length: u64::from(d.array_length),
        })
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
        let encoder = self.encoder_notes.iter().copied();
        let codes: BTreeSet<&'static str> = (dispatched.chain(drawn).chain(encoder))
            .map(|note| crate::degradation::Degradation::from_precision_note(note).code())
            .collect();
        self.dispatched.clear();
        self.drawn.clear();
        self.encoder_notes.clear();
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
        descriptor.stage_input = Some(fields::stage_input_descriptor(stage_input));
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
        descriptor.usage = fields::usage_of(spec.role);
        descriptor.storage_mode = rt::StorageMode::Shared;
        let index = (self.on).texture_index(descriptor.pixel_format, descriptor.usage);
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let texture = (self.on.device.new_texture(&descriptor, index)).map_err(of_rt)?;
        if !bytes.is_empty() {
            let region = rt::Region {
                origin: [0, 0, 0],
                size: [spec.size[0], spec.size[1], 1],
            };
            self.replace(&texture, (region, 0, 0), bytes, (spec.row_pitch.into(), 0))?;
        }
        Ok(self.keep_texture(texture))
    }

    /// The call the request names: the device's `newTextureWithDescriptor:`, the buffer's
    /// `newTextureWithDescriptor:offset:bytesPerRow:` or the parent's view, at the next texture
    /// index where a shader reaches the texture, then its uploads in order.
    ///
    /// The runtime's descriptor has no `allowGPUOptimizedContents`: each texture is in the
    /// device's own layout, which the flag allows, and Metal's texels are the same with it off, so
    /// the request's flag asks the runtime nothing. A view is
    /// `newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:` with the request's
    /// swizzle, the default one's the call without it.
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
        let pf = descriptor.pixel_format;
        let index = |usage| self.on.texture_index(pf, usage);
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
            Some(Over::View { texture }) => {
                let parent = self.held_texture(texture)?;
                parent.new_texture_view_with_swizzle(
                    format,
                    request.texture_type,
                    0..request.levels.max(1),
                    0..request.slices.max(1),
                    request.swizzle,
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
            let at = (region, upload.level, upload.slice);
            let pitches = (upload.row_pitch.into(), upload.image_pitch.into());
            self.replace(&texture, at, upload.bytes, pitches)?;
        }
        Ok(self.keep_texture(texture))
    }

    /// The program on the runtime (`rt_side::work`); a
    /// program that generates mipmaps announces the blit encoder's static note of them, and one
    /// that makes a texture its static note.
    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        let on = self.on;
        self.queue()?;
        let queue = self.queue.as_ref().expect("made above");
        let done = work::run(on, queue, work, &mut self.encoder_notes)?;
        let mipmaps =
            |step: &crate::work::Step| matches!(step, crate::work::Step::GenerateMipmaps { .. });
        if work.steps.iter().any(mipmaps) {
            (self.encoder_notes).insert(rt::BlitCommandEncoder::GENERATE_MIPMAPS_NOTE);
        }
        Ok(done)
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
        self.encoder_notes.extend(sampler.precision_note());
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
        /// One region call: a region, or the buffer and offset an indirect one reads.
        enum Set {
            Direct(rt::Region),
            Indirect(rt::Buffer, u64),
        }
        let mut regions = Vec::with_capacity(stage_in.len());
        for region in stage_in {
            regions.push(match *region {
                StageInRegion::Direct { origin, size } => Set::Direct(rt::Region { origin, size }),
                StageInRegion::Indirect { buffer, offset } => {
                    Set::Indirect(self.held(buffer)?.clone(), offset)
                }
            });
        }
        self.encode(pipeline, bindings, |encoder| {
            for region in &regions {
                match region {
                    Set::Direct(region) => encoder.set_stage_in_region(*region),
                    Set::Indirect(buffer, offset) => {
                        encoder.set_stage_in_region_with_indirect_buffer(buffer, *offset)?
                    }
                }
            }
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
        let texture = self.held_texture(texture)?.clone();
        self.replace(&texture, (region, 0, 0), bytes, (row_pitch.into(), 0))
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

    /// A wait made from a command buffer's own handler: a
    /// command buffer with one small fill, whose handler `wait.from` names makes the wait
    /// `wait.wait` names on that same command buffer, and sends back what the wait answered. Both
    /// waits here are bounded, so a wait the runtime neither refused nor returned from is recorded
    /// as never returning instead of hanging the suite: the handler would still be inside it and
    /// the command buffer would never complete.
    fn handler_wait(&mut self, wait: crate::handler_waits::OwnWait) -> Answer<Value> {
        use crate::handler_waits::{HandlerKind, WaitKind};
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let filled = self.buffer(&[0u8; HANDLER_FILL])?;
        let target = self.held(filled)?.clone();
        let queue = self.queue()?.clone();
        let command_buffer = queue.command_buffer().map_err(of_rt)?;
        let mut blit = command_buffer.blit_command_encoder().map_err(of_rt)?;
        (blit.fill_buffer(&target, 0..HANDLER_FILL as u64, 0x11)).map_err(of_rt)?;
        blit.end_encoding();
        let (sender, answer) = mpsc::channel();
        let kind = wait.wait;
        let handler = move |own: &rt::CommandBuffer| {
            let answered = match kind {
                WaitKind::UntilCompleted => own.wait_until_completed(),
                WaitKind::UntilScheduled => own.wait_until_scheduled(),
            };
            let _ = sender.send(answered.err().map(|error| Failure::of_rt(&error)));
        };
        match wait.from {
            HandlerKind::Scheduled => command_buffer.add_scheduled_handler(handler),
            HandlerKind::Completed => command_buffer.add_completed_handler(handler),
        }
        .map_err(of_rt)?;
        command_buffer.commit().map_err(of_rt)?;
        if waited_for(&command_buffer, WAIT).is_none() {
            return Ok(Value::Nothing(Negative::NeverReturned));
        }
        match answer.recv_timeout(WAIT) {
            Ok(None) => Ok(Value::ok(Value::Absent)),
            Ok(Some(failure)) => Err(failure),
            Err(_) => Ok(Value::Nothing(Negative::NeverReturned)),
        }
    }

    fn event_cross_queue_wait(&mut self) -> Answer<crate::event_orders::CrossQueueWait> {
        self.cross_queue_wait()
    }

    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        self.highest_signal(values)
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        // The runtime keeps no label to set: the side states none, and refuses its read-back.
        let _ = label;
        self.event_rise()
    }

    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        self.held_command_buffer_probe()
    }

    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        self.signal_before_wait()
    }

    fn event_wait_behind_a_running_signal(
        &mut self,
        shared: bool,
    ) -> Answer<crate::event_orders::WaitBehindRunningSignal> {
        self.wait_behind_a_running_signal(shared)
    }

    fn fence_order(
        &mut self,
        shape: crate::fence_orders::FenceShape,
    ) -> Answer<crate::fence_orders::FenceOrder> {
        self.fence_probe(shape)
    }

    fn scheduled_handler_order(
        &mut self,
    ) -> Answer<crate::scheduled_handlers::ScheduledHandlerOrder> {
        self.scheduled_handler_order_probe()
    }

    fn compute_pipeline_limits(
        &mut self,
        library: Handle,
        spec: &crate::pipeline_limits::ComputeLimitsSpec<'_>,
    ) -> Answer<crate::pipeline_limits::Creation<crate::pipeline_limits::ComputeStateLimits>> {
        self.compute_limits(library, spec)
    }

    fn compute_pipeline_for_commands(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: Option<&StageInput>,
    ) -> Answer<Handle> {
        self.compute_state_for_commands(library, entry, stage_input)
    }

    fn compute_pipeline_with_descriptor(
        &mut self,
        library: Handle,
        spec: &crate::pipeline_facts::ComputeStateSpec<'_>,
    ) -> Answer<Handle> {
        self.described_compute(library, spec)
    }

    fn render_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::render_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::render_descriptors::RenderDescriptorFacts> {
        self.render_descriptor(library, spec)
    }

    fn compute_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::compute_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::compute_descriptors::ComputeDescriptorFacts> {
        self.compute_descriptor(library, spec)
    }

    fn function_facts(
        &mut self,
        library: Handle,
        entry: &str,
        constants: &[crate::function_facts::ConstantValue],
    ) -> Answer<crate::function_facts::FunctionFacts> {
        self.function_facts_of(library, entry, constants)
    }

    fn library_facts(&mut self, library: Handle) -> Answer<crate::library_facts::LibraryFacts> {
        self.library_facts_of(library)
    }

    fn function_facts_from_descriptor(
        &mut self,
        library: Handle,
        descriptor: &crate::library_facts::FunctionDescriptorSpec,
    ) -> Answer<crate::function_facts::FunctionFacts> {
        self.described_function_facts(library, descriptor)
    }

    fn queue_slots(
        &mut self,
        max: Option<u32>,
        cap: u32,
    ) -> Answer<crate::queue_slots::QueueSlots> {
        self.slots(max, cap)
    }

    fn fragment_fill(
        &mut self,
        fill: &crate::case::FragmentFill<'_>,
    ) -> Answer<crate::case::FragmentFilled> {
        self.fill(fill)
    }

    fn encoded_range(
        &mut self,
        run: &crate::case::EncodedRange<'_>,
    ) -> Answer<crate::case::FragmentFilled> {
        self.execute_range(run)
    }

    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        self.mesh_target(width, height)
    }

    fn visible_function_table(
        &mut self,
        caller: Handle,
        functions: Handle,
        spec: &crate::visible_tables::TableSpec<'_>,
    ) -> Answer<crate::visible_tables::TableOutcome> {
        self.table(caller, functions, spec)
    }

    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let mut words = Vec::with_capacity(slots.len() * 8);
        let mut named = Vec::new();
        for slot in slots {
            let id = match *slot {
                ArgumentSlot::Texture(handle) => {
                    named.push(handle);
                    self.held_texture(handle)?.gpu_resource_id()
                }
                ArgumentSlot::Sampler(handle) => self.held_sampler(handle)?.gpu_resource_id(),
            };
            words.extend(id.to_bits().to_le_bytes());
        }
        let buffer = self.buffer(&words)?;
        self.argument_residents.push((buffer, named));
        Ok(buffer)
    }

    /// An IOSurface as the runtime's consumer makes one: an exported surface of the case's planes
    /// (`Device::new_exported_surface`), each a texture of its format and extent that shaders
    /// read, by the case's route ([`SurfaceRoute`](crate::case::SurfaceRoute)): VA-API's, the
    /// first of a dma-buf, an opaque fd and an NT handle the device exports at 256-byte pitch and
    /// start multiples; NVIDIA's, the first of an opaque fd and an NT handle at a pitch multiple
    /// of 4 and a start multiple of 1. The layout is the runtime's rule's. The case's texture is its plane's,
    /// at an index of its own; the other planes are reached through views alone. The case reads
    /// the texels alone, so the runtime's layout is not the case's to state.
    fn surface_texture(&mut self, spec: &crate::case::SurfaceTexture) -> Answer<Handle> {
        let at = spec.plane as usize;
        if at >= spec.planes.len() {
            return Err(Failure::named(
                "NoSuchPlane",
                format!("plane {at} of a surface of {}", spec.planes.len()),
            ));
        }
        let index = self.on.next_index(rt::IndexClass::Texture);
        let planes: Vec<rt::PlaneSpec> = (spec.planes.iter().enumerate())
            .map(|(plane, laid)| {
                let [width, height] = laid.size;
                let mut descriptor =
                    rt::TextureDescriptor::new(rt::TextureType::Type2D, laid.format, width, height);
                descriptor.usage = rt::TextureUsage::SHADER_READ;
                rt::PlaneSpec {
                    descriptor,
                    index: (plane == at).then_some(index),
                    rows_below: 0,
                }
            })
            .collect();
        let on = self.on;
        let (handles, pitch_multiple, start_multiple) = match spec.route {
            crate::case::SurfaceRoute::VaApi => (
                &[
                    rt::ExportHandle::DmaBuf,
                    rt::ExportHandle::OpaqueFd,
                    rt::ExportHandle::OpaqueWin32,
                ][..],
                256,
                256,
            ),
            crate::case::SurfaceRoute::Nvidia => (
                &[rt::ExportHandle::OpaqueFd, rt::ExportHandle::OpaqueWin32][..],
                4,
                1,
            ),
        };
        let make = |handle| {
            on.device.new_exported_surface(&rt::SurfaceSpec {
                handle,
                pitch_multiple,
                start_multiple,
                planes: &planes,
            })
        };
        let unexported = |answer: &Result<(rt::Surface, rt::SurfaceExport), rt::Error>| {
            matches!(
                answer,
                Err(rt::Error::Limit(
                    rt::Limit::NoExport { .. } | rt::Limit::ExportType { .. }
                ))
            )
        };
        let in_place = |answer: &Result<(rt::Surface, rt::SurfaceExport), rt::Error>| match answer {
            Ok((_, export)) => export.form == rt::SurfaceForm::InPlace,
            Err(_) => false,
        };
        // The handles in turn, as a consumer takes the route its device lays in place: the first
        // surface made in place is kept, and where none is, the first made at all, in the copy
        // form. A handle the platform or the device does not export is passed over. Measured: the
        // RX 550's dma-buf surfaces took the copy form and its opaque fd's were in place (53627),
        // and the RTX 3090's two-plane surfaces took the copy form under both (53626).
        let mut answer = make(handles[0]);
        for &handle in &handles[1..] {
            if in_place(&answer) || (answer.is_err() && !unexported(&answer)) {
                break;
            }
            let next = make(handle);
            if in_place(&next) || answer.is_err() {
                answer = next;
            }
        }
        let (surface, _export) = answer.map_err(|error| on.failure(&error))?;
        let texture = surface.planes()[at].clone();
        self.surfaces.push(surface);
        Ok(self.keep_texture(texture))
    }

    /// A texture over the buffer itself, the buffer's `newTextureWithDescriptor:offset:bytesPerRow:`
    /// of one 2D level at its first plane's offset and pitch, readable and writable by shaders, as
    /// the old side makes it. A texture over a surface's plane the buffer stands for is one of
    /// Metal's two-plane YCbCr formats, which converts the codes of both planes as it reads, and
    /// the runtime has no texture of one; another format over a surface's plane is
    /// [`Side::surface_texture`]'s.
    fn plane_texture(
        &mut self,
        buffer: Handle,
        over: &crate::case::PlaneTexture,
    ) -> Answer<Handle> {
        if over.ostype.is_some() {
            return Err(Failure::unavailable(
                "newTextureWithDescriptor:iosurface:plane: of a two-plane YCbCr format",
            ));
        }
        let format = rt::PixelFormat::try_from(over.format).map_err(|problem| {
            Failure::named("NoSuchPixelFormat", format!("{}: {problem}", over.format))
        })?;
        let [width, height] = over.size;
        let mut descriptor =
            rt::TextureDescriptor::new(rt::TextureType::Type2D, format, width, height);
        descriptor.usage = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
        let index = (self.on).texture_index(descriptor.pixel_format, descriptor.usage);
        let (offset, bytes_per_row) = over.planes[0];
        let texture = (self
            .held(buffer)?
            .new_texture(&descriptor, index, offset, bytes_per_row))
        .map_err(|error| self.on.failure(&error))?;
        Ok(keep(&mut self.textures, texture))
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

/// Each call this side answers on the runtime is the runtime's answer: replayed here, a case that
/// reaches one records no refusal of it as unavailable.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod answered {
    use crate::device::Venues;
    use crate::transcript::{Node, Value};

    const VENUE: Venues = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };

    /// Each call the side answers, with a case that reaches it.
    const CALLS: [(&str, &str); 20] = [
        (
            "heap_buffer_size_and_align",
            crate::cases::device_queries::HEAP_BUFFERS,
        ),
        (
            "fence_order",
            crate::cases::fence_orders::TRACKED_BEHIND_BUSY,
        ),
        (
            "event_cross_queue_wait",
            crate::cases::event_orders::CROSS_QUEUE_WAIT,
        ),
        (
            "event_highest_signal",
            crate::cases::event_orders::HIGHEST_SIGNAL,
        ),
        (
            "held_command_buffer",
            crate::cases::event_orders::HELD_COMMAND_BUFFER,
        ),
        (
            "signal_before_a_wait",
            crate::cases::event_orders::SIGNAL_BEFORE_WAIT,
        ),
        (
            "event_wait_behind_a_running_signal",
            crate::cases::event_orders::WAIT_BEHIND_A_RUNNING_SIGNAL,
        ),
        (
            "scheduled_handler_order",
            crate::cases::scheduled_handlers::SCHEDULED_HANDLER_ORDER,
        ),
        ("compute_pipeline_limits", "compute-limits-plain-max-96"),
        (
            "compute_pipeline_for_commands",
            crate::cases::indirect_commands::COMPUTE_EXECUTED,
        ),
        ("queue_slots", crate::cases::command_queues::QUEUE_SLOTS),
        ("fragment_fill", "rgb10a2-srgb-stores-f"),
        ("visible_function_table", "visible-tables-compute-size-0"),
        ("visible_function_table", "visible-tables-stage-vertex"),
        ("argument_buffer", "half-storage-writes-r16f-held"),
        ("argument_buffer", "read-write-tier-held"),
        (
            "compute_pipeline_with_descriptor",
            "pipeline-facts-compute-plain",
        ),
        ("plane_texture", "buffer-tails-kernels"),
        (
            "heap_texture",
            crate::cases::texture_descriptor_pairs::HEAP_TEXTURES,
        ),
        (
            "newBufferWithBytesNoCopy",
            "host-page-writes-kernel-store-writable",
        ),
    ];

    /// The calls `value` records as unavailable, wherever in it they sit.
    fn unavailable(value: &Value, calls: &mut Vec<String>) {
        match value {
            Value::Variant { name, payload } if name == "Unavailable" => {
                if let Value::Fields(fields) = payload.as_ref() {
                    if let Some(Value::Lines(call)) = fields.get("call") {
                        calls.extend(call.iter().cloned());
                    }
                }
            }
            Value::Variant { payload, .. } => unavailable(payload, calls),
            Value::List(values) => values.iter().for_each(|held| unavailable(held, calls)),
            Value::Fields(fields) => fields.values().for_each(|held| unavailable(held, calls)),
            _ => {}
        }
    }

    /// The calls a site and the sites under it record as unavailable.
    fn unavailable_under(node: &Node, calls: &mut Vec<String>) {
        unavailable(&node.observed.value, calls);
        for child in &node.children {
            unavailable_under(child, calls);
        }
    }

    #[test]
    fn a_call_the_side_answers_is_never_unavailable() {
        let names: Vec<String> = CALLS.iter().map(|(_, case)| (*case).to_owned()).collect();
        let cases = crate::run::selected(&names).unwrap();
        let (report, _) = crate::run::replay_rt_on_under(VENUE, &cases, 1).unwrap();
        for (call, case) in CALLS {
            let row = (report.rows().iter())
                .find(|row| row.case == case)
                .unwrap_or_else(|| panic!("{case} gave no row: {:?}", report.errors()));
            let mut calls = Vec::new();
            unavailable_under(&row.ours.root, &mut calls);
            assert!(
                !calls.iter().any(|named| named == call),
                "{case} recorded {call} as unavailable"
            );
        }
    }
}
