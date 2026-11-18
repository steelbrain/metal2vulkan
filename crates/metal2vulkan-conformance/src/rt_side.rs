//! The new runtime's side: the case vocabulary on `metal2vulkan-rt`'s public API.
//!
//! The vocabulary is ported one family at a time, as the old runtime's side was. A call this side
//! does not answer yet answers [`Failure::unavailable`] with its name, so a case that needs it
//! ends refused, its row naming the call: nothing is skipped, and nothing is answered for it. The
//! rows it refuses are held to their own ledger, `known-gaps-rt.tsv`
//! ([`crate::known_gaps::rt_ledger`]). The old runtime's side ([`crate::runtime_side`]) is
//! untouched by this one.
//!
//! The runtime places a buffer where its consumer says, in a block the consumer took; this side
//! takes a block of the buffer's own length for each buffer, Shared, as a case's buffers are, and
//! the runtime zeroes it as `newBufferWithLength:options:` does. It grants itself every texture and
//! sampler index once the device is made ([`RtDevice::open`]), and gives each texture a shader
//! reaches the next texture index and each constexpr sampler the index the first state that
//! numbered its state words took, as Metal gives an equal one the same ID.
//!
//! The runtime announces no degrade notice: where a kernel's answers may differ from Metal's, its
//! state carries static precision notes ([`metal2vulkan_rt::ComputePipelineState::precision_notes`]).
//! This side announces the notes of each state its case dispatched, by the codes the suite's
//! allowances name (`metal2vulkan-degrade-<kind>`).

use crate::case::{
    Answer, Bindings, Failure, Handle, ShaderSource, Side, StageInRegion, StageInput, Venue,
};
use crate::device::Venues;
use crate::empty_command_buffers::{
    Alone, Behind, EmptyCommandBuffers, ARRIVAL, EMPTY, FILL, FILL_BYTES,
};
use crate::event_orders::Status;
use crate::transcript::{Negative, Value};
use ash::vk;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_rt as rt;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

/// How long a wait for a command buffer waits before the side records that it never returned:
/// the old runtime's side's bound.
const WAIT: Duration = Duration::from_secs(60);

/// The command buffers a queue of this side's holds at once: `newCommandQueue`'s.
const QUEUE_COMMAND_BUFFERS: u32 = 64;

/// The new runtime's device a replay runs on, with the venue it reports and the indices it hands
/// out.
pub struct RtDevice {
    device: rt::Device,
    venue: Venue,
    indices: RefCell<Indices>,
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
    /// The pipelines the case dispatched, whose precision notes [`Side::announced`] answers.
    dispatched: BTreeSet<usize>,
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
            dispatched: BTreeSet::new(),
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
        if !bindings.samplers.is_empty() {
            return Err(Failure::unavailable("set_sampler_state"));
        }
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
/// `bound`. The runtime's wait has no bound, so it runs on a thread of its own, which a wait that
/// never returns leaves behind.
fn waited_for(command_buffer: &rt::CommandBuffer, bound: Duration) -> Option<String> {
    let (sent, received) = mpsc::channel();
    let waiting = command_buffer.clone();
    std::thread::spawn(move || {
        waiting.wait_until_completed();
        let _ = sent.send(format!("{:?}", waiting.status()));
    });
    received.recv_timeout(bound).ok()
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
            .map_err(|error| Failure::of_translation(&error))?;
        Ok(keep(&mut self.libraries, library))
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let function = (self.held_library(library)?.function(entry))
            .map_err(|error| Failure::of_translation(&error))?;
        let descriptor = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(
            rt::FunctionDescriptor::new(function),
        ));
        let on = self.on;
        let state = (on.device)
            .new_compute_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| Failure::of_rt(&error))?;
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

    /// The codes of the precision notes of every state the case dispatched, once.
    fn announced(&mut self) -> Vec<String> {
        let codes: BTreeSet<&'static str> = (self.dispatched.iter())
            .filter_map(|&at| self.pipelines.get(at))
            .flat_map(rt::ComputePipelineState::precision_notes)
            .map(|note| metal2vulkan_runtime::Degradation::from_precision_note(note).code())
            .collect();
        self.dispatched.clear();
        codes.into_iter().map(str::to_owned).collect()
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        _: Handle,
        _: &str,
        _: &StageInput,
    ) -> Answer<Handle> {
        Err(Failure::unavailable("compute_pipeline_with_stage_input"))
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
