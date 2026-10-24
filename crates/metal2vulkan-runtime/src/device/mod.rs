//! `Device` (`MTLDevice`).
//!
//! A runtime device owns or adopts the Vulkan instance, physical device, `VkDevice` and its
//! queue, and holds what every object made from it shares: the compile-and-plan layer's
//! [`metal2vulkan::device::Device`] (built from the description restricted to what was actually
//! enabled), the memory sub-allocator, the Metal address space, the scheduler, the queues'
//! waiters, and the handler threads.
//!
//! - **Create** ([`Device::system_default`], [`Device::all`], `create.rs`) makes the instance and
//!   one device per physical device with the description's recommended enables, then adopts
//!   them.
//! - **Adopt** ([`Device::from_raw`], `adopt.rs`) takes the consumer's instance, physical
//!   device, device and queue family with the [`EnableSet`] it enabled. The consumer keeps
//!   ownership: dropping the runtime device destroys only what the runtime made.
//! - Every `new_*` constructor lives with its object (`buffer.rs` has [`Device::new_buffer`],
//!   `queue.rs` [`Device::new_command_queue`], and so on).

mod adopt;
pub(super) mod argument_buffers;
mod compiles;
pub(super) mod create;
pub(super) mod float_tininess;
pub(super) mod functions;
mod handles;
pub(super) mod instance_requirements;
pub(super) mod keeper;
pub(super) mod prime;
pub(super) mod properties;
pub(crate) mod relay;
mod staging;
pub(super) mod submit;

pub use adopt::RawDevice;
pub use create::{
    DebugMessage, DeviceOptions, LayerSetting, LayerSettingValue, MessageKind, MessageSeverity,
    MessageSink,
};
pub use handles::DeviceHandles;
pub use properties::DeviceKind;

use super::address::AddressSpace;
use super::attachmentless::KeptPasses;
use super::capabilities::DeviceProperties;
use super::color_resolve::ColorResolveObjects;
use super::command_buffer::fences::Fences;
use super::command_buffer::{Completion, Item};
use super::contexts::{Contexts, THREADS};
use super::degradation::{Announced, Degradation};
use super::depth_copy::DepthCopyObjects;
use super::depth_stencil_resolve::DepthStencilResolveObjects;
use super::draw_source::{sized_index_bind, SizedIndexBind};
use super::dynamic_state::{ColorWriteMasks, ExtendedDynamicState};
use super::error::{vulkan, Error};
use super::exit_guard::{self, Call};
use super::freed_memory::Parking;
use super::function_table::TableShared;
use super::generated_commands::Commands;
use super::gpu_pools::GpuPools;
use super::gpu_time::{self, FromHost, Timing};
use super::heap_sets::{DeviceHeaps, NanResidue};
use super::host_memory::Importer;
use super::interop::Interop;
use super::loader_word::LoaderWord;
use super::memory::Memory;
use super::memory_log::MemoryLog;
use super::mipmap_scratch::MipmapScratch;
use super::owned_kernel::OwnedKernels;
use super::recycle::{self, BufferMaking, Recycler};
use super::released_programs::Keepers;
use super::render_state::{depth_bias_command, DepthBiasCommand, DepthRanges};
use super::render_target::KeptRenderPasses;
use super::resource_ids::{ResourceIds, SamplerIdReads};
use super::sample_planes::PlanesObjects;
use super::sampler::{SamplerCount, SamplerKey, SamplerShared};
use super::scheduler::Scheduler;
use super::shader_cache::{CacheSetup, ShaderCache};
use super::sharing::Sharing;
use super::spares::Spares;
use super::sparse::device::SparseDevice;
use super::stencil_copy::StencilCopyObjects;
use super::texture::TextureState;
use super::unique_identifier::UniqueIdentifiers;
use crate::image_barrier::MemoryReads;
use crate::recorder::{Command, MeshDraws, Recorder, RecorderShared, Slot};
use crate::release;
use ash::vk::{self, Handle};
use functions::{Functions, Vk};
use metal2vulkan::bind::RegionLimits;
use metal2vulkan::device::{DeviceDescription, EnableSet};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::DriverId;
use relay::Flights;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Instant;
use submit::{Internal, OwnTimeline};

/// A Metal device (`MTLDevice`) on a Vulkan device. Cheap to clone (a retain); `Send + Sync`.
#[derive(Clone)]
pub struct Device(Arc<DeviceShared>);

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.0.properties.name)
            .field("registry_id", &self.registry_id())
            .finish()
    }
}

/// The Vulkan instance a device came from.
pub(super) struct InstanceShared {
    /// Keeps the loader (and its function pointers) alive.
    entry: ash::Entry,
    pub(super) raw: ash::Instance,
    /// The `VkApplicationInfo::apiVersion` the instance was created with.
    pub(super) api_version: metal2vulkan::vk::ApiVersion,
    /// The instance extensions the runtime created it with; none for an adopted instance.
    pub(super) extensions: Vec<std::ffi::CString>,
    /// The runtime created the instance and destroys it with its last device.
    owned: bool,
    messenger: Option<create::Messenger>,
    /// The relay's messenger, on an instance the runtime created with `VK_EXT_debug_utils`
    /// (`relay.rs`).
    relay: Option<relay::Messenger>,
    /// The SHA-256 of the layer settings the runtime chained at the instance's creation (none for
    /// an adopted instance): part of a making's pipeline cache record's name
    /// (`shader_cache/driver.rs`).
    pub(super) settings: metal2vulkan::cache::Hash256,
    /// The instance layers the caller asked for that the loader could not provide, which the
    /// instance was created without (`create::layers_after`), in the order asked: each device
    /// made of the instance says so as it is made (`create::layers_dropped`). Empty for an
    /// instance created with every layer asked for, for one adopted, and where none was asked
    /// for.
    pub(super) dropped_layers: Vec<std::ffi::CString>,
}

impl InstanceShared {
    /// The sink the instance's messenger reports to, if it has one.
    pub(super) fn sink(&self) -> Option<&MessageSink> {
        self.messenger.as_ref().map(create::Messenger::sink)
    }

    /// The relay of the instance's failures, if it has one (`relay.rs`).
    pub(super) fn relay(&self) -> Option<&relay::Relay> {
        self.relay.as_ref().map(relay::Messenger::relay)
    }
}

impl Drop for InstanceShared {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        // Once the process is exiting, the instance is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        if let Some(messenger) = &self.messenger {
            messenger.destroy();
        }
        if let Some(relay) = &self.relay {
            relay.destroy();
        }
        // SAFETY: the runtime created the instance, and every device and messenger made from it
        // is destroyed (devices hold this instance, so they dropped first).
        unsafe { release::instance(&self.raw) };
    }
}

/// Everything one device's objects share.
pub(super) struct DeviceShared {
    /// The instance outlives the device.
    pub(super) instance: Arc<InstanceShared>,
    pub(super) physical: vk::PhysicalDevice,
    /// The device's functions: through [`DeviceShared::vk`] only, or a share that holds nothing
    /// of the device ([`DeviceShared::functions`]).
    functions: Arc<Functions>,
    /// The primary queue family: its first queue's, where internal submissions go.
    pub(super) queue_family: u32,
    /// The `VkQueue`s the runtime submits to, the primary family's first queue first: at least
    /// one.
    pub(super) queues: Vec<QueueSlot>,
    /// The families of those queues, the primary first, each once: what the device's buffers and
    /// images are shared among (`sharing.rs`).
    pub(super) queue_families: Vec<u32>,
    /// How many of `queues`, the first, logical queues share round robin; the rest are reserved
    /// for presenting queues.
    round_robin: usize,
    /// The logical queues made round robin so far: the next one takes `VkQueue` this modulo
    /// `round_robin`.
    round_robin_made: AtomicU64,
    /// The presenting queues made so far on the reserved `VkQueue`s: the next one takes the
    /// reserved queue this modulo their number.
    presenting_made: AtomicU64,
    /// The compile-and-plan device: the description restricted to what is enabled.
    compile: metal2vulkan::device::Device,
    /// What `compile`'s profile takes of viewport depth ranges past `[0, 1]`, which draws read
    /// (`render_state::native_depth_range`).
    pub(super) depth_ranges: DepthRanges,
    /// What the `VkDevice` enabled, in the compile layer's vocabulary.
    enabled: EnableSet,
    /// The names of the device extensions it enabled ([`Device::enables_extension`]).
    extensions: Vec<std::ffi::CString>,
    properties: DeviceProperties,
    /// The kinds of degradation the device announced (`degradation.rs`).
    pub(super) announced: Announced,
    /// The kinds that refuse a render encoder's call met while lowering it after it returned
    /// (`DeviceOptions::lowering_refuses`, `render/deferral.rs`).
    lowering_refuses: BTreeSet<Degradation>,
    /// Its render encoders whose calls are logged, lowered before a read of what their lowering
    /// leaves (`render/deferral.rs`).
    pub(crate) deferred: crate::render::DeferredEncoders,
    /// Buffers carry Vulkan device addresses (`bufferDeviceAddress` is enabled).
    pub(super) device_address: bool,
    /// The alignment of every upload (`upload.rs`): 16 bytes, and the device's storage and
    /// uniform buffer offset alignments.
    pub(super) upload_alignment: u64,
    /// The most dynamic uniform and storage buffers a pipeline layout takes
    /// (`maxDescriptorSetUniformBuffersDynamic`, `maxDescriptorSetStorageBuffersDynamic`), which
    /// the set-0 bindings a variant makes dynamic stay within (`pipeline/objects.rs`).
    pub(super) dynamic_buffers: [u32; 2],
    /// Shared with the releases of freed memory posted to the handler threads, which hold
    /// nothing of the device (`freed_memory.rs`).
    pub(super) memory: Arc<Mutex<Memory>>,
    pub(super) addresses: Mutex<AddressSpace>,
    /// Parked buffers of the runtime's own memory, and the makings of the live recyclable ones
    /// (`recycle.rs`).
    pub(super) recycled_buffers: Mutex<Recycler<vk::Buffer, BufferMaking>>,
    /// The resource IDs of textures and sampler states (`resource_ids.rs`).
    pub(super) resource_ids: Arc<Mutex<ResourceIds>>,
    /// The buffers whose records the GPU may write, which keep the programs of the pipeline
    /// states the program releases (`released_programs.rs`).
    pub(super) keepers: Keepers,
    /// The `uniqueIdentifier`s the device allocates from 2^36 (`unique_identifier.rs`).
    pub(super) identifiers: Arc<UniqueIdentifiers>,
    /// How many holders hold a sampler ID now and the sampler pool's size (`resource_ids.rs`),
    /// read without the IDs' lock ([`Device::sampler_ids_live`],
    /// [`Device::sampler_id_pool_size`]).
    pub(super) sampler_ids: SamplerIdReads,
    /// The live sampler states, by what decides whether a request shares one (`sampler.rs`):
    /// an equal request gets the live state, as Metal's does.
    pub(super) samplers: Mutex<HashMap<SamplerKey, Weak<SamplerShared>>>,
    /// The device's live Vulkan samplers, held to `maxSamplerAllocationCount`
    /// (`DeviceShared::create_sampler`).
    pub(super) sampler_count: SamplerCount,
    /// The immutable sampler sets of the variants' layouts, by their samplers
    /// (`pipeline/objects.rs`, `pipeline/content_keyed.rs`).
    pub(crate) sampler_sets: Mutex<crate::pipeline::objects::SamplerSets>,
    /// The heap sets and the texture table every command buffer shares (`heap_sets.rs`). Its
    /// lock is taken before `resource_ids`', never under it.
    pub(super) heaps: Mutex<DeviceHeaps>,
    /// The live textures the heap holds whose NaN a heap write of a half-float twin's function
    /// converts (`heap_sets::NanResidue`), read at draws and dispatches without `heaps`.
    pub(super) heap_nan_residue: NanResidue,
    /// The live visible-function tables, by resource ID: an argument buffer can name each, for
    /// a dispatch of the state that made it (`function_table.rs`).
    pub(super) function_tables: Mutex<BTreeMap<u64, Weak<TableShared>>>,
    internal: Mutex<Option<Internal>>,
    /// The internal submissions' timeline as the host saw it complete (`submit.rs`).
    internal_done: OwnTimeline,
    /// The staging copies' timeline as the host saw it complete (`staging.rs`).
    staging_done: OwnTimeline,
    /// Each batch with work makes what completed work wrote visible by a command buffer of
    /// barriers that starts it, and waits for no timeline the host saw complete: the driver's
    /// submissions stall on semaphore operations (`DriverFact::SemaphoreOperationsStallSubmits`,
    /// `command_buffer/submit.rs`).
    pub(super) completed_by_barrier: bool,
    /// Each batch ends with `Barrier::to_all`, so a later batch of its `VkQueue` plans no barrier
    /// for its accesses, and a batch's start timestamp is written at the bottom of the pipe
    /// (`batch_order.rs`).
    pub(crate) by_ends: bool,
    /// The last sparse binding of a no-copy buffer's dma-bufs (`dma_buf.rs`) the host has not
    /// seen complete: the internal timeline and the value it signals, which every later
    /// submission that may use the buffer waits for (`submit.rs`).
    bound_at: Mutex<Option<(vk::Semaphore, u64)>>,
    /// The queue reserved for staging copies, where the device has one (`staging.rs`).
    staging: Option<staging::StagingQueue>,
    pub(super) scheduler: Scheduler,
    /// The scheduler's nudge, a timeline semaphore its wakers signal where its thread waits for
    /// the GPU (`scheduler/mod.rs`): made on every device, used where the driver's wait for any
    /// of several semaphores loses no signal (not MoltenVK's).
    nudge: vk::Semaphore,
    pub(super) completion: Completion,
    /// The idle fences the batches that end command buffers signal (`command_buffer/fences.rs`).
    pub(super) fences: Fences,
    /// The queues' serial contexts: every command buffer's transitions and handlers run there,
    /// and the releases handed off a thread that may run none of the program's code
    /// ([`Device::let_go_elsewhere`]).
    pub(super) contexts: Contexts<Item>,
    loader_word: LoaderWord,
    /// The runtime created the `VkDevice` and destroys it.
    owned: bool,
    /// The device was lost: no further submission. Shared with the relay (`relay.rs`), which
    /// records a loss it hears of.
    lost: Arc<AtomicBool>,
    pub(super) textures: super::texture::TextureState,
    /// The translated-shader cache on disk and the driver's pipeline cache.
    pub(super) shader_cache: ShaderCache,
    /// How no-copy buffers import host memory, or why they are refused.
    pub(super) host_import: Result<Importer, metal2vulkan::error::Unsupported>,
    /// Whether the device imports a pointer to shared memory, once asked
    /// (`Device::imports_shared_host_pointers`).
    pub(super) shared_pointer_imports: OnceLock<bool>,
    /// The commands of a render pipeline's extended dynamic states, where the device has them
    /// (`dynamic_state.rs`).
    pub(super) extended_dynamic_state: Option<ExtendedDynamicState>,
    /// The command that binds an index buffer range of a size, where the device enabled
    /// `maintenance5` (`draw_source.rs`).
    pub(super) sized_index_bind: Option<SizedIndexBind>,
    /// The command that records a depth bias of a pinned unit, where the device enabled
    /// `VK_EXT_depth_bias_control` (`render_state.rs`).
    pub(super) depth_bias_command: Option<DepthBiasCommand>,
    /// The mesh draw commands, where the device enabled `meshShader` (`render/mesh.rs`).
    pub(super) mesh_draws: Option<MeshDraws>,
    /// `vkCmdPushDescriptorSet`, from Vulkan 1.4 or `VK_KHR_push_descriptor`
    /// (`binding::push_set`); none where the driver has
    /// `RuntimeDriverDefect::PushedBufferSizesUnset`.
    pub(super) push_descriptor: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
    /// A draw after a pipeline bind pushes at least one of set 0's descriptors, the first where
    /// none changed (`render/bound.rs`), where the driver has
    /// `RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`.
    pub(super) pushes_after_pipeline_binds: bool,
    /// The command that sets a draw's color write masks, where the device enabled everything a
    /// draw that reads its own pass's texture through a snapshot takes (`render/snapshot.rs`).
    pub(super) color_write_masks: Option<ColorWriteMasks>,
    /// The device's slot among those its commands go through (`recorder`), where it took one:
    /// let go of after `vkDestroyDevice` (it drops after the teardown, `Drop`). Held, not read.
    _recording: Option<Slot>,
    /// The thread that replays its encoders' deferred command buffers, where it has a slot
    /// (`recorder`).
    recorder: Option<Recorder>,
    /// The reasons a draw that reads its pass's texture took the split that the device said
    /// already, where `METAL2VULKAN_LOG_PIPELINES` is set (`render/snapshot.rs`).
    pub(super) snapshot_fallbacks: Mutex<Vec<&'static str>>,
    /// The draws the area rule weighed, by size, where `METAL2VULKAN_LOG_PIPELINES` is set
    /// (`render/snapshot.rs`, `reader_sizes.rs`).
    pub(super) reader_sizes: Mutex<crate::reader_sizes::ReaderSizes>,
    /// The command buffers' index and mesh records pools: their bytes and spare blocks
    /// (`gpu_pools.rs`).
    pub(super) gpu_pools: GpuPools,
    /// The most bytes of images each command queue keeps for its heaps' multisample textures'
    /// moves: a sixty-fourth of the largest device-local heap, as a GPU pool's
    /// (`texture/copy_images.rs`, `gpu_pools::device_share`).
    pub(super) copy_images_bytes: u64,
    /// The largest framebuffer: `maxFramebufferWidth`, `maxFramebufferHeight` and
    /// `maxFramebufferLayers` (`render_pass.rs`).
    pub(super) max_framebuffer: [u32; 3],
    /// The viewport limits a rate-mapped pass's regions keep within: `maxViewportDimensions` and
    /// `viewportBoundsRange` (`rate_mapped_pass.rs`).
    pub(super) region_limits: RegionLimits,
    /// Whether a rasterization-rate map of one or more layers was released (`rate_map.rs`): a
    /// map of no layers made after it is one Metal may draw from the released map's memory.
    pub(super) layered_rate_map_released: AtomicBool,
    /// Whether a command buffer holding a render pass with a draw call over a `Mutable`
    /// rasterization-rate map was committed (`rate_mapped_pass.rs`): before one, Metal's GPU hangs
    /// at a pass of no draw call over such a map.
    pub(super) mutable_rate_map_called: AtomicBool,
    /// Whether the runtime made a sparse image on the device: the driver's first sparse resource
    /// may wait for every queue's work (`DriverFact::FirstSparseResourceWaitsForQueues`).
    pub(super) sparse_image_made: AtomicBool,
    /// The crate-owned compute kernels the runtime dispatches for itself, made on first use
    /// (`owned_kernel.rs`).
    pub(super) owned_kernels: Mutex<OwnedKernels>,
    /// The spare scratch block of `generate_mipmaps`' filter chains (`mipmap_scratch.rs`).
    pub(super) mipmap_scratch: Mutex<MipmapScratch>,
    /// The sample planes' copy kernels and merge pipelines, made on first use
    /// (`sample_planes.rs`).
    pub(super) planes: Mutex<PlanesObjects>,
    /// The buffer<->multisampled-texture copy's interleave kernels and merge pipelines, made on
    /// first use (`sample_interleave.rs`).
    pub(super) interleave: Mutex<super::sample_interleave::InterleaveObjects>,
    /// Released variants' objects and `VkPipeline`s, kept for a state made again
    /// (`pipeline/kept.rs`).
    pub(super) kept: Mutex<crate::pipeline::kept::Kept>,
    /// The variants' shader modules by their exact SPIR-V, shared device-wide
    /// (`pipeline/content_keyed.rs`).
    pub(super) shader_modules: Mutex<crate::pipeline::ShaderModules>,
    /// The variants' pipeline layouts by their exact definition, shared device-wide
    /// (`pipeline/shared_layouts.rs`).
    pub(crate) pipeline_layouts: Mutex<crate::pipeline::shared_layouts::SharedLayouts>,
    /// The variants' makings in the air, by their compile, which a twin's waits for to land
    /// (`pipeline/flight.rs`).
    pub(crate) flights: crate::pipeline::flight::Flights,
    /// The exact color resolve's pipelines and passes, made on first use (`color_resolve.rs`).
    pub(super) color_resolves: Mutex<ColorResolveObjects>,
    /// The multisampled stencil's copies without colour copies: the split kernel and the plane
    /// draws, made on first use (`stencil_copy.rs`).
    pub(super) stencil_copies: Mutex<StencilCopyObjects>,
    /// The multisampled depth's copies without colour copies: the draw's shaders and pipelines,
    /// made on first use (`depth_copy.rs`).
    pub(super) depth_copies: Mutex<DepthCopyObjects>,
    /// The stencil resolves picked by depth: the draws' shaders and pipelines, made on first use
    /// (`depth_stencil_resolve.rs`).
    pub(super) depth_stencil_resolves: Mutex<DepthStencilResolveObjects>,
    /// The render passes of its passes without attachments, one per sample count, made on first
    /// use (`attachmentless.rs`).
    pub(super) attachmentless: Mutex<KeptPasses>,
    /// The render passes it keeps: compatible ones and those instances begin, by what each is
    /// made of (`render_target::KeptRenderPasses`).
    pub(super) render_passes: Mutex<KeptRenderPasses>,
    /// The caller's images that move between layouts, and the events' timeline semaphores
    /// (`interop.rs`).
    pub(super) interop: Interop,
    /// Its heaps with bytes of their own, by the key their resources' uses name (`heap.rs`).
    pub(super) heap_keys: super::heap::HeapKeys,
    /// The buffers over host pages, and the holds of uncommitted command buffers on them.
    pub(super) rebinds: super::rebind::Rebinds,
    /// The no-copy buffers by the host memory they were made over (`host_aliases.rs`).
    pub(super) aliases: super::host_aliases::HostAliases,
    /// The new textures that read zero where Metal reads its pattern, until a command observes
    /// or covers that (`fresh_unwritten.rs`).
    pub(super) fresh: super::fresh_unwritten::FreshUnwritten,
    /// The program's resources by key, with what owns their memory (`used_resources.rs`).
    pub(super) owners: super::used_resources::Owners,
    /// The program's live buffers of their own, and their resident set (`live_buffers.rs`).
    pub(super) live_buffers: super::live_buffers::LiveBuffers,
    /// The imports of the caller's dma-bufs it releases while idle, and the thread that releases
    /// them (`host_residency.rs`).
    #[cfg(unix)]
    pub(super) residency: super::host_residency::Sweeper,
    /// How its command buffers are timed on the GPU (`gpu_time.rs`).
    pub(super) timing: Timing,
    /// Its Vulkan views may present an image in another component layout: it has no portability
    /// subset, or the subset's `imageViewFormatReinterpretation` is enabled, by the runtime on a
    /// device it made or, on an adopted one, by the consumer, who says so
    /// (`RawDevice::view_format_reinterpretation`; `view_plan.rs`), and a made device's options
    /// do not withhold such views (`DeviceOptions::withhold_view_reinterpretation`). Where not,
    /// Metal's views of another layout are shadowed (`texture/view_shadow.rs`).
    pub(super) view_reinterpretation: bool,
    /// `VK_EXT_device_generated_commands`'s commands, where the runtime enabled the extension
    /// (`create.rs`): an execution of commands a kernel writes runs on the GPU alone through
    /// them where every record it may hold is covered and the product sets their path up for its
    /// variants (`generated_execute.rs`), and by indirect dispatches elsewhere (`gpu_execute.rs`).
    pub(super) generated_commands: Option<Commands>,
    /// The device enabled `hostImageCopy`: the runtime on a device it creates that has it
    /// (`create.rs`), the consumer on one it adopts that says so (`adopt.rs`). A heap's colour
    /// texture takes host-transfer usage on a driver that serves it (`texture/heap.rs`).
    pub(super) host_image_copy: bool,
    /// The host image copy functions, where the device enabled `hostImageCopy` and copies on the
    /// host to and from images in `GENERAL`: a new `Shared` or `Managed` texture's image is
    /// moved and zeroed on the host, and its CPU copies run there (`texture/host_copy.rs`).
    pub(super) host_copies: Option<ash::ext::host_image_copy::DeviceFn>,
    /// What it has of sparse binding and residency (`sparse/device.rs`).
    pub(super) sparse: SparseDevice,
    /// The command pools and descriptor pools command buffers that completed gave back, for the
    /// next to take (`spares.rs`).
    pub(super) spares: Spares,
    /// The bind states its ended render encoders left, which the next ones start from
    /// (`live_bind.rs`).
    pub(super) live_binds: crate::live_bind::SpareLiveBinds,
    /// The bind memos its ended render encoders left, which the next ones take (`binding.rs`).
    pub(super) bind_memos: crate::binding::SpareBindMemos,
    /// The rooms of the draws its render encoders kept for instances recorded inline after a
    /// split, which the next such instances take (`render/inline_splits.rs`).
    pub(super) kept_rooms: crate::render::SpareKept,
    /// Counts of its commit path's work, for the lanes (`work_counts.rs`).
    pub(super) counts: super::work_counts::Counters,
    /// Where the driver fails pending work as memory a device-address buffer lay in is freed,
    /// the record of its batches the memory's blocks that go wait by (`freed_memory.rs`).
    pub(crate) parking: OnceLock<Arc<Parking>>,
    /// Its command buffers made and not committed, over all its queues (`queue.rs`).
    pub(super) uncommitted: Arc<super::queue::Uncommitted>,
    /// The threads that make `VkPipeline`s ahead of the draws that record them
    /// (`pipeline/pool.rs`).
    pub(super) pipeline_pool: crate::pipeline::pool::Pool,
    /// The thread that lets go of what a released state's pool pieces gave back as they ended
    /// (`pipeline/reaper.rs`).
    pub(super) reaper: crate::pipeline::reaper::Reaper,
    /// The trails its threads open between a signal and their last let-go of what they held for
    /// the work, which a command buffer's waiters wait for (`trails.rs`).
    pub(crate) trails: Arc<crate::trails::Trails>,
    /// Its GPU's recoveries from faults and hangs, which end a predicated loop running meanwhile
    /// (`command_buffer/recovery.rs`).
    pub(crate) recoveries: Arc<crate::command_buffer::recovery::Recoveries>,
    /// Its graphics pipeline libraries, by part and content, where it links its render
    /// pipelines from libraries (`pipeline/libraries.rs`, `create.rs`); `None` where it makes
    /// each whole.
    pub(crate) libraries: Option<crate::pipeline::libraries::Libraries>,
    /// Its command buffers keep execution records (`DeviceOptions::record_executions`): where
    /// not, its encoders build none (`EncoderCore::executed`).
    pub(crate) record_executions: bool,
    /// Counts the times an image of the device let its views go, under its views lock (a
    /// rebind's or a storage form's new `VkImage`, a detach's stand-in, a reattach): a view kept
    /// by what it was made of (`binding::BindMemo`) is current while this holds.
    pub(crate) views_epoch: AtomicU64,
}

impl DeviceShared {
    /// Whether `kind`, met while lowering a render encoder's call after it returned, refuses the
    /// call (`DeviceOptions::lowering_refuses`).
    pub(crate) fn lowering_refuses(&self, kind: Degradation) -> bool {
        self.lowering_refuses.contains(&kind)
    }

    /// What `create` makes with the `VkPipelineCache` it is handed, its `vkCreate*Pipelines` call:
    /// a cache of the making's own, seeded from the record `making` names
    /// (`ShaderCache::make_pipelines`, `shader_cache/driver.rs`). On a pool's thread the exit waits
    /// for it longer (`exit_guard::driver_making`).
    pub(crate) fn make_pipelines<T, E>(
        &self,
        making: crate::shader_cache::driver::Making,
        create: impl FnOnce(vk::PipelineCache) -> Result<T, E>,
    ) -> Result<T, E> {
        self.make_pipelines_reported(making, |cache| (create(cache), None))
    }

    /// [`make_pipelines`](Self::make_pipelines), with what the driver reported of the making:
    /// `create` returns, with what it made, whether the driver found its pipeline in the cache it
    /// was handed (`shader_cache::found_in_cache`, `None` where it reports nothing). A making
    /// from its record that the driver reports found reads no data back and writes nothing
    /// (`shader_cache/driver.rs`).
    pub(crate) fn make_pipelines_reported<T, E>(
        &self,
        making: crate::shader_cache::driver::Making,
        create: impl FnOnce(vk::PipelineCache) -> (Result<T, E>, Option<bool>),
    ) -> Result<T, E> {
        // What every making is measured to report, withheld inside it (`driver_facts.rs`); a
        // render pipeline's making expects them with its own.
        let expected = crate::driver_facts::making_diagnostics(&self.properties);
        crate::driver_facts::expecting_where_unset(expected, || {
            crate::exit_guard::driver_making(|| {
                self.shader_cache.make_pipelines(self.vk(), making, create)
            })
        })
    }

    /// The device's functions. The runtime's own threads reach them only inside a section of
    /// the exit guard, or while they run the program's code, and a section reaches only the exit
    /// guard's table of entry points; builds with debug assertions end the process otherwise
    /// (`functions.rs`).
    pub(crate) fn vk(&self) -> &Vk {
        self.functions.get()
    }

    /// A share of the device's functions, which holds nothing of the device: for the releases
    /// posted to its handler threads, which its teardown waits for (`freed_memory.rs`). Reached
    /// through [`Functions::get`] alone, as [`vk`](Self::vk) reaches them.
    pub(crate) fn functions(&self) -> Arc<Functions> {
        Arc::clone(&self.functions)
    }

    /// What its encoders' deferred command buffers' replays share, where it records them so
    /// (`recorder`).
    pub(crate) fn recorder(&self) -> Option<&Arc<RecorderShared>> {
        self.recorder.as_ref().map(Recorder::shared)
    }

    /// `vkAllocateCommandBuffers` of `info`: the loader serves it and reads the device's loader
    /// word first ([`LoaderWord`]).
    pub(super) fn allocate_command_buffers(
        &self,
        info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, Error> {
        // SAFETY: the device is live; `info` names one of its pools, which the caller holds
        // exclusively.
        let allocated = unsafe {
            self.loader_word.restore();
            self.vk().allocate_command_buffers(info)
        };
        exit_guard::called(Call::AllocateCommandBuffers);
        allocated.map_err(super::error::vulkan)
    }

    /// Name `pipeline`, just made and held by this thread alone, `name` for the layers' messages,
    /// where the device's instance reports messages (`Vk::object_name`): the loader serves the
    /// call and reads the device's loader word first ([`LoaderWord`]). A failure to name it is
    /// ignored: the name serves messages alone.
    pub(crate) fn name_pipeline(&self, pipeline: vk::Pipeline, name: &std::ffi::CStr) {
        let functions = self.vk();
        let Some(object_name) = functions.object_name() else {
            return;
        };
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(pipeline)
            .object_name(name);
        // SAFETY: the device is live and `pipeline` its own, which no other thread holds yet; the
        // instance enables `VK_EXT_debug_utils` (it reports messages), and the function is the
        // device's.
        let _ = unsafe {
            self.loader_word.restore();
            object_name(functions.handle(), &info)
        };
        exit_guard::called(Call::SetDebugUtilsObjectName);
    }

    /// How the device's buffers and images are shared among its queue families (`sharing.rs`).
    pub(crate) fn sharing(&self) -> Sharing<'_> {
        Sharing::new(&self.queue_families)
    }

    /// The family of the device's `VkQueue` at `index` (a logical queue's), which the command
    /// pools of what it runs are made for.
    pub(crate) fn queue_family_of(&self, index: usize) -> u32 {
        self.queues[index].family
    }

    /// The queue flags of the family of the device's `VkQueue` at `index`.
    pub(crate) fn queue_flags_of(&self, index: usize) -> vk::QueueFlags {
        self.queues[index].flags
    }

    /// A Vulkan sampler of `info`, counted against the device's `maxSamplerAllocationCount`
    /// (`SamplerCount`): refused, `Error::SamplerLimit`, where that many are live. Whoever
    /// destroys it gives the count back (`DeviceShared::sampler_destroyed`).
    ///
    /// # Safety
    ///
    /// `info` and its chain are valid for the device and live for the call.
    pub(crate) unsafe fn create_sampler(
        &self,
        info: &vk::SamplerCreateInfo<'_>,
    ) -> Result<vk::Sampler, Error> {
        self.sampler_count.take()?;
        // SAFETY: the caller's.
        match unsafe { self.vk().create_sampler(info, None) } {
            Ok(sampler) => Ok(sampler),
            Err(result) => {
                self.sampler_count.give();
                Err(vulkan(result))
            }
        }
    }

    /// A sampler of `DeviceShared::create_sampler`'s was destroyed.
    pub(crate) fn sampler_destroyed(&self) {
        self.sampler_count.give();
    }
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        // The whole teardown is one section of the exit guard: once the process is exiting, the
        // device, its threads and its objects are left to its end, and an exit that begins
        // meanwhile waits for the teardown (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // The watch of the program's marks and the releases of freed memory, posted to the
        // handler threads, hold nothing of the device: those that begin from here on call
        // nothing, and the threads' stop below waits for those running (`freed_memory.rs`).
        if let Some(parking) = self.parking.get() {
            parking.close();
        }
        // The draws the snapshot's area rule weighed since the last line (`reader_sizes.rs`).
        if let Some(text) = lock(&self.reader_sizes).last_line() {
            let sink = self.instance.sink();
            crate::driver_facts::inform_through(sink, "metal2vulkan-snapshot-readers", text);
        }
        // Nothing of the device is in flight: its failures are no longer relayed.
        if let Some(relay) = self.instance.relay() {
            relay.forget(&self.lost);
        }
        // Every command buffer holds its device, so none is held or pending: the threads only
        // stop.
        self.scheduler.stop();
        // Every open render encoder holds the device too: none is left to lower.
        self.deferred.stop();
        // Every decode job holds its image, and so the device: none is left.
        self.textures.twin_decodes.stop();
        // What the reaper is handed holds the device: none is left, and it only stops, unless
        // this teardown runs on its thread.
        self.reaper.stop();
        // What the residency thread releases holds the device: none is left.
        #[cfg(unix)]
        self.residency.stop();
        self.completion.stop();
        self.contexts.stop();
        // SAFETY: the device is live; waiting for idle covers any work the consumer of an
        // adopted device did not wait for before dropping the runtime's last handle. No batch is
        // pending then, so no timeline is in use.
        unsafe {
            let _ = self.vk().device_wait_idle();
            exit_guard::called(Call::DeviceWaitIdle);
            // The scheduler's thread, which waited on its nudge, has stopped.
            release::object(self.vk(), self.nudge);
            for queue in &self.queues {
                release::object(self.vk(), queue.timeline);
                // The marks after the last uses of the queue outside a batch (`relay.rs`).
                for fence in queue.flights.get().map(|f| f.drain()).unwrap_or_default() {
                    release::object(self.vk(), fence);
                }
            }
            // Nor the marks of the program's uses of the queues (`freed_memory.rs`).
            if let Some(parking) = self.parking.get() {
                parking.release(self.vk());
            }
            // No fence is pending either: each command buffer's was given back as it finished.
            self.fences.release(self.vk());
            // Nor a timestamp slot: each command buffer gave its slot back as it finished, or
            // with its state.
            self.timing.release(self.vk());
        }
        // Nor a kept barrier command buffer: the queues are idle.
        self.scheduler.release_kept(self.vk());
        // No command buffer is left to give back a spare, or to hold one.
        self.spares.release(self);
        if let Some(internal) = lock(&self.internal).take() {
            internal.destroy(self.vk());
        }
        if let Some(staging) = &self.staging {
            staging.destroy(self.vk());
        }
        lock(&self.textures.neutrals).destroy(self.vk(), &mut lock(&self.memory));
        lock(&self.textures.recycled).destroy(self.vk(), &mut lock(&self.memory));
        lock(&self.recycled_buffers).destroy(self.vk(), &mut lock(&self.memory));
        // No framebuffer is left to hold a placeholder (`storage_placeholder.rs`).
        lock(&self.textures.storage_placeholders).destroy(self.vk(), &mut lock(&self.memory));
        // The zeros new resources' first work copied from (`texture/first_move.rs`).
        (self.textures.first_moves).destroy(self.vk(), &mut lock(&self.memory));
        lock(&self.heaps).destroy(self.vk(), &mut lock(&self.memory));
        lock(&self.owned_kernels).destroy(self.vk());
        lock(&self.mipmap_scratch).destroy(self.vk(), &mut lock(&self.memory));
        self.gpu_pools.destroy(self.vk(), &mut lock(&self.memory));
        lock(&self.planes).destroy(self.vk());
        lock(&self.interleave).destroy(self.vk());
        lock(&self.color_resolves).destroy(self.vk());
        lock(&self.stencil_copies).destroy(self.vk());
        lock(&self.depth_copies).destroy(self.vk());
        lock(&self.depth_stencil_resolves).destroy(self.vk());
        lock(&self.attachmentless).destroy(self.vk());
        lock(&self.render_passes).destroy(self.vk());
        // What released variants left (`pipeline/kept.rs`).
        lock(&self.kept).destroy(self);
        // The graphics pipeline libraries, after every pipeline they linked: each linked
        // pipeline holds the device, and the kept ones went above.
        if let Some(libraries) = &self.libraries {
            for gone in libraries.drain() {
                gone.destroy(&*self);
            }
        }
        // The shader modules, every one the map holds, destroyed here while the device lives: no
        // variant is left but one dropping this device right now (whose share then calls
        // nothing), and the kept ones let go of theirs above (`pipeline/content_keyed.rs`).
        let modules = lock(&self.shader_modules).drain();
        modules.destroy(&*self);
        // The pipeline layouts, likewise (`pipeline/shared_layouts.rs`).
        let layouts = lock(&self.pipeline_layouts).drain();
        layouts.destroy(&*self);
        // The immutable sampler sets, likewise, each counted back as its samplers go: the
        // layouts that baked them are gone.
        let sampler_sets = lock(&self.sampler_sets).drain();
        sampler_sets.destroy(&*self);
        // Every other pipeline was destroyed with its state (each holds this device); the cache
        // thread stops.
        self.shader_cache.close();
        // What waited with blocks of its memory (a host import's deallocators) goes once they are
        // freed, with no lock held.
        let after = lock(&self.memory).destroy(self.vk());
        drop(after);
        // The device's record of pipeline makings goes, before `vkDestroyDevice`: in builds with
        // debug assertions, no making is in progress (`compiles.rs`).
        compiles::unregister(self.vk().handle());
        if self.owned {
            // SAFETY: the runtime created the device, every object made from it is destroyed,
            // and the loader's `vkDestroyDevice` reads the device's loader word first.
            unsafe {
                self.loader_word.restore();
                release::object(self.vk(), release::WholeDevice);
            }
        }
    }
}

/// One of the device's `VkQueue`s.
pub(super) struct QueueSlot {
    /// The queue, under its lock: submissions are externally synchronized.
    raw: Mutex<vk::Queue>,
    /// The queue's family, which the command pools of what it runs are made for.
    pub(super) family: u32,
    /// The family's queue flags: the commands it runs, and so the accesses a barrier recorded
    /// for it may name (`image_barrier.rs`).
    flags: vk::QueueFlags,
    /// The queue's loader word, which MoltenVK overwrites where a message names the queue
    /// (`relay.rs`): put back before each use.
    word: LoaderWord,
    /// Its submissions in flight, where the relay relays its failures (`relay.rs`): set once the
    /// device is primed.
    pub(super) flights: OnceLock<Arc<Flights>>,
    /// The timeline semaphore every batch submitted here signals, with rising values: Vulkan's
    /// signal operation order makes them complete in that order.
    pub(super) timeline: vk::Semaphore,
    /// The value through which every batch submitted here completed, as the host saw.
    completed: AtomicU64,
    /// The value of the last batch submitted here: what the exit hook waits for on a device
    /// whose queues run in this process ([`exit_guard::InProcess`]).
    submitted: AtomicU64,
    /// The value of the submission before the last that signalled the timeline here: while it is
    /// not seen complete, two or more submissions here are in flight
    /// ([`pending_beyond_one`](Self::pending_beyond_one)).
    submitted_before: AtomicU64,
}

impl QueueSlot {
    /// `f` with the queue, under its lock, its loader word put back first.
    pub(super) fn with<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        let queue = lock(&self.raw);
        // SAFETY: the queue lives while its slot does.
        unsafe { self.word.restore() };
        f(*queue)
    }

    /// The host saw the timeline reach `value`.
    pub(super) fn completed(&self, value: u64) {
        self.completed.fetch_max(value, Ordering::AcqRel);
    }

    /// The value through which every batch completed, as the host saw.
    pub(super) fn completed_value(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// A batch signalling `value` was submitted here, inside its section of the exit guard: the
    /// hook, which waits for that section, sees it. Submissions here are made under the queue's
    /// lock, so the last two values move together.
    pub(super) fn submitted(&self, value: u64) {
        let before = self.submitted.fetch_max(value, Ordering::AcqRel);
        if before < value {
            self.submitted_before.store(before, Ordering::Release);
        }
    }

    /// Where two or more submissions that signalled the timeline here are not seen complete, the
    /// value of the one before the last, whose completion leaves one in flight: each submission
    /// signals a higher value, and the timeline completes them in order
    /// (`command_buffer/group.rs`). Read without the queue's lock, so a submission made meanwhile
    /// is seen or not.
    pub(super) fn pending_beyond_one(&self) -> Option<u64> {
        let before = self.submitted_before.load(Ordering::Acquire);
        (self.completed_value() < before).then_some(before)
    }

    /// The value of the last batch submitted here; 0 before the first.
    fn submitted_value(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }
}

/// `mutex`'s guard; a lock poisoned by a panicking holder is taken over (the runtime's state is
/// updated before any call that can panic).
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Device {
    /// `registryID`: an identifier of the physical device, from its `deviceUUID`.
    pub fn registry_id(&self) -> u64 {
        properties::registry_id(&self.0.properties.device_uuid)
    }

    /// `name`.
    pub fn name(&self) -> &str {
        &self.0.properties.name
    }

    /// The device's identity and Metal's public device queries, with the documented facts a
    /// consumer computes Apple's private ones from.
    pub fn properties(&self) -> &DeviceProperties {
        &self.0.properties
    }

    /// The compile-and-plan device this device compiles and plans for. No Metal counterpart.
    pub fn compile_device(&self) -> &metal2vulkan::device::Device {
        &self.0.compile
    }

    /// What the Vulkan device enabled, in the compile layer's vocabulary. No Metal counterpart.
    pub fn enabled(&self) -> &EnableSet {
        &self.0.enabled
    }

    /// Whether the Vulkan device enabled the device extension `name`: one the options asked for
    /// (`DeviceOptions::device_extensions`, and each of `optional_device_extensions` the device
    /// could enable), or one the runtime enabled itself (of the vocabulary's names, the one the
    /// device listed). An adopted device answers for its `RawDevice::device_extensions`. No Metal
    /// counterpart.
    pub fn enables_extension(&self, name: &std::ffi::CStr) -> bool {
        self.0
            .extensions
            .iter()
            .any(|enabled| enabled.as_c_str() == name)
    }

    pub(super) fn shared(&self) -> &DeviceShared {
        &self.0
    }

    /// Keep room in the memory pools the runtime's encodings and commits place their own
    /// resources in (`memory::keep_room`): called by the creation calls that may wait (a queue's,
    /// a pipeline state's, a buffer's, a texture's, a heap's), so that no encoding or commit makes
    /// a memory block, whose `vkAllocateMemory` waited 2.37 s for other processes' GPU work on
    /// NVIDIA's driver. Nothing in a host step, a serial point of the command buffers' stream
    /// that makes only what it needs.
    pub(crate) fn keep_room(&self) {
        if crate::exit_guard::in_host_step() {
            return;
        }
        let shared = self.shared();
        crate::memory::keep_room(&shared.memory, shared.vk(), &shared.counts);
    }

    /// Whether `other` is this device (the same handle, not merely the same physical device).
    pub(super) fn same(&self, other: &Device) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Whether the device was lost.
    pub(super) fn is_lost(&self) -> bool {
        self.0.lost.load(Ordering::Acquire)
    }

    /// Record that the device was lost.
    pub(super) fn mark_lost(&self) {
        self.0.lost.store(true, Ordering::Release);
    }

    /// Wait for the work submitted to each of the device's queues to end, each queue under its
    /// lock, the waits one section of the exit guard: what a completion turn does after a
    /// failed wait, before that wait's command buffers let go of what they used
    /// (`command_buffer/completion.rs`). Once the process is exiting it waits for nothing, and
    /// the releases that follow make no Vulkan call.
    ///
    /// Vulkan lets go of a lost device's objects at once. MoltenVK 1.4.1 (`MVKQueue.mm`,
    /// `MVKCommandBuffer.mm`) does not: a failed Metal command buffer's first completed handler
    /// marks the device lost, so a wait for its fence returns `VK_ERROR_DEVICE_LOST` while its
    /// later handlers have yet to run, and one of them returns each scratch buffer a command
    /// such as `vkCmdUpdateBuffer` took to its `VkCommandPool`'s pool; with that command pool
    /// destroyed first, the process aborted there (`mutex lock failed`). Its
    /// `vkQueueWaitIdle` commits an empty Metal command buffer to the queue and waits for it,
    /// lost or not, and Metal completes a queue's command buffers in order, so the earlier
    /// ones' handlers have run when it returns. Another driver's returns at once on a lost
    /// device, and waits for the work on a device that is not.
    pub(super) fn drain_queues(&self) {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = &self.0;
        let raw = shared.vk();
        for slot in shared.queues.iter() {
            // SAFETY: the queue is the device's, used under its lock, inside a section. Its
            // answer is not needed: the device is lost, or the work it waited for ended.
            let _ = slot.with(|queue| unsafe { raw.queue_wait_idle(queue) });
            exit_guard::called(Call::QueueWaitIdle);
        }
        shared.counts.queue_drain();
    }

    /// Let go of `held` at once on one of the device's handler threads, in a context of its own
    /// (an identity no logical queue takes, so it waits behind no queue's handlers): what the
    /// scheduler's thread lets go of under its locks where its drop may run the program's code,
    /// which may not run there (`exit_guard::program_release`). Its drop runs there as on the
    /// program's own thread, with no command buffer's step marked.
    pub(crate) fn let_go_elsewhere(&self, held: Box<dyn Send>) {
        let (contexts, context) = self.elsewhere();
        contexts.post(context, Item::Release(held));
    }

    /// Where [`let_go_elsewhere`](Self::let_go_elsewhere) posts: the device's handler threads,
    /// and a context of its own. The handle of the pool holds nothing of the device, so what holds
    /// the device can be posted with no handle of it taken here.
    pub(crate) fn elsewhere(&self) -> (Contexts<Item>, u64) {
        (self.0.contexts.clone(), self.0.contexts.fresh())
    }

    /// A new logical queue's identity, and the index of the `VkQueue` it submits to: round
    /// robin over the device's queues that are not reserved for presenting.
    pub(super) fn next_queue(&self) -> (u64, usize) {
        let id = self.0.contexts.fresh();
        let made = self.0.round_robin_made.fetch_add(1, Ordering::Relaxed);
        (id, (made % self.0.round_robin as u64) as usize)
    }

    /// A new presenting queue's identity, and the index of the `VkQueue` it submits to: the
    /// queues reserved for presenting, in turn. `None` where the device reserved none.
    pub(super) fn next_presenting_queue(&self) -> Option<(u64, usize)> {
        let reserved = self.0.queues.len() - self.0.round_robin;
        if reserved == 0 {
            return None;
        }
        let id = self.0.contexts.fresh();
        let made = self.0.presenting_made.fetch_add(1, Ordering::Relaxed);
        Some((id, self.0.round_robin + (made % reserved as u64) as usize))
    }

    /// `vkAllocateCommandBuffers` of `info` (`DeviceShared::allocate_command_buffers`).
    pub(super) fn allocate_command_buffers(
        &self,
        info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, Error> {
        self.0.allocate_command_buffers(info)
    }
}

/// The parts [`DeviceShared`] is built from, by `create.rs` and `adopt.rs` alike.
pub(super) struct Parts {
    pub(super) instance: Arc<InstanceShared>,
    pub(super) physical: vk::PhysicalDevice,
    pub(super) raw: ash::Device,
    /// The device's loader word as it stood when the device reached the runtime.
    pub(super) loader_word: LoaderWord,
    /// The primary queue family, the first queue's.
    pub(super) queue_family: u32,
    /// The queues the runtime submits to, each its family and index there, the primary family's
    /// first queue first: at least one.
    pub(super) queues: Vec<(u32, u32)>,
    /// How many of `queues`, the last, are reserved for presenting queues (fewer than all).
    pub(super) present_queues: usize,
    /// The queue reserved for staging copies, its family and index there, outside `queues`
    /// (`staging.rs`): `None` for an adopted device, or where the families have no more.
    pub(super) staging: Option<(u32, u32)>,
    pub(super) compile: metal2vulkan::device::Device,
    /// The description `compile` was built from: what the device enables.
    pub(super) description: DeviceDescription,
    pub(super) enabled: EnableSet,
    /// The names of the device extensions it enables ([`Device::enables_extension`]).
    pub(super) extensions: Vec<std::ffi::CString>,
    /// The form of `VK_KHR_calibrated_timestamps` it enables, outside `enabled`'s vocabulary.
    pub(super) calibration: Option<gpu_time::Extension>,
    /// It answers GPU times from the host, writing no timestamps (`gpu_time.rs`).
    pub(super) gpu_times_from_host: bool,
    /// The degradation kinds it announces at every occurrence (`degradation.rs`).
    pub(super) announce_each: BTreeSet<Degradation>,
    /// The degradation kinds that refuse a render encoder's call met while lowering it after it
    /// returned (`render/deferral.rs`).
    pub(super) lowering_refuses: BTreeSet<Degradation>,
    /// Its command buffers keep execution records (`DeviceOptions::record_executions`).
    pub(super) record_executions: bool,
    /// Its views may reinterpret an image's components ([`DeviceShared`]'s field), outside
    /// `enabled`'s vocabulary.
    pub(super) view_reinterpretation: bool,
    /// It enables `VK_EXT_device_generated_commands` (`create.rs`), outside `enabled`'s
    /// vocabulary.
    pub(super) generated_commands: bool,
    /// It enables `hostImageCopy` where a heap's colour texture takes host-transfer usage
    /// (`create.rs`; `adopt.rs`), outside `enabled`'s vocabulary.
    pub(super) host_image_copy: bool,
    /// It enables `VK_NV_linear_color_attachment` and its `linearColorAttachment` (`create.rs`;
    /// `adopt.rs`, where the consumer lists it), outside `enabled`'s vocabulary: a linear image
    /// renders in place where the instance's color attachments are all linear
    /// (`linear_attachment.rs`).
    pub(super) linear_color_attachment: bool,
    /// It enables `VK_EXT_graphics_pipeline_library` and links its libraries fast (`create.rs`;
    /// `adopt.rs`, where the consumer lists it), outside `enabled`'s vocabulary: its render
    /// pipelines are linked from libraries (`pipeline/libraries.rs`).
    pub(super) graphics_pipeline_library: bool,
    /// It enables `VK_EXT_pipeline_creation_feedback` (`create.rs`), outside `enabled`'s
    /// vocabulary, wherever listed: each making of a program's pipeline reads whether its cache
    /// held the pipeline (`shader_cache/mod.rs`).
    pub(super) creation_feedback: bool,
    /// It enables the dma-buf route's extensions (`create.rs`'s `dma_buf_names`), outside
    /// `enabled`'s vocabulary (`dma_buf.rs`).
    pub(super) dma_buf: bool,
    /// It enables the route's image extensions (`create.rs`'s `dma_buf_image_names`), likewise.
    pub(super) dma_buf_images: bool,
    /// The sparse members of the core features it enables, outside `enabled`'s vocabulary
    /// (`sparse/device.rs`).
    pub(super) sparse_features: vk::PhysicalDeviceFeatures,
    /// The reads its global memory barriers name (`image_barrier::MemoryReads`): those of the
    /// primary queue family's commands with the extensions it enables.
    pub(super) memory_reads: MemoryReads,
    pub(super) owned: bool,
    /// The shader cache, its directory opened before any Vulkan object was made.
    pub(super) cache: CacheSetup,
    /// The Metal level it answers at, and how it was chosen (`metal_level.rs`).
    pub(super) metal_level: crate::metal_level::Chosen,
}

impl Device {
    /// The device of `parts`: create is built on adopt, so both end here.
    ///
    /// # Safety
    ///
    /// `parts.raw` is a live device of `parts.physical`, made from `parts.instance` with
    /// `parts.enabled` enabled (and `timelineSemaphore`) and the queues `parts.queues`, and
    /// `parts.loader_word` is its loader word. When this fails, a device
    /// the runtime owns is destroyed.
    unsafe fn assemble(parts: Parts) -> Result<Device, Error> {
        let Parts {
            instance,
            physical,
            raw,
            loader_word,
            queue_family,
            queues: queue_places,
            present_queues,
            staging: staging_place,
            compile,
            description,
            enabled,
            extensions,
            calibration,
            gpu_times_from_host,
            announce_each,
            lowering_refuses,
            record_executions,
            view_reinterpretation,
            generated_commands,
            host_image_copy,
            linear_color_attachment,
            graphics_pipeline_library,
            creation_feedback,
            dma_buf,
            dma_buf_images,
            sparse_features,
            memory_reads,
            owned,
            cache,
            metal_level,
        } = parts;
        // Every pipeline made on the device from here on is made at the user-initiated class.
        // SAFETY: the device is live (the caller's contract), and its functions are its own.
        let raw = unsafe { compiles::route(raw) };
        // Every command recorded with the device's functions goes through a slot of its own
        // (`recorder`), dropped with the device after `vkDestroyDevice`.
        // SAFETY: as above; its functions were routed for its pipeline makings alone.
        let (raw, recording) = unsafe { crate::recorder::route(raw) };
        // What the runtime made from the device, destroyed with it when a later step fails.
        let undo = |timelines: &[vk::Semaphore]| {
            // The record of pipeline makings first, as in the teardown (`compiles.rs`).
            compiles::unregister(raw.handle());
            // SAFETY: nothing submitted uses the semaphores, and nothing else the runtime made
            // from the device remains.
            unsafe {
                for &timeline in timelines {
                    release::object(&raw, timeline);
                }
                if owned {
                    loader_word.restore();
                    release::object(&raw, release::WholeDevice);
                }
            }
        };
        // Each `VkQueue`'s timeline, then the scheduler's nudge (`scheduler/mod.rs`).
        let mut timelines = Vec::with_capacity(queue_places.len() + 1);
        for _ in 0..=queue_places.len() {
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
            // SAFETY: the device is live and enables `timelineSemaphore` (the caller's contract).
            match unsafe { raw.create_semaphore(&info, None) } {
                Ok(timeline) => timelines.push(timeline),
                Err(error) => {
                    undo(&timelines);
                    return Err(super::error::vulkan(error));
                }
            }
        }
        // Each batch ends with a barrier that orders all its commands before every later one of
        // its queue, where the driver runs a queue's submissions one after another anyway
        // (`batch_order.rs`).
        let by_ends = crate::batch_order::by_ends(compile.policy(), description.driver_id);
        let threads = Contexts::start(THREADS).and_then(|contexts| {
            match Completion::start(queue_places.len()) {
                Ok(completion) => match Scheduler::start(queue_places.len(), by_ends) {
                    Ok(scheduler) => Ok((contexts, completion, scheduler)),
                    Err(error) => {
                        completion.stop();
                        contexts.stop();
                        Err(error)
                    }
                },
                Err(error) => {
                    contexts.stop();
                    Err(error)
                }
            }
        });
        let (contexts, completion, scheduler) = match threads {
            Ok(threads) => threads,
            Err(error) => {
                undo(&timelines);
                return Err(error);
            }
        };
        // SAFETY: `physical` belongs to `instance`.
        let families = unsafe {
            instance
                .raw
                .get_physical_device_queue_family_properties(physical)
        };
        let flags_of = |family: u32| {
            (families.get(family as usize)).map_or(vk::QueueFlags::empty(), |p| p.queue_flags)
        };
        let nudge = timelines[queue_places.len()];
        let queues: Vec<QueueSlot> = queue_places
            .iter()
            .zip(&timelines)
            .map(|(&(family, index), &timeline)| {
                // SAFETY: the queue exists (the caller's contract), and the loader serves the
                // call after the word is put back.
                let queue = unsafe {
                    loader_word.restore();
                    raw.get_device_queue(family, index)
                };
                QueueSlot {
                    raw: Mutex::new(queue),
                    family,
                    flags: flags_of(family),
                    // SAFETY: the queue was just returned, its word the loader's.
                    word: unsafe { LoaderWord::of_handle(queue.as_raw()) },
                    flights: OnceLock::new(),
                    timeline,
                    completed: AtomicU64::new(0),
                    submitted: AtomicU64::new(0),
                    submitted_before: AtomicU64::new(0),
                }
            })
            .collect();
        let mut queue_families: Vec<u32> = Vec::with_capacity(queues.len());
        for slot in &queues {
            if !queue_families.contains(&slot.family) {
                queue_families.push(slot.family);
            }
        }
        let staging = staging_place.map(|(family, index)| {
            // SAFETY: the queue exists (the caller's contract), and the loader serves the call
            // after the word is put back; it was just returned, its word the loader's.
            unsafe {
                loader_word.restore();
                let queue = raw.get_device_queue(family, index);
                staging::StagingQueue::new(family, flags_of(family), queue, &queue_families)
            }
        });
        let family = (families.get(queue_family as usize).copied()).unwrap_or_default();
        let timestamp_bits = family.timestamp_valid_bits;
        // SAFETY: as above.
        let sparse_properties =
            unsafe { instance.raw.get_physical_device_properties(physical) }.sparse_properties;
        let sparse = SparseDevice::of(&sparse_features, family.queue_flags, &sparse_properties);
        let device_address = enabled
            .features
            .contains(&metal2vulkan::device::VkFeature::BufferDeviceAddress);
        let generated_commands = if generated_commands {
            // SAFETY: the device enables the extension; the loader serves `vkGetDeviceProcAddr`
            // once the word is put back.
            unsafe {
                loader_word.restore();
                Commands::load(raw.handle(), |name| {
                    instance
                        .raw
                        .get_device_proc_addr(raw.handle(), name.as_ptr())
                })
            }
            .map(|commands| commands.routed(recording.as_ref()))
        } else {
            None
        };
        // None where the driver's pushed sets lose their buffers' sizes
        // (`RuntimeDriverDefect::PushedBufferSizesUnset`): every draw and dispatch binds set 0
        // there, as on a device without the command.
        let pushes_whole_sets = !crate::driver_defects::of(&description)
            .contains(&metal2vulkan::error::RuntimeDriverDefect::PushedBufferSizesUnset);
        // At least one write of set 0 pushed after each pipeline bind where the driver's pushed
        // descriptors go stale across one
        // (`RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`).
        let pushes_after_pipeline_binds = crate::driver_defects::of(&description).contains(
            &metal2vulkan::error::RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds,
        );
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let push_descriptor = unsafe {
            loader_word.restore();
            crate::binding::push_descriptor_command(
                &instance.raw,
                &raw,
                description.api_version,
                &enabled,
            )
        }
        .filter(|_| pushes_whole_sets)
        .map(|push| {
            // SAFETY: the device's own command of that name.
            unsafe {
                crate::recorder::through(recording.as_ref(), Command::CmdPushDescriptorSet, push)
            }
        });
        // Its naming of an object, where its instance reports messages: a graphics pipeline that
        // reads its input attachments in rasterization order is named for the layers' messages
        // (`pipeline/graphics.rs`).
        let object_name = match instance.sink() {
            // SAFETY: as above; the instance enables `VK_EXT_debug_utils` where it reports
            // messages, and the loader serves `vkGetDeviceProcAddr` once the word is put back.
            Some(_) => unsafe {
                loader_word.restore();
                instance
                    .raw
                    .get_device_proc_addr(raw.handle(), c"vkSetDebugUtilsObjectNameEXT".as_ptr())
            },
            None => None,
        }
        .map(|function| {
            // SAFETY: the device gave the function of this name.
            unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(),
                    vk::PFN_vkSetDebugUtilsObjectNameEXT,
                >(function)
            }
        });
        // Its calibrated timestamps, where its queue family writes timestamps: a completion turn
        // reads them (`gpu_time.rs`).
        // SAFETY: the device enables `calibration`'s form where it is set; the loader serves
        // `vkGetDeviceProcAddr` once the word is put back.
        let calibrated = unsafe {
            loader_word.restore();
            gpu_time::calibrated(&instance.entry, &instance.raw, physical, &raw, calibration)
        }
        .filter(|_| timestamp_bits > 0);
        // The render encoder's commands of extensions and of Vulkan 1.3, which priming makes too.
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let extended_dynamic_state = unsafe {
            loader_word.restore();
            ExtendedDynamicState::load(
                &instance.raw,
                &raw,
                description.api_version,
                &enabled,
                recording.as_ref(),
            )
        };
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let sized_index_bind = unsafe {
            loader_word.restore();
            sized_index_bind(&instance.raw, &raw, &enabled)
        }
        .map(|bind| {
            // SAFETY: the device's own command of that name.
            unsafe {
                crate::recorder::through(recording.as_ref(), Command::CmdBindIndexBuffer2, bind)
            }
        });
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let depth_bias_command = unsafe {
            loader_word.restore();
            depth_bias_command(&instance.raw, &raw, &enabled)
        }
        .map(|bias| {
            // SAFETY: the device's own command of that name.
            unsafe { crate::recorder::through(recording.as_ref(), Command::CmdSetDepthBias2, bias) }
        });
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let color_write_masks = unsafe {
            loader_word.restore();
            ColorWriteMasks::load(&instance.raw, &raw, &enabled)
        }
        .map(|ColorWriteMasks(masks)| {
            // SAFETY: the device's own command of that name.
            ColorWriteMasks(unsafe {
                crate::recorder::through(recording.as_ref(), Command::CmdSetColorWriteMask, masks)
            })
        });
        let drawn = prime::PrimedDraws {
            dynamic: extended_dynamic_state,
            color_write_masks,
            depth_bias: depth_bias_command,
            sized_index_bind,
        };
        // Before the exit hook is registered for this device (below): the first call of each
        // entry point of the exit guard's table (`prime.rs`).
        // SAFETY: the device is live, of `physical`, and enables timeline semaphores, and
        // `bufferDeviceAddress` where `device_address` says so; `queues` holds at least one of
        // its queues, which nothing uses yet.
        if let Err(error) = unsafe {
            prime::prime(
                &instance.raw,
                physical,
                &Vk::new(
                    raw.clone(),
                    memory_reads,
                    push_descriptor,
                    object_name,
                    Default::default(),
                ),
                &loader_word,
                queue_family,
                &queues,
                Sharing::new(&queue_families),
                drawn.has(prime::Has {
                    timestamps: timestamp_bits > 0,
                    // A device the runtime created enables it (`enables.rs`).
                    host_query_reset: owned,
                    device_addresses: device_address,
                    generated_commands: generated_commands.is_some(),
                    sparse_binding: sparse.binds(),
                    push_descriptor: push_descriptor.is_some(),
                    calibration: calibrated.is_some(),
                    object_names: object_name.is_some(),
                    extended_dynamic_state: false,
                    primitive_restart: false,
                    color_write_masks: false,
                    depth_bias: false,
                    sized_index_bind: false,
                }),
                generated_commands.as_ref(),
                push_descriptor,
                calibrated,
                drawn,
            )
        } {
            scheduler.stop();
            completion.stop();
            contexts.stop();
            undo(&timelines);
            return Err(error);
        }
        // SAFETY: `physical` belongs to `instance`, created at Vulkan 1.2 or later (the compile
        // device exists, so the description's version passed the floor).
        let (identity, max_allocation) =
            unsafe { properties::query(&instance.raw, physical, instance.api_version) };
        let properties =
            DeviceProperties::new(identity, description, &compile, metal_level.level());
        // SAFETY: as above.
        let memory_properties =
            unsafe { instance.raw.get_physical_device_memory_properties(physical) };
        // Freed memory waits for the batches in flight, and a buffer of addresses goes with no
        // submission inside its `vkQueueSubmit`, where the driver fails pending work as memory a
        // device-address buffer lay in is freed, or a submission as such a buffer is destroyed
        // meanwhile (`freed_memory.rs`).
        let parking = [
            RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork,
            RuntimeDriverDefect::DestroyedAddressBufferFailsSubmission,
        ]
        .iter()
        .any(|defect| properties.driver_defects.contains(defect))
        .then(|| Arc::new(Parking::new(queues.len())));
        let mut memory = Memory::new(
            &memory_properties,
            max_allocation,
            device_address,
            properties.has_unified_memory,
            queue_families.clone(),
            parking.clone(),
            properties.description.driver_id,
        );
        memory.log_to(MemoryLog::new(instance.sink()));
        // SAFETY: as above.
        let physical_properties = unsafe { instance.raw.get_physical_device_properties(physical) };
        let limits = physical_properties.limits;
        let upload_alignment = [
            16,
            limits.min_storage_buffer_offset_alignment,
            limits.min_uniform_buffer_offset_alignment,
        ]
        .into_iter()
        .fold(1, u64::max);
        let textures = TextureState::new(
            &properties.description,
            limits.buffer_image_granularity,
            linear_color_attachment,
            compile.policy().multisample_depth_twin
                == metal2vulkan::device::MultisampleDepthTwinPolicy::Exact,
        );
        // SAFETY: as above; the device is live, made with `enabled` enabled, and `loader_word`
        // is its loader word.
        let host_import = unsafe {
            Importer::new(
                &instance.raw,
                physical,
                &raw,
                &enabled,
                super::buffer::usage(device_address),
                &loader_word,
                dma_buf,
                dma_buf_images,
            )
        };
        // The host image copy functions, where the device enabled `hostImageCopy` and copies to
        // and from `GENERAL` on the host: by their core names from Vulkan 1.4, else by the
        // extension's, which the device then enabled.
        // SAFETY: `physical` belongs to `instance`, created at Vulkan 1.2 or later.
        let general = host_image_copy
            && unsafe { crate::texture::host_copy::reaches_general(&instance.raw, physical) };
        let host_copies = general.then(|| {
            // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put
            // back.
            unsafe { loader_word.restore() };
            let handle = raw.handle();
            ash::ext::host_image_copy::DeviceFn::load(|name| {
                let bytes = name.to_bytes();
                let core = std::ffi::CString::new(bytes.strip_suffix(b"EXT").unwrap_or(bytes))
                    .unwrap_or_default();
                // SAFETY: the device is live; a name it does not serve answers null.
                let function = unsafe {
                    (instance.raw.get_device_proc_addr(handle, core.as_ptr()))
                        .or_else(|| instance.raw.get_device_proc_addr(handle, name.as_ptr()))
                };
                function.map_or(std::ptr::null(), |function| {
                    function as *const std::ffi::c_void
                })
            })
        });
        // The mesh draw commands, which exist where `meshShader` is enabled (it needs
        // `VK_EXT_mesh_shader`).
        let mesh_draws = if enabled
            .features
            .contains(&metal2vulkan::device::VkFeature::MeshShader)
        {
            // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put
            // back.
            unsafe { loader_word.restore() };
            let loaded = ash::ext::mesh_shader::Device::new(&instance.raw, &raw);
            Some(MeshDraws::new(&loaded, recording.as_ref()))
        } else {
            None
        };
        let budget = |header| {
            crate::gpu_pools::pool_budget(
                &memory_properties,
                limits.max_storage_buffer_range,
                max_allocation,
                header,
            )
        };
        let index_header = metal2vulkan::bind::IndexBoundsPass::POOL_HEADER_BYTES;
        // The index pool's bytes are a `u32`'s, its counter's.
        let index_pool_bytes = budget(index_header).min(u64::from(u32::MAX) / 4 * 4);
        let gpu_pools = GpuPools::new(index_pool_bytes, budget(0));
        let copy_images_bytes = crate::gpu_pools::device_share(&memory_properties);
        // From the host's clock where the options say, and on a driver whose timestamps wait for
        // every queue's whatever they say (`gpu_time.rs`).
        let from_host = if gpu_times_from_host {
            Some(FromHost::Asked)
        } else if properties
            .description
            .driver_facts()
            .contains(&metal2vulkan::device::DriverFact::TimestampsWaitForEveryQueue)
        {
            Some(FromHost::Driver)
        } else {
            None
        };
        // What completed work wrote is made visible to a batch by a barrier where semaphore
        // operations stall the driver's submissions, else by waits (`command_buffer/submit.rs`).
        let completed_by_barrier = properties
            .description
            .driver_facts()
            .contains(&metal2vulkan::device::DriverFact::SemaphoreOperationsStallSubmits);
        // One timestamp pool at most where each is one of Metal's scarce counter sample buffers.
        let scarce_pools = properties
            .description
            .driver_facts()
            .contains(&metal2vulkan::device::DriverFact::TimestampPoolsAreCounterSampleBuffers);
        let timing = Timing::new(
            timestamp_bits,
            limits.timestamp_period,
            calibrated,
            from_host,
            scarce_pools,
            // A command buffer's first encoder writes its start where its first batch can begin
            // with no command buffer of barriers (`gpu_time.rs`).
            !completed_by_barrier,
            // Its slots are reset on the host where the device enables `hostQueryReset`: every
            // device the runtime created (`enables.rs`).
            owned,
        );
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let functions = unsafe {
            loader_word.restore();
            Functions::new(
                &instance.raw,
                raw.clone(),
                memory_reads,
                push_descriptor,
                object_name,
                recording.as_ref(),
            )
        };
        // Its encoders' commands replay on a thread of its own where it has a slot (`recorder`).
        // SAFETY: the device's handle, live, its word the loader's (put back just now).
        let recorder = recording.as_ref().and_then(|slot| {
            Recorder::start(slot, &functions, unsafe {
                LoaderWord::of_device(raw.handle())
            })
        });
        let shader_cache = match ShaderCache::start(
            cache,
            &physical_properties,
            compile.profile_fingerprint(),
            (instance.settings, creation_feedback),
            instance.sink().cloned(),
        ) {
            Ok(shader_cache) => shader_cache,
            Err(error) => {
                scheduler.stop();
                completion.stop();
                contexts.stop();
                undo(&timelines);
                return Err(error);
            }
        };
        // The device's driver and layers are loaded and primed: the exit hook, registered (again)
        // after them when the device can have made statics since its last registration, runs
        // before their destructors. The device's identity: its driver's and its own UUIDs, and
        // the entry points its sections reach.
        // SAFETY: as above.
        let uuids = unsafe { properties::uuids(&instance.raw, physical) };
        let identity = uuids
            .iter()
            .flat_map(|uuid| uuid.chunks(8))
            .map(|half| u64::from_le_bytes(half.try_into().unwrap_or_default()))
            .chain(functions.entry_points().iter().copied())
            .collect();
        exit_guard::register_device(identity);
        let in_process = properties.kind == DeviceKind::Cpu;
        // A device whose driver reports a command buffer's execution failure through the
        // instance's messenger alone, and whose instance has the relay, has its queues' failures
        // relayed to the command buffers that failed (`relay.rs`), from its first batch on. The
        // question is the driver's, and the fact answers it, never the driver's name: a defect
        // fact, so a driver with a row takes it at every release and on every GPU.
        let lost = Arc::new(AtomicBool::new(false));
        let relays = (properties.description.driver_facts())
            .contains(&metal2vulkan::device::DriverFact::QueueFailuresReportedByMessenger);
        // Read before the properties move into the device's shared state (below): still keyed on
        // the driver's name, for the scheduler's wait alone.
        let moltenvk = properties.description.driver_id == Some(DriverId::MOLTENVK);
        if let (Some(relay), true) = (instance.relay(), relays) {
            for slot in &queues {
                let queue = slot.with(|queue| queue);
                // SAFETY: the queue is the device's, live, and its word was put back just now.
                let flights =
                    Arc::new(unsafe { Flights::new(queue, functions.clone(), lost.clone()) });
                if slot.flights.set(Arc::clone(&flights)).is_ok() {
                    relay.watch(flights);
                }
            }
        }
        let resource_ids: Arc<Mutex<ResourceIds>> = Arc::default();
        let sampler_ids = lock(&resource_ids).sampler_id_reads();
        let depth_ranges = DepthRanges::of(compile.profile());
        let shared = Arc::new(DeviceShared {
            instance,
            physical,
            functions: Arc::new(functions),
            queue_family,
            round_robin: queues.len() - present_queues,
            queues,
            queue_families,
            round_robin_made: AtomicU64::new(0),
            presenting_made: AtomicU64::new(0),
            compile,
            depth_ranges,
            enabled,
            extensions,
            properties,
            announced: Announced::new(&announce_each),
            lowering_refuses,
            deferred: crate::render::DeferredEncoders::default(),
            record_executions,
            views_epoch: AtomicU64::new(0),
            device_address,
            upload_alignment,
            dynamic_buffers: [
                limits.max_descriptor_set_uniform_buffers_dynamic,
                limits.max_descriptor_set_storage_buffers_dynamic,
            ],
            memory: Arc::new(Mutex::new(memory)),
            addresses: Mutex::new(AddressSpace::new()),
            recycled_buffers: Mutex::new(Recycler::new(recycle::BUFFERS)),
            resource_ids,
            keepers: Keepers::default(),
            identifiers: Arc::default(),
            sampler_ids,
            samplers: Mutex::default(),
            sampler_count: SamplerCount::new(limits.max_sampler_allocation_count),
            sampler_sets: Mutex::default(),
            heaps: Mutex::default(),
            heap_nan_residue: NanResidue::default(),
            function_tables: Mutex::default(),
            internal: Mutex::new(None),
            internal_done: OwnTimeline::default(),
            staging_done: OwnTimeline::default(),
            completed_by_barrier,
            by_ends,
            bound_at: Mutex::new(None),
            staging,
            scheduler,
            nudge,
            completion,
            fences: Fences::default(),
            contexts,
            loader_word,
            owned,
            lost,
            textures,
            shader_cache,
            host_import,
            shared_pointer_imports: OnceLock::new(),
            extended_dynamic_state,
            sized_index_bind,
            depth_bias_command,
            mesh_draws,
            push_descriptor,
            pushes_after_pipeline_binds,
            color_write_masks,
            _recording: recording,
            recorder,
            snapshot_fallbacks: Mutex::new(Vec::new()),
            reader_sizes: Mutex::new(Default::default()),
            gpu_pools,
            copy_images_bytes,
            max_framebuffer: [
                limits.max_framebuffer_width,
                limits.max_framebuffer_height,
                limits.max_framebuffer_layers,
            ],
            region_limits: RegionLimits::new(
                limits.max_viewport_dimensions,
                limits.viewport_bounds_range,
            ),
            layered_rate_map_released: AtomicBool::new(false),
            mutable_rate_map_called: AtomicBool::new(false),
            sparse_image_made: AtomicBool::new(false),
            owned_kernels: Mutex::new(OwnedKernels::default()),
            mipmap_scratch: Mutex::default(),
            planes: Mutex::new(PlanesObjects::default()),
            interleave: Mutex::default(),
            kept: Mutex::default(),
            shader_modules: Mutex::default(),
            pipeline_layouts: Mutex::default(),
            flights: crate::pipeline::flight::Flights::default(),
            color_resolves: Mutex::new(ColorResolveObjects::default()),
            stencil_copies: Mutex::new(StencilCopyObjects::default()),
            depth_copies: Mutex::new(DepthCopyObjects::default()),
            depth_stencil_resolves: Mutex::new(DepthStencilResolveObjects::default()),
            attachmentless: Mutex::default(),
            render_passes: Mutex::default(),
            interop: Interop::default(),
            heap_keys: Default::default(),
            rebinds: Default::default(),
            aliases: Default::default(),
            fresh: Default::default(),
            owners: Default::default(),
            live_buffers: Default::default(),
            #[cfg(unix)]
            residency: Default::default(),
            timing,
            view_reinterpretation,
            generated_commands,
            host_image_copy,
            host_copies,
            sparse,
            spares: Spares::default(),
            live_binds: Default::default(),
            bind_memos: Default::default(),
            kept_rooms: Default::default(),
            counts: Default::default(),
            parking: parking.map_or_else(OnceLock::new, OnceLock::from),
            uncommitted: Arc::default(),
            pipeline_pool: Default::default(),
            reaper: Default::default(),
            trails: Arc::default(),
            recoveries: Arc::default(),
            libraries: graphics_pipeline_library.then(Default::default),
        });
        // A CPU device runs the work submitted to it on the process's own threads: the exit hook
        // waits for that work before the destructors run under it (`exit_guard`).
        if in_process {
            let device: Weak<dyn exit_guard::InProcess> = Arc::<DeviceShared>::downgrade(&shared);
            exit_guard::runs_in_process(device);
        }
        // The scheduler's thread waits for the GPU itself where heads wait for signals to land,
        // but on MoltenVK, whose wait for any of several semaphores loses a signal that lands
        // while it registers a later one (`command_buffer/completion.rs`). Still keyed on the
        // driver's name: its home is a `RuntimeDriverDefect` beside
        // `ReadsCommandBuffersAfterTheirSignal`, which this commit does not open.
        if !moltenvk {
            (shared.scheduler).wait_on_gpu(Functions::clone(&shared.functions), shared.nudge);
        }
        let device = Device(shared);
        // The pages its replays held last go on its handler threads (`recorder`).
        if let Some(recorder) = device.shared().recorder() {
            recorder.set_elsewhere(device.elsewhere());
            // Woken by the scheduler's thread for what a commit queued (`recorder`).
            (device.shared().scheduler).wake_recorder(Arc::clone(recorder));
        }
        device.say_unmeasured_driver();
        device.degrade_argument_buffers_unread();
        device.degrade_float_tininess();
        if let Some((kind, notice)) = metal_level.announcement() {
            device.degrade(kind, || notice);
        }
        Ok(device)
    }
}

impl exit_guard::InProcess for DeviceShared {
    /// Wait on each `VkQueue`'s timeline for the value of the last batch submitted there.
    fn drain(&self, deadline: Option<Instant>) -> bool {
        let (timelines, values): (Vec<vk::Semaphore>, Vec<u64>) = self
            .queues
            .iter()
            .map(|queue| (queue.timeline, queue.submitted_value()))
            .filter(|&(_, value)| value != 0)
            .unzip();
        if timelines.is_empty() {
            return true;
        }
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&timelines)
            .values(&values);
        // No deadline is the longest timeout, `UINT64_MAX` nanoseconds.
        let timeout = deadline.map_or(u64::MAX, |deadline| {
            let left = deadline.saturating_duration_since(Instant::now());
            u64::try_from(left.as_nanos()).unwrap_or(u64::MAX)
        });
        // SAFETY: the timelines are the device's, live while this handle holds it; a wait for
        // values already signalled or submitted to be (each batch signals its queue's).
        let waited = unsafe { self.vk().wait_semaphores(&info, timeout) };
        waited.is_ok()
    }
}

/// The runtime error of a refusal by the compile layer's device: an unsupported device is
/// `Unsupported`, anything else (an inconsistent description) `Compile`.
pub(super) fn device_error(error: metal2vulkan::error::Error) -> Error {
    match error {
        metal2vulkan::error::Error::Unsupported(unsupported) => Error::Unsupported(unsupported),
        other => Error::Compile(other),
    }
}
