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
pub(crate) mod carry;
mod compiles;
pub(super) mod create;
pub(super) mod float_tininess;
pub(super) mod functions;
pub(crate) mod handles;
pub(crate) mod holds;
pub(super) mod instance_requirements;
pub(super) mod keeper;
pub(crate) mod line;
pub(super) mod prime;
pub(super) mod properties;
pub(crate) mod queue_quiet;
pub(crate) mod relay;
mod staging;
pub(super) mod submit;
pub(crate) mod submit_stalls;

pub use adopt::RawDevice;
pub(crate) use compiles::holding;
pub(crate) use create::{queue_limits_reached, QUEUE_LIMITS_REACHED};
pub use create::{
    DebugMessage, DeviceOptions, LayerSetting, LayerSettingValue, MessageKind, MessageSeverity,
    MessageSink, QueueLimits,
};
pub use handles::{DeviceHandles, ExtraQueue, ReservedQueue};
pub use properties::{DeviceKind, PhysicalDeviceIdentity};

use super::address::AddressSpace;
use super::attachmentless::KeptPasses;
use super::capabilities::DeviceProperties;
use super::color_resolve::ColorResolveObjects;
use super::command_buffer::fences::Fences;
use super::command_buffer::{Completion, Item};
use super::contexts::{Contexts, THREADS};
use super::degradation::{Announced, Degradation, Told};
use super::depth_copy::DepthCopyObjects;
use super::depth_stencil_resolve::DepthStencilResolveObjects;
use super::draw_source::{sized_index_bind, SizedIndexBind};
use super::dynamic_state::{ColorWriteMasks, ExtendedDynamicState};
use super::error::{internal, vulkan, Error};
use super::exit_guard::{self, Call};
use super::freed_memory::Parking;
use super::function_table::TableShared;
use super::generated_commands::Commands;
use super::gpu_pools::GpuPools;
use super::gpu_time::{self, FromHost, Timing};
use super::heap_sets::DeviceHeaps;
use super::host_memory::Importer;
use super::interop::Interop;
use super::loader_word::{DeviceWord, LoaderWord};
use super::memory::Memory;
use super::memory_log::MemoryLog;
use super::mipmap_scratch::MipmapScratch;
use super::owned_kernel::OwnedKernels;
use super::pending_audit::PendingAudit;
use super::recycle::{self, BufferMaking, Recycler};
use super::released_programs::Keepers;
use super::render_state::DepthRanges;
use super::render_target::KeptRenderPasses;
use super::resource_ids::{PassingHolds, ResourceIds, SamplerIdReads};
use super::sample_planes::PlanesObjects;
use super::sampler::{LiveSamplers, SamplerCount};
use super::scheduler::Scheduler;
use super::shader_cache::driver::Making;
use super::shader_cache::{found_in_cache, CacheSetup, ShaderCache, REPORTED_STAGES};
use super::sharing::Sharing;
use super::spares::Spares;
use super::sparse::device::SparseDevice;
use super::stencil_copy::StencilCopyObjects;
use super::texture::TextureState;
use super::unique_identifier::UniqueIdentifiers;
use crate::image_barrier::MemoryReads;
use crate::recorder::{Command, MeshDraws, Slot};
use crate::release;
use ash::vk::{self, Handle};
use functions::{Functions, Vk};
use metal2vulkan::bind::RegionLimits;
use metal2vulkan::device::{DeviceDescription, EnableSet};
use metal2vulkan::error::RuntimeDriverDefect;
use relay::Flights;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Instant;
use submit::{Internal, OwnTimeline};

/// A Metal device (`MTLDevice`) on a Vulkan device. Cheap to clone (a retain); `Send + Sync`.
///
/// The program's last drop of its handles (the device's own and its objects'), on a thread of the
/// program's, stops the runtime's own background work on the device and waits for it: at most the
/// round of its check of textures over host pages in progress, or the record its warm pass is
/// making ([`Device::warm`]). So the device is torn down on that thread before the drop returns,
/// and a consumer that adopted the `VkDevice` ([`Device::from_raw`]) may destroy it then.
pub struct Device(
    Pin,
    /// The handle's place in the registry of the runtime's own holds (`holds.rs`).
    holds::Share,
);

/// What a device handle holds (`line.rs`): one generation, or the device's line alone.
pub(crate) enum Pin {
    /// A generation's handle: the device of the objects made on it, whose Vulkan calls reach that
    /// generation's `VkDevice` alone.
    Generation(Arc<DeviceShared>),
    /// A root handle: the program's device, whose calls go to the generation that serves.
    Root(Arc<RootCell>),
}

impl Pin {
    /// The address of the cell it holds, a generation's or a root's: two handles of one cell
    /// hold the same address, and no generation's is a root's.
    #[inline(always)]
    fn address(&self) -> *const () {
        match self {
            Pin::Generation(shared) => Arc::as_ptr(shared).cast(),
            Pin::Root(root) => Arc::as_ptr(root).cast(),
        }
    }
}

/// What a root handle holds, its clones sharing it: the device's line, and the generation a reach
/// of the handle's state pinned, for the cell's life ([`Device::shared`]). The public calls never
/// reach it; a test's may.
pub(crate) struct RootCell {
    pub(super) line: Arc<line::Line>,
    pinned: std::sync::OnceLock<Arc<DeviceShared>>,
}

impl RootCell {
    /// A root handle's cell of `line`, pinning nothing.
    pub(super) fn new(line: Arc<line::Line>) -> Arc<RootCell> {
        Arc::new(RootCell {
            line,
            pinned: std::sync::OnceLock::new(),
        })
    }
}

/// A handle on a device that does not keep it ([`Device::downgrade`]), with the registry of its
/// holds (`holds.rs`).
#[derive(Clone)]
pub(crate) struct WeakDevice(Weak<DeviceShared>, Arc<holds::Holds>);

impl WeakDevice {
    /// The device, while it lives.
    pub(crate) fn upgrade(&self) -> Option<Device> {
        holds::upgrade(self)
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name())
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

/// Everything one device's objects share: one generation of the device's line, over its own
/// `VkDevice` (`line.rs`).
pub(super) struct DeviceShared {
    /// The line the generation is of. Its state holds the generation it serves, a cycle the
    /// program's last drop of its handles breaks (`holds.rs`, `Line::release`).
    pub(crate) line: Arc<line::Line>,
    /// The line's counts of its generations, read without the line (`line::Lineage`).
    pub(crate) lineage: Arc<line::Lineage>,
    /// The generation's number on its line: 0 for the first `VkDevice`.
    pub(crate) generation: u64,
    /// The instance outlives the device.
    pub(super) instance: Arc<InstanceShared>,
    pub(super) physical: vk::PhysicalDevice,
    /// The device's functions: through [`DeviceShared::vk`] only, or a share that holds nothing
    /// of the device ([`DeviceShared::functions`]).
    functions: Arc<Functions>,
    /// The primary queue family: its first queue's, where internal submissions go.
    pub(super) queue_family: u32,
    /// The `VkQueue`s the runtime submits to, the primary family's first queue first: at least
    /// one. Shared with the scheduler's thread where it takes the driver's stalls idle
    /// (`submit_stalls.rs`).
    pub(super) queues: Arc<[QueueSlot]>,
    /// The families of those queues and of the caller's reserved queues, the primary first, each
    /// once: what the device's buffers and images are shared among (`sharing.rs`).
    pub(super) queue_families: Vec<u32>,
    /// Those families, then the staging queue's where the device reserved one: what the device's
    /// buffers are shared among (`sharing.rs`).
    pub(super) buffer_families: Vec<u32>,
    /// How many of `queues`, the first, logical queues share round robin; the rest are reserved
    /// for presenting queues.
    round_robin: usize,
    /// The logical queues made round robin so far: the next one takes `VkQueue` this modulo
    /// `round_robin`.
    round_robin_made: AtomicU64,
    /// The presenting queues made so far on the reserved `VkQueue`s: the next one takes the
    /// reserved queue this modulo their number.
    presenting_made: AtomicU64,
    /// Where the device's queue limits changed its queues (`DeviceOptions::queue_limits`), the
    /// queue plan's round robin and presenting `VkQueue`s: what the device's notes of the limits
    /// compare with (`Device::next_queue`).
    planned: Option<(usize, usize)>,
    /// The driver refused the device with its queue plan's `VkQueue`s, and it took its least
    /// plan's: the reason a presenting queue shares one (`Device::place_queue`).
    pub(super) least_plan: bool,
    /// The presenting queues made so far, on a reserved `VkQueue` or not, counted only where
    /// `planned` is.
    presenting_asked: AtomicU64,
    /// The program's command queues made round robin so far, unshared ones that took the round
    /// robin's next among them, counted only where `planned` is: the runtime's own are not.
    program_queues_made: AtomicU64,
    /// Whether its command queues share `VkQueue`s, and its check's verdict
    /// (`queue_share.rs`). Shared with the check's thread, which holds no device handle
    /// (`queue_switch_check.rs`).
    pub(crate) queue_switch: Arc<crate::queue_share::QueueSwitch>,
    /// What the compile device's profile takes of viewport depth ranges past `[0, 1]`, which draws read
    /// (`render_state::native_depth_range`).
    pub(super) depth_ranges: DepthRanges,
    /// What every generation of its line answers alike, its own copy: a generation handle reads
    /// it here, at one hop as it reads the rest (`line.rs`).
    invariant: line::Invariant,
    /// The kinds of degradation the device announced (`degradation.rs`): its line's, so a kind is
    /// announced once across its recoveries (`line.rs`).
    pub(super) announced: Arc<Announced>,
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
    /// The Metal address space, the line's: every generation's buffers take their addresses from
    /// it, so none made after a recovery takes one an object made before holds (`line.rs`).
    pub(super) addresses: Arc<Mutex<AddressSpace>>,
    /// Parked buffers of the runtime's own memory, and the makings of the live recyclable ones
    /// (`recycle.rs`).
    pub(super) recycled_buffers: Mutex<Recycler<vk::Buffer, BufferMaking>>,
    /// The resource IDs of textures and sampler states (`resource_ids.rs`).
    pub(super) resource_ids: Arc<Mutex<ResourceIds>>,
    /// The runtime's passing holds of the program's textures and sampler states, which a
    /// program's making waits out before it takes its ID (`resource_ids.rs`): the line's, as the
    /// IDs are.
    pub(super) passing_holds: Arc<PassingHolds>,
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
    pub(super) samplers: Mutex<LiveSamplers>,
    /// The device's live Vulkan samplers, held to `maxSamplerAllocationCount`
    /// (`DeviceShared::create_sampler`).
    pub(super) sampler_count: SamplerCount,
    /// The device's live `VkDeviceMemory` allocations, held to `maxMemoryAllocationCount`
    /// (`allocations.rs`).
    pub(crate) allocations: crate::allocations::Registered,
    /// The thread the blocks of its memory that go are freed on (`free_thread.rs`).
    pub(crate) frees: Arc<crate::free_thread::FreeThread>,
    /// The immutable sampler sets of the variants' layouts, by their samplers
    /// (`pipeline/objects.rs`, `pipeline/content_keyed.rs`).
    pub(crate) sampler_sets: Mutex<crate::pipeline::objects::SamplerSets>,
    /// The heap sets and the texture table every command buffer shares (`heap_sets.rs`). Its
    /// lock is taken before `resource_ids`', never under it.
    pub(super) heaps: Mutex<DeviceHeaps>,
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
    /// The caller's extra queues (`DeviceOptions::extra_queues`), to which the runtime never
    /// submits (`handles.rs`).
    pub(super) extra_queues: Vec<handles::ExtraQueue>,
    /// The queues of its own families reserved for the caller (`DeviceOptions::reserved_queues`),
    /// outside `queues`: the runtime never submits to them (`handles.rs`).
    pub(super) reserved_queues: Vec<handles::ReservedQueue>,
    pub(super) scheduler: Scheduler,
    /// The scheduler's nudge, a timeline semaphore its wakers signal where its thread waits for
    /// the GPU (`scheduler/mod.rs`): made on every device, used where the driver's wait for any
    /// of several semaphores loses no signal (not MoltenVK's).
    nudge: vk::Semaphore,
    pub(super) completion: Completion,
    /// The idle fences the batches that end command buffers signal (`command_buffer/fences.rs`).
    /// Shared with the queue-switch check's thread, whose uses of a queue take them where its
    /// failures are relayed (`queue_switch_check.rs`).
    pub(super) fences: Arc<Fences>,
    /// The queues' serial contexts: every command buffer's transitions and handlers run there,
    /// and the releases handed off a thread that may run none of the program's code
    /// ([`Device::let_go_elsewhere`]).
    pub(super) contexts: Contexts<Item>,
    /// Shared with the queue-switch check's thread, which allocates command buffers
    /// (`queue_switch_check.rs`).
    pub(super) loader_word: Arc<DeviceWord>,
    /// The runtime created the `VkDevice` and destroys it.
    owned: bool,
    /// The device was lost: no further submission. Shared with the relay (`relay.rs`), which
    /// records a loss it hears of.
    lost: Arc<AtomicBool>,
    pub(super) textures: super::texture::TextureState,
    /// The translated-shader cache on disk and the driver's pipeline cache.
    pub(super) shader_cache: ShaderCache,
    /// Its check of the first move of an image over host pages, per class of pages, and its
    /// verdicts (`in_place.rs`).
    pub(crate) in_place: crate::in_place::InPlaceChecks,
    /// Its checks of what its submissions cost and whether its pointer imports follow their
    /// address, and their verdicts (`submit_check.rs`).
    pub(crate) submit_checks: Arc<crate::submit_check::SubmitChecks>,
    /// How no-copy buffers import host memory, or why they are refused.
    pub(super) host_import: Result<Importer, metal2vulkan::error::Unsupported>,
    /// How it exports an exportable buffer's memory as an opaque file descriptor; `None` where
    /// it enables no `VK_KHR_external_memory_fd` or exports no buffer's memory so
    /// (`exported_memory.rs`).
    #[cfg(unix)]
    pub(super) opaque_fd: Option<crate::exported_memory::FdExport>,
    /// How it exports a dma-buf-exportable buffer's memory as a dma-buf; `None` where it enables
    /// not the dma-buf route's extensions and `VK_EXT_queue_family_foreign`, or exports no
    /// buffer's memory so (`exported_memory.rs`).
    #[cfg(unix)]
    pub(super) dma_buf_export: Option<crate::exported_memory::FdExport>,
    /// How it exports an exportable shared event's timeline semaphore as an opaque file
    /// descriptor; `None` where it enables no `VK_KHR_external_semaphore_fd` or exports no
    /// timeline semaphore so (`exported_semaphore.rs`).
    #[cfg(unix)]
    pub(super) opaque_fd_semaphores: Option<crate::exported_semaphore::OpaqueFdSemaphores>,
    /// Whether the device imports a pointer to shared memory, once asked
    /// (`Device::imports_shared_host_pointers`).
    pub(super) shared_pointer_imports: OnceLock<bool>,
    /// The commands of a render pipeline's extended dynamic states, where the device has them
    /// (`dynamic_state.rs`).
    pub(super) extended_dynamic_state: Option<ExtendedDynamicState>,
    /// The command that binds an index buffer range of a size, where the device enabled
    /// `maintenance5` (`draw_source.rs`).
    pub(super) sized_index_bind: Option<SizedIndexBind>,
    /// The mesh draw commands, where the device enabled `meshShader` (`render/mesh.rs`).
    pub(super) mesh_draws: Option<MeshDraws>,
    /// `vkCmdPushDescriptorSet`, from Vulkan 1.4 or `VK_KHR_push_descriptor`
    /// (`binding::push_set`). Where the driver has `RuntimeDriverDefect::PushedBufferSizesUnset`
    /// only the variants `pipeline::objects::pushes_set_zero` names push with it there, and no
    /// runtime kernel (`OwnedKernel::pushed`).
    pub(super) push_descriptor: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
    /// A draw after a pipeline bind pushes at least one of set 0's descriptors, the first where
    /// none changed (`render/bound.rs`), where the driver has
    /// `RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`.
    pub(super) pushes_after_pipeline_binds: bool,
    /// The command that sets a draw's color write masks, where the device enabled everything a
    /// draw that reads its own pass's texture through a snapshot takes (`render/snapshot.rs`).
    pub(super) color_write_masks: Option<ColorWriteMasks>,
    /// The device's slot among those its commands go through (`recorder`), where it took one:
    /// let go of after `vkDestroyDevice` (it drops after the teardown, `Drop`). Its counts are
    /// `WorkCounts::commands_recorded`.
    recording: Option<Slot>,
    /// The reasons a draw that reads its pass's texture took the split that the device said
    /// already, each with the clauses that left its slots out where it is
    /// `SplitReason::UnreadableOwnPixel`, where `METAL2VULKAN_LOG_PIPELINES` is set
    /// (`render/snapshot.rs`).
    pub(super) snapshot_fallbacks:
        Mutex<Vec<(crate::render::SplitReason, crate::work_counts::Unreadable)>>,
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
    /// How much GPU work of the program's own the device has been given: every commit of a command
    /// buffer raises it, and so does the program's own use of a `VkQueue`
    /// (`CommandQueue::with_vk_queue`). A texture's host shadow holds the image's texels only
    /// while it stands where it stood when the shadow was read (`texture/host_shadow.rs`): a
    /// commit is the only way a command of the program's reaches the device, so this one count
    /// stands for every route its GPU work could write a texture by.
    gpu_work: AtomicU64,
    /// Whether the device takes a set-0 buffer binding whose entry point uses it through more
    /// than one variable as a **dynamic** descriptor, asked of the device the first time a
    /// variant has such a binding and kept for its lifetime
    /// (`pipeline::objects::takes_multi_variable_dynamic`). `None` until then, and on a device no
    /// variant has asked.
    pub(super) dynamic_multi_variable: OnceLock<bool>,
    /// The crate-owned compute kernels the runtime dispatches for itself, made on first use
    /// (`owned_kernel.rs`).
    pub(super) owned_kernels: Mutex<OwnedKernels>,
    /// The two-plane YCbCr formats the device converts itself (`ycbcr_device.rs`).
    pub(super) ycbcr_formats: crate::ycbcr_device::YCbCrFormats,
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
    /// The guest RAMs it uses, and their chunks it imported (`host_ram.rs`).
    pub(crate) host_ram: super::host_ram::RamSource,
    /// Its copied host buffers, compared at each commit and each batch's completion
    /// (`copied.rs`).
    pub(crate) copied: Arc<super::copied::CopiedBuffers>,
    /// How many of the next imports of host pages by their pointer fail as the driver's out of
    /// device memory: for the tests of `copied.rs` (`Device::fail_next_host_imports`).
    #[cfg(test)]
    pub(crate) failing_host_imports: std::sync::atomic::AtomicU64,
    /// How many of the next `vkQueueSubmit`s of its command buffers' work fail as the driver's
    /// out of device memory, never made: for the device tests of a failed submission
    /// (`Device::fail_next_submissions`).
    #[cfg(test)]
    pub(crate) failing_submissions: std::sync::atomic::AtomicU64,
    /// Every linear image of a render target renders as a color attachment only under
    /// `VK_NV_linear_color_attachment`'s rule, whatever the device's facts: for the tests of
    /// `merge_scratch.rs` on a device that renders one beside any other
    /// (`linear_attachment::bound`).
    #[cfg(test)]
    pub(crate) linear_rule_forced: AtomicBool,
    /// `STORAGE` costs every image whose plan lacks it something (`Device::if_free`), whatever
    /// the device lays out: for the tests of `tests/self_read_storage.rs` on a device that gives a
    /// render target `STORAGE` for nothing.
    #[cfg(test)]
    pub(crate) storage_costs_forced: AtomicBool,
    /// The route a test takes for images over dma-buf pages, whatever the driver answers of an
    /// explicit plane's offset (`dma_buf.rs`'s `PlaneRoute`, as its `u8`): for the tests of
    /// `tests/dma_buf_plane_offset.rs` on a driver that answers the offset asked.
    #[cfg(test)]
    pub(crate) plane_route_forced: std::sync::atomic::AtomicU8,
    /// The logs of its shadows over host pages a consumer logs the writes to
    /// (`dirty_pages.rs`).
    pub(super) dirty: super::dirty_pages::DirtyShadows,
    /// The writes that let go of its shadows' standings across command buffers
    /// (`shadow_standing.rs`).
    pub(super) standings: super::shadow_standing::Standings,
    /// The new textures that read zero where Metal reads its pattern, until a command observes
    /// or covers that (`fresh_unwritten.rs`).
    pub(super) fresh: super::fresh_unwritten::FreshUnwritten,
    /// The program's resources by key, with what owns their memory (`used_resources.rs`).
    pub(super) owners: super::used_resources::Owners,
    /// The program's live buffers of their own, and their resident set (`live_buffers.rs`).
    pub(super) live_buffers: super::live_buffers::LiveBuffers,
    /// What its recovery carries to the next generation, and what it lets go of once nothing
    /// records on it (`carry.rs`).
    pub(crate) carry: carry::Carry,
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
    /// The device enabled `zeroInitializeDeviceMemory` (`create.rs`): memory it allocates with
    /// `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` reads zero, and an image made in such memory
    /// starts in `VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT` (`texture/host_copy.rs`).
    pub(super) zero_initialize_device_memory: bool,
    /// What it has of sparse binding and residency (`sparse/device.rs`).
    pub(super) sparse: SparseDevice,
    /// The command pools and descriptor pools command buffers that completed gave back, for the
    /// next to take (`spares.rs`).
    pub(super) spares: Spares,
    /// The bind states its ended render encoders left, which the next ones start from
    /// (`live_bind.rs`).
    pub(super) live_binds: crate::live_bind::SpareLiveBinds,
    /// The occlusion query pools no render pass holds, which a pass that measures what its
    /// readers cover takes (`render/coverage.rs`).
    pub(super) coverage_pools: crate::render::coverage::SparePools,
    /// The bind states its ended compute encoders left, which the next ones start from
    /// (`compute/live.rs`).
    pub(super) live_dispatches: crate::compute::SpareLiveDispatches,
    /// The usage shadows its textures made (`texture/usage_shadow.rs`): a kept list of resident
    /// textures' shadows is listed again where it moved (`shadow::ResidentShadows`).
    pub(super) usage_shadows_made: std::sync::atomic::AtomicU64,
    /// One past the highest resource ID a texture is placed at in its heap sets, 0 for none,
    /// stored under `heaps` as it changes and read without it (`heap_sets::TextureTableAt`).
    pub(super) heap_textures_end: std::sync::atomic::AtomicU64,
    /// The bind memos its ended render encoders left, which the next ones take (`binding.rs`).
    pub(super) bind_memos: crate::binding::SpareBindMemos,
    /// The boxes of its ended render encoders' cores, emptied, which the next ones take
    /// (`encoder::SpareCores`).
    pub(super) cores: crate::encoder::SpareCores,
    /// The rooms its ended render encoders' draws gathered into, which the next ones take
    /// (`render/draw_room.rs`).
    pub(super) draw_rooms: crate::render::SpareDrawRooms,
    /// The rooms of the draws its render encoders kept for instances recorded inline after a
    /// split, which the next such instances take (`render/inline_splits.rs`).
    pub(super) kept_rooms: crate::render::SpareKept,
    /// Counts of its commit path's work, for the lanes (`work_counts.rs`). Shared with the
    /// queue-switch check's thread (`queue_switch_check.rs`).
    pub(super) counts: Arc<super::work_counts::Counters>,
    /// Its uses of its `VkQueue`s seen as one, which its blocks made ahead wait on for a moment
    /// right after a burst of submissions (`queue_quiet.rs`, `memory/ready.rs`). Shared with
    /// each of its queues' slots.
    pub(crate) queue_quiet: Arc<queue_quiet::QueueQuiet>,
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
    pub(super) reaper: Arc<crate::pipeline::reaper::Reaper>,
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
    /// Its pipeline records: the hook they go to, and the counts (`warm/record.rs`).
    pub(crate) pipeline_recording: crate::warm::Recording,
    /// Its warm calls: the records queued, its warm thread, and the counts (`warm/pass.rs`).
    pub(crate) warmer: crate::warm::Warmer,
    /// The registry of the runtime's own holds of the device, which its handles share
    /// (`holds.rs`).
    pub(crate) holds: Arc<holds::Holds>,
}

impl DeviceShared {
    /// Wait each of the device's queues idle, each under its lock, as every use of a queue is
    /// made: a thread of the device's may submit meanwhile (its submission checks,
    /// `submit_check.rs`), where `vkDeviceWaitIdle` would need every queue externally
    /// synchronized (vkspec 7.8, "Host access to all VkQueue objects created from device that are
    /// not created with VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR must be externally
    /// synchronized"). A wait on a lost device returns in finite time (vkspec 5.2.3). Called
    /// inside a section of the exit guard.
    pub(super) fn queues_idle(&self) {
        let raw = self.vk();
        for slot in self.queues.iter() {
            // SAFETY: the queue is the device's, used under its lock, inside a section. Its
            // answer is not needed: the device is lost, or the work it waited for ended.
            let _ = slot.with(|queue| unsafe { raw.queue_wait_idle(queue) });
            exit_guard::called(Call::QueueWaitIdle);
        }
    }
}

impl DeviceShared {
    /// Whether the generation's `VkDevice` was lost.
    pub(crate) fn lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Whether `kind`, met while lowering a render encoder's call after it returned, refuses the
    /// call (`DeviceOptions::lowering_refuses`).
    pub(crate) fn lowering_refuses(&self, kind: Degradation) -> bool {
        self.lowering_refuses.contains(&kind)
    }

    /// What `create` makes with the `VkPipelineCache` it is handed, its `vkCreate*Pipelines` call:
    /// the device's cache of the record `making` names, seeded from that record
    /// (`ShaderCache::make_pipelines`, `shader_cache/driver.rs`). `create` returns, with what it
    /// made, whether the driver found its pipeline in the cache it was handed
    /// (`shader_cache::found_in_cache`, `None` where it reports nothing). A making from its record
    /// that the driver reports found reads no data back and writes nothing
    /// (`shader_cache/driver.rs`). On a pool's thread the exit waits for it longer
    /// (`exit_guard::driver_making`).
    pub(crate) fn make_pipelines_reported<T, E>(
        &self,
        making: Making,
        create: impl FnOnce(vk::PipelineCache) -> (Result<T, E>, Option<bool>),
    ) -> Result<T, E> {
        // What every making is measured to report, withheld inside it (`driver_facts.rs`); a
        // render pipeline's making expects them with its own.
        let expected = crate::driver_facts::making_diagnostics(&self.invariant.properties);
        crate::driver_facts::expecting_where_unset(expected, || {
            crate::exit_guard::driver_making(|| {
                self.shader_cache.make_pipelines(self.vk(), making, create)
            })
        })
    }

    /// The compute pipeline of `info`, made as
    /// [`make_pipelines_reported`](Self::make_pipelines_reported) makes one under the record
    /// `making` names, with what its driver reports of it where the device's makings report
    /// (`VK_EXT_pipeline_creation_feedback`, counted in `driver_pipeline_hits` and `misses`): one
    /// found in a cache loaded from its record reads no data back and writes nothing, whatever
    /// bytes the driver's data holds after it. A program's compute pipelines and the runtime's
    /// own kernels.
    pub(crate) fn make_compute_pipeline(
        &self,
        making: Making,
        info: vk::ComputePipelineCreateInfo<'_>,
    ) -> Result<vk::Pipeline, Error> {
        let reports = self.shader_cache.driver_reports();
        let mut whole = vk::PipelineCreationFeedback::default();
        let mut stage = [vk::PipelineCreationFeedback::default()];
        let made = self.make_pipelines_reported(making, |cache| {
            let mut feedback = vk::PipelineCreationFeedbackCreateInfo::default()
                .pipeline_creation_feedback(&mut whole)
                .pipeline_stage_creation_feedbacks(&mut stage);
            let mut info = info;
            if reports {
                info = info.push_next(&mut feedback);
            }
            // SAFETY: what `info` points to outlives it, and so the call; the cache is the
            // device's of the record, or none.
            let made = unsafe { self.vk().create_compute_pipelines(cache, &[info], None) };
            let found = reports.then(|| found_in_cache(&whole, &stage)).flatten();
            (made, found)
        });
        self.made_one(made, reports.then_some((&whole, &stage[..])))
    }

    /// The graphics pipeline, or graphics pipeline library, of `info`, made and reported as
    /// [`make_compute_pipeline`](Self::make_compute_pipeline) makes a compute pipeline: a library
    /// of shaders of a render pipeline linked from libraries (`pipeline/graphics.rs`), and the
    /// runtime's own draws (its copies, resolves and planes steps).
    pub(crate) fn make_graphics_pipeline(
        &self,
        making: Making,
        info: vk::GraphicsPipelineCreateInfo<'_>,
    ) -> Result<vk::Pipeline, Error> {
        let reports = self.shader_cache.driver_reports();
        // Its stages' feedback where it has at most `REPORTED_STAGES`, none where more (a count of
        // 0 is valid, one equal to `stageCount` else:
        // VUID-VkGraphicsPipelineCreateInfo-pipelineStageCreationFeedbackCount-06594).
        let mut whole = vk::PipelineCreationFeedback::default();
        let mut stages = [vk::PipelineCreationFeedback::default(); REPORTED_STAGES];
        let count = info.stage_count as usize;
        let reported = if count <= REPORTED_STAGES { count } else { 0 };
        let made = self.make_pipelines_reported(making, |cache| {
            let mut feedback = vk::PipelineCreationFeedbackCreateInfo::default()
                .pipeline_creation_feedback(&mut whole)
                .pipeline_stage_creation_feedbacks(&mut stages[..reported]);
            let mut info = info;
            if reports {
                info = info.push_next(&mut feedback);
            }
            // SAFETY: as `make_compute_pipeline`'s.
            let made = unsafe { self.vk().create_graphics_pipelines(cache, &[info], None) };
            let found = reports
                .then(|| found_in_cache(&whole, &stages[..reported]))
                .flatten();
            (made, found)
        });
        self.made_one(made, reports.then_some((&whole, &stages[..reported])))
    }

    /// The one pipeline a making `made`, with what its driver reported of it, `reported`, counted
    /// where the making chained the reports and made it.
    fn made_one(
        &self,
        made: Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
        reported: Option<(
            &vk::PipelineCreationFeedback,
            &[vk::PipelineCreationFeedback],
        )>,
    ) -> Result<vk::Pipeline, Error> {
        let made = made.map_err(|(_, error)| vulkan(error))?;
        if let Some((whole, stages)) = reported {
            self.shader_cache.driver_reported(whole, stages);
        }
        made.first()
            .copied()
            .ok_or_else(|| internal("the driver made no pipeline"))
    }

    /// The device's functions. The runtime's own threads reach them only inside a section of
    /// the exit guard, or while they run the program's code, and a section reaches only the exit
    /// guard's table of entry points; builds with debug assertions end the process otherwise
    /// (`functions.rs`).
    pub(crate) fn vk(&self) -> &Vk {
        self.functions.get()
    }

    /// The device's pending audit, where it is on (`pending_audit.rs`): read with no call of the
    /// device's, on any thread; its checks read the device's timelines through [`vk`](Self::vk).
    pub(crate) fn pending_audit(&self) -> Option<&PendingAudit> {
        self.functions.pending_audit()
    }

    /// Nanoseconds the device's pipeline makings spent inside its driver, graphics then compute
    /// (`compiles.rs`, `WorkCounts::render_pipeline_create_nanos` and
    /// `compute_pipeline_create_nanos`).
    pub(super) fn pipeline_create_nanos(&self) -> (u64, u64) {
        compiles::driver_nanos(self.vk().handle())
    }

    /// How much GPU work of the program's own the device has been given
    /// ([`gpu_work`](Self::gpu_work) field): the count a texture's host shadow is read against
    /// (`texture/host_shadow.rs`).
    pub(crate) fn gpu_work(&self) -> u64 {
        self.gpu_work.load(Ordering::Acquire)
    }

    /// The program gave the device GPU work of its own: a command buffer's commit, or its own use
    /// of a `VkQueue`. Called before the work can run, so a host shadow read after it is read
    /// against the new count (`texture/host_shadow.rs`).
    pub(crate) fn program_gave_gpu_work(&self) {
        self.gpu_work.fetch_add(1, Ordering::AcqRel);
    }

    /// A share of the device's functions, which holds nothing of the device: for the releases
    /// posted to its handler threads, which its teardown waits for (`freed_memory.rs`). Reached
    /// through [`Functions::get`] alone, as [`vk`](Self::vk) reaches them.
    pub(crate) fn functions(&self) -> Arc<Functions> {
        Arc::clone(&self.functions)
    }

    /// The commands recorded through its slot so far, by kind (`recorder`); none where it took
    /// no slot.
    pub(crate) fn commands_recorded(&self) -> crate::CommandsRecorded {
        self.recording
            .as_ref()
            .map_or_else(Default::default, Slot::counts)
    }

    /// The recorder's slot the device's commands go through, where it took one (`recorder`).
    pub(crate) fn recording(&self) -> Option<&Slot> {
        self.recording.as_ref()
    }

    /// `vkAllocateCommandBuffers` of `info`: the loader serves it and reads the device's loader
    /// word first ([`DeviceWord::restored`]). Counted by the place that asked, for the crate's
    /// device tests (`Spares::churn`).
    #[cfg_attr(test, track_caller)]
    pub(super) fn allocate_command_buffers(
        &self,
        info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, Error> {
        // SAFETY: the device is live; `info` names one of its pools, which the caller holds
        // exclusively.
        let allocated = unsafe {
            self.loader_word
                .restored(|| self.vk().allocate_command_buffers(info))
        };
        exit_guard::called(Call::AllocateCommandBuffers);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if allocated.is_ok() {
            (self.spares).allocated(std::panic::Location::caller(), info.command_buffer_count);
        }
        allocated.map_err(super::error::vulkan)
    }

    /// Name `pipeline`, just made and held by this thread alone, `name` for the layers' messages,
    /// where the device's instance reports messages (`Vk::object_name`): the loader serves the
    /// call and reads the device's loader word first ([`DeviceWord::restored`]). A failure to name it is
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
            self.loader_word
                .restored(|| object_name(functions.handle(), &info))
        };
        exit_guard::called(Call::SetDebugUtilsObjectName);
    }

    /// How the device's buffers and images are shared among its queue families (`sharing.rs`).
    pub(crate) fn sharing(&self) -> Sharing<'_> {
        Sharing::with_buffers(&self.queue_families, &self.buffer_families)
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

    /// How many of its `VkQueue`s, the first, the round robin lends logical queues; the rest are
    /// reserved for presenting queues.
    pub(crate) fn round_robin(&self) -> usize {
        self.round_robin
    }

    /// Where the device's queue limits changed its round robin's or presenting queues'
    /// `VkQueue`s from the queue plan's (`DeviceOptions::queue_limits`), how many it took of
    /// each and how many the plan gives.
    pub(crate) fn queues_limited(&self) -> Option<((usize, usize), (usize, usize))> {
        let planned = self.planned?;
        let taken = (self.round_robin, self.queues.len() - self.round_robin);
        (taken != planned).then_some((taken, planned))
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
        // meanwhile waits for the teardown, which leaves the rest at its next step (below,
        // `exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // The queue-switch check holds nothing of the device either: it stops at its next step,
        // and its objects go before the wait returns (`queue_switch_check.rs`).
        self.queue_switch.stop();
        // So do the submission checks (`submit_check.rs`).
        self.submit_checks.stop();
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
        // An exit that began meanwhile waits for this teardown (`exit_guard`): from here on the
        // device, its objects and its memory are left to the process's end, as where the exit
        // began first, so that the exit waits for no more of the teardown than its step in
        // progress (the wait for idle, and `vkDestroyDevice` above all, take long beside other
        // processes' work on the GPU).
        if exit_guard::closed() {
            return;
        }
        // SAFETY: the device is live; waiting for idle covers any work the consumer of an
        // adopted device did not wait for before dropping the runtime's last handle. No batch is
        // pending then, so no timeline is in use.
        unsafe {
            let _ = self.vk().device_wait_idle();
            exit_guard::called(Call::DeviceWaitIdle);
            // The scheduler's thread, which waited on its nudge, has stopped.
            release::object(self.vk(), self.nudge);
            for queue in self.queues.iter() {
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
            // Nor a coverage query pool: each pass's came back as its command buffer finished.
            self.coverage_pools.release(self.vk());
        }
        // Nor a kept barrier command buffer, nor a submission taking a driver's stall idle: the
        // queues are idle.
        self.scheduler.release_kept(self.vk());
        self.scheduler.release_idle(self.vk());
        // No command buffer is left to give back a spare, or to hold one; the spare command pools
        // whose command buffers the free thread frees are kept again once it did (`spares.rs`).
        self.frees.drain();
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
        // The linked pipelines, then the graphics pipeline libraries they linked: each variant's
        // pipeline holds the device, and the kept ones went above.
        if let Some(libraries) = &self.libraries {
            libraries.drain(&*self);
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
        // Likewise once the exit began: the caches' records stay unwritten, and the memory's
        // blocks stay with their deallocators unrun, as a release after the exit leaves them
        // (`exit_guard`).
        if exit_guard::closed() {
            return;
        }
        // Every other pipeline was destroyed with its state (each holds this device), so the
        // `VkPipelineCache` the cache keeps for each record goes, and the cache thread stops.
        // SAFETY: the device is live and no making is left to be handed a cache.
        unsafe { self.shader_cache.close(self.vk()) };
        // What waited with blocks of its memory (a host import's deallocators) goes once they are
        // freed, with no lock held.
        let after = lock(&self.memory).destroy(self.vk());
        drop(after);
        // Every free posted to the free thread runs, and the thread ends (`free_thread.rs`).
        self.frees.stop();
        // And the device itself.
        if exit_guard::closed() {
            return;
        }
        // The device's record of pipeline makings goes, before `vkDestroyDevice`: in builds with
        // debug assertions, no making is in progress (`compiles.rs`).
        compiles::unregister(self.vk().handle());
        if self.owned {
            // SAFETY: the runtime created the device, every object made from it is destroyed,
            // and the loader's `vkDestroyDevice` reads the device's loader word first.
            unsafe {
                self.loader_word.restored(|| {
                    release::object(self.vk(), release::WholeDevice);
                })
            }
        }
        self.lineage.destroyed();
    }
}

/// One of the device's `VkQueue`s.
pub(crate) struct QueueSlot {
    /// The queue, under its lock: submissions are externally synchronized. With it, its place in
    /// its driver's period of stalls (`submit_stalls.rs`), which each use under the lock moves.
    raw: Mutex<(vk::Queue, submit_stalls::Period)>,
    /// Its place's `Period::may_be_due`, as of the last use: read without the lock.
    stall_due: AtomicBool,
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
    /// Its part in the command queues' sharing of `VkQueue`s (`queue_share.rs`).
    pub(crate) share: crate::queue_share::SlotShare,
    /// Its command buffers' waits behind other command queues' batches (`queue_wait.rs`).
    pub(crate) waits: crate::queue_wait::QueueWaits,
    /// The device's queue uses, which each hold of the queue's lock counts (`queue_quiet.rs`).
    quiet: Arc<queue_quiet::QueueQuiet>,
}

impl QueueSlot {
    /// `f` with the queue, under its lock, its loader word put back first: a use whose
    /// signalling calls are not counted, so its place in its driver's period is forgotten
    /// (`submit_stalls.rs`).
    pub(super) fn with<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        self.hold().with(f)
    }

    /// `f` with the queue and its place in its driver's period, under its lock, its loader word
    /// put back first: `f` counts each signalling call it makes (`Period::called`). For a use of
    /// `caller`'s, also the wait for the lock where it waited behind a stall taken idle
    /// (`Period::entered`).
    pub(crate) fn with_counted<R>(
        &self,
        caller: submit_stalls::Caller,
        f: impl FnOnce(vk::Queue, &mut submit_stalls::Period) -> R,
    ) -> (R, Option<u64>) {
        self.hold().with_counted(caller, f)
    }

    /// The queue's lock, held until the hold drops, the wait for it timed: each use under it is
    /// made through the hold ([`SlotHold::with`], [`SlotHold::with_counted`]). A caller that
    /// holds several queues at once (`QueueUses::use_queues_kept`) takes their locks in index
    /// order.
    pub(super) fn hold(&self) -> SlotHold<'_> {
        let asked = gpu_time::host_nanos();
        let raw = lock(&self.raw);
        let waited = gpu_time::host_nanos().saturating_sub(asked);
        self.quiet.began();
        SlotHold {
            slot: self,
            raw,
            waited,
        }
    }

    /// `f` with the queue's place in its driver's period alone, under the queue's lock.
    pub(crate) fn with_period<R>(&self, f: impl FnOnce(&mut submit_stalls::Period) -> R) -> R {
        let mut held = lock(&self.raw);
        let answer = f(&mut held.1);
        self.stall_due.store(held.1.may_be_due(), Ordering::Relaxed);
        answer
    }

    /// Whether its driver's next stall may be due within `submit_stalls::MOST_EMPTY` calls, as of
    /// its last use (`Period::may_be_due`), read without its lock.
    pub(crate) fn stall_may_be_due(&self) -> bool {
        self.stall_due.load(Ordering::Relaxed)
    }

    /// The host saw the timeline reach `value`: published, then the waits behind other command
    /// queues' batches it ends are timed (`queue_wait.rs`).
    pub(super) fn completed(&self, value: u64, counts: &crate::work_counts::Counters) {
        if self.completed.fetch_max(value, Ordering::AcqRel) < value {
            self.share.completed();
            self.waits.completed(value, counts);
        }
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
            self.share.submitted(self.completed_value() >= before);
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
    pub(super) fn submitted_value(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }
}

/// A hold of a [`QueueSlot`]'s lock ([`QueueSlot::hold`]), let go as it drops: one use of the
/// device's queues (`queue_quiet.rs`) while it lasts.
pub(super) struct SlotHold<'a> {
    slot: &'a QueueSlot,
    raw: MutexGuard<'a, (vk::Queue, submit_stalls::Period)>,
    /// How long the hold waited for the lock, in nanoseconds: the first counted use's wait
    /// (`Period::entered`), 0 for the later ones.
    waited: u64,
}

impl Drop for SlotHold<'_> {
    fn drop(&mut self) {
        self.slot.quiet.ended();
    }
}

impl SlotHold<'_> {
    /// [`QueueSlot::with`], under this hold.
    pub(super) fn with<R>(&mut self, f: impl FnOnce(vk::Queue) -> R) -> R {
        self.raw.1.forget();
        self.slot.stall_due.store(false, Ordering::Relaxed);
        // SAFETY: the queue lives while its slot does.
        unsafe { self.slot.word.restore() };
        f(self.raw.0)
    }

    /// [`QueueSlot::with_counted`], under this hold.
    pub(super) fn with_counted<R>(
        &mut self,
        caller: submit_stalls::Caller,
        f: impl FnOnce(vk::Queue, &mut submit_stalls::Period) -> R,
    ) -> (R, Option<u64>) {
        let behind = match caller {
            submit_stalls::Caller::Work => {
                let waited = self.waited;
                self.waited = 0;
                self.raw.1.entered(waited)
            }
            submit_stalls::Caller::Empty => None,
        };
        // SAFETY: the queue lives while its slot does.
        unsafe { self.slot.word.restore() };
        let (queue, period) = &mut *self.raw;
        let used = f(*queue, period);
        self.slot
            .stall_due
            .store(period.may_be_due(), Ordering::Relaxed);
        (used, behind)
    }
}

#[cfg(debug_assertions)]
thread_local! {
    /// The locks this thread took of what its draws share with draws on other threads (builds
    /// with debug assertions): a pipeline state's settles, the device's function tables, a
    /// variant's quiet formats.
    static SHARED_DRAW_LOCKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// [`lock`] of something draws on other threads share (`SHARED_DRAW_LOCKS`, counted in builds
/// with debug assertions): what a test of a draw that takes none counts.
pub(super) fn shared_draw_lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    #[cfg(debug_assertions)]
    SHARED_DRAW_LOCKS.with(|locks| locks.set(locks.get() + 1));
    lock(mutex)
}

/// How many [`shared_draw_lock`]s this thread took so far.
#[cfg(all(test, debug_assertions, any(m2v_gpu, m2v_cpu_icd)))]
pub(super) fn shared_draw_locks() -> u64 {
    SHARED_DRAW_LOCKS.with(std::cell::Cell::get)
}

/// `mutex`'s guard; a lock poisoned by a panicking holder is taken over (the runtime's state is
/// updated before any call that can panic).
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl Device {
    /// Fail the next `count` imports of host pages by their pointer as the driver's out of device
    /// memory: the tests' stand-in for an import the device and the gate admitted that then fails
    /// (`copied.rs`).
    pub(crate) fn fail_next_host_imports(&self, count: u64) {
        (self.shared().failing_host_imports).store(count, Ordering::Relaxed);
    }

    /// Fail the next `count` `vkQueueSubmit`s of its command buffers' work as the driver's out of
    /// device memory, none of them made: the tests' stand-in for a submission the driver refuses
    /// (`DeviceShared::submit_logged`).
    pub(crate) fn fail_next_submissions(&self, count: u64) {
        (self.shared().failing_submissions).store(count, Ordering::Relaxed);
    }
}

impl Device {
    /// `registryID`: an identifier of the physical device, from its `deviceUUID`.
    pub fn registry_id(&self) -> u64 {
        properties::registry_id(&self.properties().device_uuid)
    }

    /// `name`.
    pub fn name(&self) -> &str {
        &self.properties().name
    }

    /// The device's identity and Metal's public device queries, with the documented facts a
    /// consumer computes Apple's private ones from: the same through a recovery of its GPU
    /// ([`Device::recover_for_test`]), which makes the new `VkDevice` of the same physical device.
    pub fn properties(&self) -> &DeviceProperties {
        &self.invariant().properties
    }

    /// What every generation of the device answers alike (`line.rs`): a generation handle's own
    /// copy, a root handle's line's.
    #[inline(always)]
    fn invariant(&self) -> &line::Invariant {
        match &self.0 {
            Pin::Generation(shared) => &shared.invariant,
            Pin::Root(root) => &root.line.invariant,
        }
    }

    /// The device's line (`line.rs`).
    #[inline(always)]
    pub(crate) fn line(&self) -> &Arc<line::Line> {
        match &self.0 {
            Pin::Generation(shared) => &shared.line,
            Pin::Root(root) => &root.line,
        }
    }

    /// A root handle of the device (`line.rs`): one whose calls go to the generation that serves.
    pub(crate) fn root(&self) -> Device {
        holds::root(Arc::clone(self.line()))
    }

    /// The generation of a generation handle (`line.rs`); of a root handle, the one its first
    /// such reach pinned ([`Device::shared`]).
    pub(super) fn shared_arc(&self) -> &Arc<DeviceShared> {
        match &self.0 {
            Pin::Generation(shared) => shared,
            Pin::Root(root) => root.pinned.get_or_init(|| self.pin_root(&root.line)),
        }
    }

    /// The generation a root handle pins at its first reach of a generation's state: the one
    /// that serves then. The runtime's own code holds generation handles alone, and every public
    /// call of a root handle goes to the generation that serves ([`Device::current`]); only a
    /// test reaches a root's state, and keeps that generation while the root's cell lives.
    #[cold]
    #[inline(never)]
    fn pin_root(&self, line: &Arc<line::Line>) -> Arc<DeviceShared> {
        Arc::clone(line.current(self).shared_arc())
    }

    /// A handle on the device that does not keep it: a generation handle's, of its generation.
    pub(crate) fn downgrade(&self) -> WeakDevice {
        let shared = self.shared_arc();
        WeakDevice(Arc::downgrade(shared), Arc::clone(&shared.holds))
    }

    /// The compile-and-plan device this device compiles and plans for. No Metal counterpart.
    pub fn compile_device(&self) -> &metal2vulkan::device::Device {
        &self.invariant().compile
    }

    /// What the Vulkan device enabled, in the compile layer's vocabulary. No Metal counterpart.
    pub fn enabled(&self) -> &EnableSet {
        &self.invariant().enabled
    }

    /// Whether the Vulkan device enabled the device extension `name`: one the options asked for
    /// (`DeviceOptions::device_extensions`, and each of `optional_device_extensions` the device
    /// could enable), or one the runtime enabled itself (of the vocabulary's names, the one the
    /// device listed). An adopted device answers for its `RawDevice::device_extensions`. No Metal
    /// counterpart.
    pub fn enables_extension(&self, name: &std::ffi::CStr) -> bool {
        (self.invariant().extensions.iter()).any(|enabled| enabled.as_c_str() == name)
    }

    /// The generation a generation handle holds (`line.rs`). The runtime reaches a device's
    /// generation through generation handles: every public call of a root handle goes to the
    /// generation that serves first ([`Device::current`]). A root handle answers with the one
    /// it pinned ([`shared_arc`](Self::shared_arc)).
    #[inline(always)]
    pub(super) fn shared(&self) -> &DeviceShared {
        match &self.0 {
            Pin::Generation(shared) => shared,
            Pin::Root(..) => self.root_shared(),
        }
    }

    /// [`shared`](Self::shared) of a root handle: out of line, as only a test reaches it.
    #[cold]
    #[inline(never)]
    fn root_shared(&self) -> &DeviceShared {
        self.shared_arc()
    }

    /// Keep room in the memory pools the runtime's encodings and commits place their own
    /// resources in, and make their blocks ahead while they grow (`memory::keep_room`,
    /// `memory/ready.rs`): called by the creation calls that may wait (a queue's, a pipeline
    /// state's, a buffer's, a texture's, a heap's), so that no encoding or commit makes a memory
    /// block, whose `vkAllocateMemory` waited 2.37 s for other processes' GPU work on NVIDIA's
    /// driver. Nothing in a host step, a serial point of the command buffers' stream that makes
    /// only what it needs.
    pub(crate) fn keep_room(&self) {
        if crate::exit_guard::in_host_step() {
            return;
        }
        let shared = self.shared();
        let copies = || self.copies_color_on_the_host();
        let memory = &shared.memory;
        // The kind of block the runtime's own buffers take now (`buffer.rs`'s
        // `placed_new_buffer`), asked as their making asks it but counting nothing.
        let exportable = lock(memory).fd_buffer_types.is_ok();
        let own = lock(memory).own_buffers(exportable && self.fd_blocks_taken());
        // And whether its host-visible ones take blocks of its own pages.
        let pay = self.fresh_pages_pay_now() && self.page_blocks_taken();
        let pages = lock(memory).page_blocks(pay);
        crate::memory::keep_room(
            memory,
            shared.vk(),
            &shared.counts,
            (&shared.functions, &shared.queue_quiet),
            (copies, own, pages),
        );
    }

    /// Whether `other` is this device (the same generation, not merely the same physical
    /// device); of two handles one of which is a root, whether they are of the same line.
    #[inline(always)]
    pub(super) fn same(&self, other: &Device) -> bool {
        // The common answer, a bind's, at one compare: the same generation (or root cell).
        self.0.address() == other.0.address() || self.same_apart(other)
    }

    /// [`same`](Self::same) of two handles that hold different cells: of one line where one is a
    /// root; two generations are two devices.
    #[cold]
    #[inline(never)]
    fn same_apart(&self, other: &Device) -> bool {
        match (&self.0, &other.0) {
            (Pin::Generation(_), Pin::Generation(_)) => false,
            _ => self.same_line(other),
        }
    }

    /// Whether `other` is of this device's line: this device, or one generation of it made before
    /// or after a recovery (`line.rs`).
    pub(super) fn same_line(&self, other: &Device) -> bool {
        Arc::ptr_eq(self.line(), other.line())
    }

    /// Whether the device was lost: not to a public call that took its arguments on it before
    /// its loss, which the loss follows (`Device::at_serving_taking`, `line.rs`).
    pub(super) fn is_lost(&self) -> bool {
        let shared = self.shared();
        shared.lost() && !line::runs_on(shared)
    }

    /// Record that the device was lost: its line recovers at the next call that needs the device,
    /// where it recovers (`line.rs`).
    pub(super) fn mark_lost(&self) {
        let shared = self.shared();
        shared.lost.store(true, Ordering::Release);
        shared.line.lose(shared);
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
    /// device, and waits for the work on a device that is not: work that completes with no more
    /// work of the host's, since none of the runtime's waits on a queue whose lock is free for a
    /// signal the host has yet to make (the queue-switch check holds its round's queues until it
    /// signalled the round's gate, `queue_switch_check.rs`).
    pub(super) fn drain_queues(&self) {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.shared();
        shared.queues_idle();
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
        (
            self.shared().contexts.clone(),
            self.shared().contexts.fresh(),
        )
    }

    /// A new logical queue's identity, and the index of the `VkQueue` it submits to: round
    /// robin over the device's queues that are not reserved for presenting.
    ///
    /// Past the round robin's queues it gives one an earlier logical queue took, whose
    /// submissions this queue's then wait behind where Metal's queues would not
    /// (`device/create.rs`'s `ROUND_ROBIN_CEILING`): counted,
    /// `WorkCounts::queues_sharing_a_vk_queue`. Where the device's queue limits changed its round
    /// robin (`DeviceOptions::queue_limits`) and this is a queue the program made, `noted` naming
    /// it (`None` for the runtime's own and for a presenting queue's, which
    /// `Device::next_presenting_queue` notes), the device says so where the program's queues
    /// made so far reach past the limits' `VkQueue`s and the plan gives this one its own: once
    /// per device ([`Told::QueueLimitsReached`]).
    pub(super) fn next_queue(&self, noted: Option<&str>) -> (u64, usize) {
        let id = self.shared().contexts.fresh();
        let made = self
            .shared()
            .round_robin_made
            .fetch_add(1, Ordering::Relaxed);
        if made >= self.shared().round_robin as u64 {
            self.shared().counts.queue_shared_a_vk_queue();
        }
        let vk_queue = (made % self.shared().round_robin as u64) as usize;
        if let (Some(queue), Some((planned, _))) = (noted, self.shared().planned) {
            let program = (self.shared().program_queues_made).fetch_add(1, Ordering::Relaxed);
            if create::shares_past_limit(program, self.shared().round_robin, planned) {
                self.inform_once(
                    Told::QueueLimitsReached,
                    create::QUEUE_LIMITS_REACHED,
                    || {
                        let what = format!(
                            "{queue}, the program's command queue {} on a round robin of {} \
                         VkQueue(s), took VkQueue {vk_queue}, which another took, where the \
                         plan's {planned} give it one of its own",
                            program + 1,
                            self.shared().round_robin
                        );
                        create::queue_limits_reached(self.name(), &what)
                    },
                );
            }
        }
        (id, vk_queue)
    }

    /// A new logical queue's identity alone, for one whose `VkQueue` is chosen otherwise
    /// (`Device::place_unshared`).
    pub(super) fn new_queue_id(&self) -> u64 {
        self.shared().contexts.fresh()
    }

    /// A new presenting queue's identity, and the index of the `VkQueue` it submits to: the
    /// queues reserved for presenting, in turn. `None` where the device reserved none.
    pub(super) fn next_presenting_queue(&self) -> Option<(u64, usize)> {
        let reserved = self.shared().queues.len() - self.shared().round_robin;
        if let Some((_, planned)) = self.shared().planned {
            let asked = (self.shared().presenting_asked).fetch_add(1, Ordering::Relaxed);
            if create::shares_past_limit(asked, reserved, planned) {
                self.inform_once(
                    Told::QueueLimitsReached,
                    create::QUEUE_LIMITS_REACHED,
                    || {
                        let what = format!(
                            "presenting queue {} shares a VkQueue, the device's queue limits \
                         reserving {reserved} for presenting queues, where the plan reserves \
                         {planned}",
                            asked + 1
                        );
                        create::queue_limits_reached(self.name(), &what)
                    },
                );
            }
        }
        if reserved == 0 {
            return None;
        }
        let id = self.shared().contexts.fresh();
        let made = self
            .shared()
            .presenting_made
            .fetch_add(1, Ordering::Relaxed);
        Some((
            id,
            self.shared().round_robin + (made % reserved as u64) as usize,
        ))
    }

    /// `vkAllocateCommandBuffers` of `info` (`DeviceShared::allocate_command_buffers`).
    #[cfg_attr(test, track_caller)]
    pub(super) fn allocate_command_buffers(
        &self,
        info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, Error> {
        self.shared().allocate_command_buffers(info)
    }
}

/// The parts [`DeviceShared`] is built from, by `create.rs` and `adopt.rs` alike.
pub(super) struct Parts {
    pub(super) instance: Arc<InstanceShared>,
    pub(super) physical: vk::PhysicalDevice,
    pub(super) raw: ash::Device,
    /// The device's loader word as it stood when the device reached the runtime.
    pub(super) loader_word: DeviceWord,
    /// The primary queue family, the first queue's.
    pub(super) queue_family: u32,
    /// The queues the runtime submits to, each its family and index there, the primary family's
    /// first queue first: at least one.
    pub(super) queues: Vec<(u32, u32)>,
    /// How many of `queues`, the last, are reserved for presenting queues (fewer than all).
    pub(super) present_queues: usize,
    /// Where the device's queue limits changed its queues (`DeviceOptions::queue_limits`), the
    /// queue plan's round robin and presenting `VkQueue`s; `None` for an adopted device.
    pub(super) planned: Option<(usize, usize)>,
    /// The driver refused the device with its queue plan's `VkQueue`s, and it took its least
    /// plan's (`create.rs`'s `made_within_resources`, `Degradation::LeastQueuePlan`).
    pub(super) least_plan: bool,
    /// The queue reserved for staging copies, its family and index there, outside `queues`
    /// (`staging.rs`): `None` for an adopted device, or where the families have no more.
    pub(super) staging: Option<(u32, u32)>,
    /// The queues reserved for the caller (`DeviceOptions::reserved_queues`), each its family and
    /// index there, of `queues`' families, outside `queues` and `staging`: none for an adopted
    /// device.
    pub(super) reserved_queues: Vec<(u32, u32)>,
    /// The caller's extra queues (`DeviceOptions::extra_queues`), each the flags asked for, its
    /// family and its index there, outside `queues` and `staging`: none for an adopted device.
    pub(super) extra_queues: Vec<(vk::QueueFlags, u32, u32)>,
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
    /// The class of Mac it presents, as the consumer declared it (`None`: its GPU's,
    /// `capabilities.rs`).
    pub(super) mac_class: Option<crate::MacClass>,
    /// The degradation kinds it announces at every occurrence (`degradation.rs`).
    pub(super) announce_each: BTreeSet<Degradation>,
    /// The degradation kinds that refuse a render encoder's call met while lowering it after it
    /// returned (`render/deferral.rs`).
    pub(super) lowering_refuses: BTreeSet<Degradation>,
    /// Its command buffers keep execution records (`DeviceOptions::record_executions`).
    pub(super) record_executions: bool,
    /// Where its pipeline records go (`DeviceOptions::pipeline_records`, `warm/record.rs`).
    pub(super) pipeline_records: Option<crate::warm::RecordHook>,
    /// Its check of the first move over host pages fails whatever the moves did
    /// (`DeviceOptions::fail_in_place_check`).
    pub(super) fail_in_place_check: bool,
    /// Its submission check of merged submissions holds without its answer
    /// (`DeviceOptions::assume_submissions_merge`).
    pub(super) assume_submissions_merge: bool,
    /// Its pending audit is on (`DeviceOptions::pending_audit`; `pending_audit.rs`), as it is
    /// where `METAL2VULKAN_PENDING_AUDIT` is set.
    pub(super) pending_audit: bool,
    /// Its views may reinterpret an image's components ([`DeviceShared`]'s field), outside
    /// `enabled`'s vocabulary.
    pub(super) view_reinterpretation: bool,
    /// It enables `VK_EXT_device_generated_commands` (`create.rs`), outside `enabled`'s
    /// vocabulary.
    pub(super) generated_commands: bool,
    /// It enables `hostImageCopy` where a heap's colour texture takes host-transfer usage
    /// (`create.rs`; `adopt.rs`), outside `enabled`'s vocabulary.
    pub(super) host_image_copy: bool,
    /// It enables `VK_EXT_zero_initialize_device_memory` and its `zeroInitializeDeviceMemory`
    /// (`create.rs`; never on a device the consumer made, `adopt.rs`), outside `enabled`'s
    /// vocabulary.
    pub(super) zero_initialize_device_memory: bool,
    /// It enables `VK_NV_linear_color_attachment` and its `linearColorAttachment` (`create.rs`;
    /// `adopt.rs`, where the consumer lists it), outside `enabled`'s vocabulary: a linear image
    /// renders in place where the instance's color attachments are all linear
    /// (`linear_attachment.rs`).
    pub(super) linear_color_attachment: bool,
    /// It enables `VK_EXT_graphics_pipeline_library` and links its libraries fast (`create.rs`;
    /// `adopt.rs`, where the consumer lists it), outside `enabled`'s vocabulary: its render
    /// pipelines are linked from libraries (`pipeline/libraries.rs`).
    pub(super) graphics_pipeline_library: bool,
    /// It enables `samplerYcbcrConversion` (`create.rs`), outside `enabled`'s vocabulary: a
    /// two-plane YCbCr texture's planes convert through the device's own conversion where it
    /// converts their format (`ycbcr_device.rs`). An adopted device's consumer does not say, so
    /// it is false there.
    pub(super) ycbcr_conversion: bool,
    /// It takes the two-plane 4:4:4 formats with that conversion: Vulkan 1.3, or
    /// `VK_EXT_ycbcr_2plane_444_formats` enabled with its feature (`create.rs`).
    pub(super) ycbcr_444: bool,
    /// It enables `VK_EXT_pipeline_creation_feedback` (`create.rs`), outside `enabled`'s
    /// vocabulary, wherever listed: each making of a program's pipeline reads whether its cache
    /// held the pipeline (`shader_cache/mod.rs`).
    pub(super) creation_feedback: bool,
    /// It enables the dma-buf route's extensions (`create.rs`'s `dma_buf_names`), outside
    /// `enabled`'s vocabulary (`dma_buf.rs`).
    pub(super) dma_buf: bool,
    /// It enables the route's image extensions (`create.rs`'s `dma_buf_image_names`), likewise.
    pub(super) dma_buf_images: bool,
    /// It enables `VK_KHR_external_memory_fd` (`create.rs`'s `memory_fd_name`; `adopt.rs`, where
    /// the consumer lists it), outside `enabled`'s vocabulary (`exported_memory.rs`).
    pub(super) memory_fd: bool,
    /// It enables `VK_EXT_queue_family_foreign` beside the dma-buf route's extensions
    /// (`create.rs`'s `foreign_family_name`; `adopt.rs`, where the consumer lists them), outside
    /// `enabled`'s vocabulary (`exported_memory.rs`).
    pub(super) foreign_family: bool,
    /// It enables `VK_KHR_external_semaphore_fd` (`create.rs`'s `semaphore_fd_name`; `adopt.rs`,
    /// where the consumer lists it), outside `enabled`'s vocabulary (`exported_semaphore.rs`).
    pub(super) semaphore_fd: bool,
    /// The sparse members of the core features it enables, outside `enabled`'s vocabulary
    /// (`sparse/device.rs`).
    pub(super) sparse_features: vk::PhysicalDeviceFeatures,
    /// The reads its global memory barriers name (`image_barrier::MemoryReads`): those of the
    /// primary queue family's commands with the extensions it enables.
    pub(super) memory_reads: MemoryReads,
    pub(super) owned: bool,
    /// The shader cache, its directory opened before any Vulkan object was made.
    pub(super) cache: CacheSetup,
    /// The line a recovery makes this generation of (`line.rs`); `None` for a device made now.
    pub(super) line: Option<Arc<line::Line>>,
    /// How a device made now makes its next `VkDevice` once one is lost (`create.rs`); `None`
    /// for one the consumer made, and for a recovery's, whose line has it.
    pub(super) recipe: Option<line::Recipe>,
}

impl Device {
    /// The device of `parts`: create is built on adopt, so both end here, and so does a
    /// recovery's new generation (`line.rs`). A generation handle of it, which the program's
    /// calls that made a device turn into a root handle ([`Device::root`]).
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
            planned,
            least_plan,
            staging: staging_place,
            reserved_queues: reserved_places,
            extra_queues: extra_places,
            compile,
            description,
            enabled,
            extensions,
            calibration,
            gpu_times_from_host,
            mac_class,
            announce_each,
            lowering_refuses,
            record_executions,
            pipeline_records,
            fail_in_place_check,
            assume_submissions_merge,
            pending_audit,
            view_reinterpretation,
            generated_commands,
            host_image_copy,
            zero_initialize_device_memory,
            linear_color_attachment,
            graphics_pipeline_library,
            ycbcr_conversion,
            ycbcr_444,
            creation_feedback,
            dma_buf,
            dma_buf_images,
            #[cfg(unix)]
            memory_fd,
            #[cfg(not(unix))]
                memory_fd: _,
            #[cfg(unix)]
            foreign_family,
            #[cfg(not(unix))]
                foreign_family: _,
            #[cfg(unix)]
            semaphore_fd,
            #[cfg(not(unix))]
                semaphore_fd: _,
            sparse_features,
            memory_reads,
            owned,
            cache,
            line: recovering,
            recipe,
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
                    loader_word.restored(|| {
                        release::object(&raw, release::WholeDevice);
                    })
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
        let queue_quiet = Arc::new(queue_quiet::QueueQuiet::default());
        // Whether the driver stalls in one signalling call of every 1024 to a queue: its calls'
        // CPU time tells its stall (`submit_stalls.rs`).
        let stalls = (description.driver_facts())
            .contains(&metal2vulkan::device::DriverFact::SignallingSubmitsStallEvery1024);
        let queues: Vec<QueueSlot> = queue_places
            .iter()
            .zip(&timelines)
            .map(|(&(family, index), &timeline)| {
                // SAFETY: the queue exists (the caller's contract), and the loader serves the
                // call after the word is put back.
                let queue = unsafe { loader_word.restored(|| raw.get_device_queue(family, index)) };
                QueueSlot {
                    raw: Mutex::new((queue, submit_stalls::Period::new(stalls))),
                    stall_due: AtomicBool::new(false),
                    family,
                    flags: flags_of(family),
                    // SAFETY: the queue was just returned, its word the loader's.
                    word: unsafe { LoaderWord::of_handle(queue.as_raw()) },
                    flights: OnceLock::new(),
                    timeline,
                    completed: AtomicU64::new(0),
                    submitted: AtomicU64::new(0),
                    submitted_before: AtomicU64::new(0),
                    share: crate::queue_share::SlotShare::default(),
                    waits: crate::queue_wait::QueueWaits::default(),
                    quiet: Arc::clone(&queue_quiet),
                }
            })
            .collect();
        let mut queue_families: Vec<u32> = Vec::with_capacity(queues.len());
        let reserved_families = reserved_places.iter().map(|&(family, _)| family);
        for family in queues
            .iter()
            .map(|slot| slot.family)
            .chain(reserved_families)
        {
            if !queue_families.contains(&family) {
                queue_families.push(family);
            }
        }
        let buffer_families: Vec<u32> = (queue_families.iter().copied())
            .chain(staging_place.map(|(family, _)| family))
            .collect();
        let staging = staging_place.map(|(family, index)| {
            // SAFETY: the queue exists (the caller's contract), and the loader serves the call
            // after the word is put back; it was just returned, its word the loader's.
            unsafe {
                loader_word.restored(|| {
                    let queue = raw.get_device_queue(family, index);
                    staging::StagingQueue::new(family, flags_of(family), queue, &queue_families)
                })
            }
        });
        let reserved_queues = (reserved_places.iter())
            .map(|&(family, index)| handles::ReservedQueue {
                family,
                // SAFETY: the queue exists (the caller's contract), and the loader serves the
                // call after the word is put back.
                queue: unsafe { loader_word.restored(|| raw.get_device_queue(family, index)) },
            })
            .collect();
        let extra_queues = (extra_places.iter())
            .map(|&(flags, family, index)| handles::ExtraQueue {
                flags,
                family,
                // SAFETY: the queue exists (the caller's contract), and the loader serves the
                // call after the word is put back.
                queue: unsafe { loader_word.restored(|| raw.get_device_queue(family, index)) },
            })
            .collect();
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
                loader_word.restored(|| {
                    Commands::load(raw.handle(), |name| {
                        instance
                            .raw
                            .get_device_proc_addr(raw.handle(), name.as_ptr())
                    })
                })
            }
            .map(|commands| commands.routed(recording.as_ref()))
        } else {
            None
        };
        // At least one write of set 0 pushed after each pipeline bind where the driver's pushed
        // descriptors go stale across one
        // (`RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`).
        let pushes_after_pipeline_binds = crate::driver_defects::of(&description).contains(
            &metal2vulkan::error::RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds,
        );
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let push_descriptor = unsafe {
            loader_word.restored(|| {
                crate::binding::push_descriptor_command(
                    &instance.raw,
                    &raw,
                    description.api_version,
                    &enabled,
                )
            })
        }
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
                loader_word.restored(|| {
                    instance.raw.get_device_proc_addr(
                        raw.handle(),
                        c"vkSetDebugUtilsObjectNameEXT".as_ptr(),
                    )
                })
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
            loader_word.restored(|| {
                gpu_time::calibrated(&instance.entry, &instance.raw, physical, &raw, calibration)
            })
        }
        .filter(|_| timestamp_bits > 0);
        // The render encoder's commands of extensions and of Vulkan 1.3, which priming makes too.
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let extended_dynamic_state = unsafe {
            loader_word.restored(|| {
                ExtendedDynamicState::load(
                    &instance.raw,
                    &raw,
                    description.api_version,
                    &enabled,
                    recording.as_ref(),
                )
            })
        };
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let sized_index_bind =
            unsafe { loader_word.restored(|| sized_index_bind(&instance.raw, &raw, &enabled)) }
                .map(|bind| {
                    // SAFETY: the device's own command of that name.
                    unsafe {
                        crate::recorder::through(
                            recording.as_ref(),
                            Command::CmdBindIndexBuffer2,
                            bind,
                        )
                    }
                });
        // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put back.
        let color_write_masks = unsafe {
            loader_word.restored(|| ColorWriteMasks::load(&instance.raw, &raw, &enabled))
        }
        .map(|ColorWriteMasks(masks)| {
            // SAFETY: the device's own command of that name.
            ColorWriteMasks(unsafe {
                crate::recorder::through(recording.as_ref(), Command::CmdSetColorWriteMask, masks)
            })
        });
        // The YCbCr formats the device converts itself, where it enables the conversion.
        let ycbcr_formats = match ycbcr_conversion {
            // SAFETY: `physical` belongs to `instance`, created at Vulkan 1.2 or later, and the
            // device enables `samplerYcbcrConversion`.
            true => unsafe {
                crate::ycbcr_device::YCbCrFormats::of(&instance.raw, physical, ycbcr_444)
            },
            false => crate::ycbcr_device::YCbCrFormats::default(),
        };
        let drawn = prime::PrimedDraws {
            dynamic: extended_dynamic_state,
            color_write_masks,
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
                    None,
                ),
                &loader_word,
                queue_family,
                &queues,
                Sharing::with_buffers(&queue_families, &buffer_families),
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
                    sized_index_bind: false,
                    ycbcr_conversion: ycbcr_formats.primer().is_some(),
                }),
                generated_commands.as_ref(),
                push_descriptor,
                calibrated,
                drawn,
                ycbcr_formats.primer(),
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
        // The limit the device's live samplers are held to, the one
        // `DeviceProperties::max_argument_buffer_sampler_count` is derived from.
        let sampler_limit = identity.max_sampler_allocation_count;
        let properties = DeviceProperties::new(identity, description, &compile, mac_class);
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
        memory.share_buffers_among(buffer_families.clone());
        memory.log_to(MemoryLog::new(instance.sink()));
        if zero_initialize_device_memory {
            memory.zero_image_blocks();
        }
        // SAFETY: as above.
        let physical_properties = unsafe { instance.raw.get_physical_device_properties(physical) };
        let limits = physical_properties.limits;
        // Counted from here, past the priming, which freed all it allocated.
        let allocations =
            crate::allocations::register(raw.handle(), limits.max_memory_allocation_count);
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
        // The program's host-visible buffers lie in blocks of pages the runtime maps and imports
        // by their pointer, in the memory types the device imports such pages as, once its
        // submission check found that a held import costs its submissions nothing (`memory.rs`'s
        // "Buffers in the runtime's own pages").
        let (page_types, page_unit) = match host_import.as_ref().ok().and_then(Importer::pointers) {
            Some(pointers) => {
                let unit = pointers.unit();
                // SAFETY: `raw` is the import's device.
                let types = unsafe { crate::fresh_pages::own_page_types(pointers, &raw, unit) };
                (types, unit)
            }
            None => (Err("the device imports no host pointer"), 1),
        };
        memory.page_buffer_blocks(&raw, page_types, page_unit)?;
        // How the device exports a buffer's memory as an opaque file descriptor, and as a dma-buf,
        // where it enables the extensions each needs and exports a buffer's memory so
        // (`exported_memory.rs`).
        #[cfg(unix)]
        let (opaque_fd, dma_buf_export) = {
            // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put
            // back, and the device enables the extension.
            let functions = memory_fd.then(|| unsafe {
                loader_word
                    .restored(|| ash::khr::external_memory_fd::Device::new(&instance.raw, &raw))
            });
            let usage = super::buffer::usage(device_address);
            let handle = crate::buffer_descriptor::MemoryExport::OpaqueFd;
            // SAFETY: `physical` belongs to `instance`, created at Vulkan 1.2 or later.
            let opaque = unsafe {
                crate::exported_memory::FdExport::new(
                    &instance.raw,
                    physical,
                    functions.clone(),
                    (usage, handle),
                )
            };
            // The device's own buffers lie in blocks exportable as opaque file descriptors,
            // never exported, where it exports such a descriptor of a pooled buffer's memory for
            // the usage of those buffers and of a buffer of addresses, and its policy takes
            // textures over memory of an export in place by its check; in the memory types its
            // linear image made for one takes as its own linear image does (`memory.rs`'s
            // "Buffers in exportable blocks").
            let in_place = compile.policy().host_page_textures
                == metal2vulkan::device::HostPageTexturePolicy::InPlace;
            let usages = [
                Some(super::buffer::usage(false)),
                device_address.then_some(usage),
            ];
            // Whether the device exports each usage's pooled memory as an opaque file descriptor.
            let pooled = |functions: &ash::khr::external_memory_fd::Device| {
                (usages.into_iter().flatten()).all(|usage| {
                    // SAFETY: as above.
                    let export = unsafe {
                        crate::exported_memory::FdExport::new(
                            &instance.raw,
                            physical,
                            Some(functions.clone()),
                            (usage, handle),
                        )
                    };
                    export.is_some_and(|export| export.pooled())
                })
            };
            let exports = match &functions {
                None => Err("the device enables no VK_KHR_external_memory_fd"),
                Some(_) if !in_place => {
                    Err("the device's policy shadows every texture over memory of an export")
                }
                Some(functions) if !pooled(functions) => Err(
                    "the device exports an opaque file descriptor of no pooled buffer's memory of \
                     the runtime's usages",
                ),
                Some(_) => Ok(()),
            };
            let types = crate::memory::fd_buffer_types(
                &memory_properties,
                properties.has_unified_memory,
                exports,
            );
            let blocks = match types {
                // SAFETY: as above; the device enables `VK_KHR_external_memory_fd` where
                // `exports` passed.
                Ok(types) => match unsafe {
                    crate::memory::fd_image_types((&instance.raw, physical), &raw, types)
                }? {
                    (0, dropped) => (
                        Err(dropped.map_or("no memory type takes such blocks", |(_, why)| why)),
                        None,
                    ),
                    (kept, dropped) => (Ok(kept), dropped),
                },
                Err(why) => (Err(why), None),
            };
            memory.fd_buffer_blocks(blocks);
            // As a dma-buf where it enables the route's extensions, which name
            // `VK_KHR_external_memory_fd`, and the foreign family its acquire names.
            let dma_buf_functions = functions.filter(|_| dma_buf && foreign_family);
            let handle = crate::buffer_descriptor::MemoryExport::DmaBuf;
            // SAFETY: as above.
            let dma_buf_export = unsafe {
                crate::exported_memory::FdExport::new(
                    &instance.raw,
                    physical,
                    dma_buf_functions,
                    (usage, handle),
                )
            };
            (opaque, dma_buf_export)
        };
        // How the device exports a timeline semaphore as an opaque file descriptor, where it
        // enables `VK_KHR_external_semaphore_fd` and exports one so (`exported_semaphore.rs`).
        #[cfg(unix)]
        let opaque_fd_semaphores = {
            // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put
            // back, and the device enables the extension.
            let functions = semaphore_fd.then(|| unsafe {
                loader_word
                    .restored(|| ash::khr::external_semaphore_fd::Device::new(&instance.raw, &raw))
            });
            // SAFETY: `physical` belongs to `instance`, created at Vulkan 1.2 or later.
            unsafe {
                crate::exported_semaphore::OpaqueFdSemaphores::new(
                    &instance.raw,
                    physical,
                    functions,
                )
            }
        };
        // The host image copy functions, where the device enabled `hostImageCopy` and copies to
        // and from `GENERAL` on the host: by their core names from Vulkan 1.4, else by the
        // extension's, which the device then enabled.
        // SAFETY: `physical` belongs to `instance`, created at Vulkan 1.2 or later.
        let general = host_image_copy
            && unsafe { crate::texture::host_copy::reaches_general(&instance.raw, physical) };
        let host_copies = general.then(|| {
            let handle = raw.handle();
            let load = |name: &std::ffi::CStr| {
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
            };
            // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put
            // back.
            unsafe { loader_word.restored(|| ash::ext::host_image_copy::DeviceFn::load(load)) }
        });
        // The mesh draw commands, which exist where `meshShader` is enabled (it needs
        // `VK_EXT_mesh_shader`).
        let mesh_draws = if enabled
            .features
            .contains(&metal2vulkan::device::VkFeature::MeshShader)
        {
            // SAFETY: as above; the loader serves `vkGetDeviceProcAddr` once the word is put
            // back.
            let loaded = unsafe {
                loader_word.restored(|| ash::ext::mesh_shader::Device::new(&instance.raw, &raw))
            };
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
            loader_word.restored(|| {
                Functions::new(
                    &instance.raw,
                    raw.clone(),
                    memory_reads,
                    push_descriptor,
                    object_name,
                    recording.as_ref(),
                    PendingAudit::new(pending_audit, instance.sink()),
                )
            })
        };
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
        // Read before the properties move into the device's shared state (below): whether the
        // scheduler's thread waits for the GPU itself (`scheduler::waits_on_gpu`).
        let waits_on_gpu = crate::scheduler::waits_on_gpu(&properties.driver_defects);
        // And whether a spare command pool is trimmed after its frees: not where a trim fails the
        // kept barriers (`RuntimeDriverDefect::TrimmedCommandBufferFailsResubmission`,
        // `spares.rs`).
        let trims_spares = !properties
            .driver_defects
            .contains(&RuntimeDriverDefect::TrimmedCommandBufferFailsResubmission);
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
        // The blocks of its memory that go are freed on a thread of the device's, counted against
        // its heaps and its allocations until they are (`free_thread.rs`).
        let functions = Arc::new(functions);
        let frees = Arc::new(crate::free_thread::FreeThread::new(Arc::clone(&functions)));
        memory.post_frees_to(Arc::clone(&frees));
        (allocations.allocations).wait_for_frees_on(Arc::clone(&frees));
        // The device's line: a new one for a device made now, else the line this generation
        // recovers, whose address space, resource IDs, identifiers and counts it goes on with
        // (`line.rs`).
        let invariant = line::Invariant {
            properties,
            compile,
            enabled,
            extensions,
            cache_directory: shader_cache.directory_root(),
        };
        let line = match recovering {
            Some(line) => line,
            None => line::Line::new(
                holds::holds(),
                invariant.clone(),
                Announced::new(&announce_each),
                recipe,
            ),
        };
        let carried = &line.carried;
        let announced = Arc::clone(&carried.announced);
        let resource_ids = Arc::clone(&carried.resource_ids);
        let passing_holds = Arc::clone(&carried.passing_holds);
        let sampler_ids = lock(&resource_ids).sampler_id_reads();
        let depth_ranges = DepthRanges::of(invariant.compile.profile());
        let lineage = Arc::clone(&line.lineage);
        let generation = lineage.made();
        let shared = Arc::new(DeviceShared {
            addresses: Arc::clone(&carried.addresses),
            identifiers: Arc::clone(&carried.identifiers),
            counts: Arc::clone(&carried.counts),
            queue_quiet,
            holds: Arc::clone(&line.holds),
            line,
            lineage,
            generation,
            instance,
            physical,
            functions,
            queue_family,
            round_robin: queues.len() - present_queues,
            queues: queues.into(),
            queue_families,
            buffer_families,
            round_robin_made: AtomicU64::new(0),
            presenting_made: AtomicU64::new(0),
            planned,
            least_plan,
            presenting_asked: AtomicU64::new(0),
            program_queues_made: AtomicU64::new(0),
            queue_switch: Arc::new(crate::queue_share::QueueSwitch::new(
                invariant.compile.policy().queue_sharing,
            )),
            depth_ranges,
            invariant,
            announced,
            lowering_refuses,
            deferred: crate::render::DeferredEncoders::default(),
            record_executions,
            views_epoch: AtomicU64::new(0),
            pipeline_recording: crate::warm::Recording::new(pipeline_records),
            warmer: crate::warm::Warmer::default(),
            device_address,
            upload_alignment,
            dynamic_buffers: [
                limits.max_descriptor_set_uniform_buffers_dynamic,
                limits.max_descriptor_set_storage_buffers_dynamic,
            ],
            memory: Arc::new(Mutex::new(memory)),
            recycled_buffers: Mutex::new(Recycler::new(recycle::BUFFERS)),
            resource_ids,
            passing_holds,
            keepers: Keepers::default(),
            sampler_ids,
            samplers: Mutex::default(),
            sampler_count: SamplerCount::new(sampler_limit),
            allocations,
            frees,
            sampler_sets: Mutex::default(),
            heaps: Mutex::default(),
            function_tables: Mutex::default(),
            internal: Mutex::new(None),
            internal_done: OwnTimeline::default(),
            staging_done: OwnTimeline::default(),
            completed_by_barrier,
            by_ends,
            bound_at: Mutex::new(None),
            staging,
            extra_queues,
            reserved_queues,
            scheduler,
            nudge,
            completion,
            fences: Arc::default(),
            contexts,
            loader_word: Arc::new(loader_word),
            owned,
            lost,
            textures,
            shader_cache,
            in_place: crate::in_place::InPlaceChecks::new(
                fail_in_place_check,
                limits.non_coherent_atom_size,
            ),
            submit_checks: Arc::new(crate::submit_check::SubmitChecks::new(
                assume_submissions_merge,
            )),
            host_import,
            #[cfg(unix)]
            opaque_fd,
            #[cfg(unix)]
            dma_buf_export,
            #[cfg(unix)]
            opaque_fd_semaphores,
            shared_pointer_imports: OnceLock::new(),
            extended_dynamic_state,
            sized_index_bind,
            mesh_draws,
            push_descriptor,
            pushes_after_pipeline_binds,
            color_write_masks,
            recording,
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
            gpu_work: AtomicU64::new(0),
            dynamic_multi_variable: OnceLock::new(),
            owned_kernels: Mutex::new(OwnedKernels::default()),
            ycbcr_formats,
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
            host_ram: Default::default(),
            copied: Default::default(),
            #[cfg(test)]
            failing_host_imports: Default::default(),
            #[cfg(test)]
            failing_submissions: Default::default(),
            #[cfg(test)]
            linear_rule_forced: Default::default(),
            #[cfg(test)]
            storage_costs_forced: Default::default(),
            #[cfg(test)]
            plane_route_forced: Default::default(),
            dirty: Default::default(),
            standings: Default::default(),
            fresh: Default::default(),
            owners: Default::default(),
            live_buffers: Default::default(),
            carry: Default::default(),
            #[cfg(unix)]
            residency: Default::default(),
            timing,
            view_reinterpretation,
            generated_commands,
            host_image_copy,
            host_copies,
            zero_initialize_device_memory,
            sparse,
            spares: Spares::new(trims_spares),
            live_binds: Default::default(),
            coverage_pools: Default::default(),
            live_dispatches: Default::default(),
            usage_shadows_made: Default::default(),
            heap_textures_end: Default::default(),
            bind_memos: Default::default(),
            cores: Default::default(),
            draw_rooms: Default::default(),
            kept_rooms: Default::default(),
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
        // but not where the driver loses a signal that lands while its wait for any of several
        // objects registers a later one (`RuntimeDriverDefect::WaitForAnyObjectLosesASignal`,
        // `driver_defects.rs`): there the turns wake it. A defect fact, never the driver's name.
        if waits_on_gpu {
            (shared.scheduler).wait_on_gpu(Functions::clone(&shared.functions), shared.nudge);
        }
        // Where the driver stalls in one signalling submission of every 1024 to a `VkQueue`
        // (`DriverFact::SignallingSubmitsStallEvery1024`), the scheduler's thread takes the
        // stalls while idle, each queue's empty submissions signalling a semaphore of its own
        // (`submit_stalls.rs`); a device that cannot make the semaphores takes none.
        let stalls = metal2vulkan::device::DriverFact::SignallingSubmitsStallEvery1024;
        if shared.invariant.properties.driver_facts.contains(&stalls) {
            if let Some(semaphores) = submit_stalls::semaphores(shared.vk(), shared.queues.len()) {
                let functions = Functions::clone(&shared.functions);
                let queues = Arc::clone(&shared.queues);
                let idle = submit_stalls::IdleStalls::new(functions, queues, semaphores);
                shared.scheduler.take_stalls_idle(idle);
            }
        }
        // The line serves its first generation; a recovery's serves it once it is made.
        shared.line.first(&shared);
        let device = holds::program(shared);
        // What the device allocated ahead goes back once its allocations come near the limit.
        device.give_back_ahead_near_the_limit();
        // The verdicts of its check of textures over host pages, where a record holds them.
        device.recall_in_place_checks();
        // Its check's verdict of the GPU idling between `VkQueue`s, where none is to be measured
        // or a record holds it (`queue_share.rs`).
        device.recall_queue_switch_check();
        // Its submission checks' verdicts, where a record holds them; else the checks start
        // (`submit_check.rs`).
        device.recall_submit_checks();
        device.degrade_argument_buffers_unread();
        device.degrade_float_tininess();
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
