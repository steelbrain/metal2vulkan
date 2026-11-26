//! Metal's indirect command buffers of indexed draws (`MTLIndirectCommandBuffer`), on
//! `VK_EXT_device_generated_commands` ([`crate::dgc`]).
//!
//! A buffer is its records, in a Private block the runtime takes, laid out as `metal2vulkan`'s
//! [`IndirectDrawLayout`] names: a header, then one command a stride. A kernel's encoding writes
//! them through the buffer's `gpuResourceID`, the header's device address, which the consumer's
//! argument encoder writes into the `command_buffer` member; an execution
//! ([`RenderCommandEncoder::execute_commands_in_buffer`](crate::RenderCommandEncoder::execute_commands_in_buffer))
//! reads them, one `VkIndirectCommandsLayoutEXT` sequence a command.
//!
//! A new buffer's commands, and the commands a reset names
//! ([`BlitCommandEncoder::reset_commands_in_buffer`](crate::BlitCommandEncoder::reset_commands_in_buffer)),
//! are copied from the device's template of reset commands of the buffer's layout: up to
//! [`TEMPLATE_BYTES`] of commands each drawing nothing, copied in regions over the range, Metal's
//! own write of it. The device keeps at most [`TEMPLATES`] templates, the least recently used
//! forgotten first; a command buffer or upload that copies from one keeps it until it retires.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;
use metal2vulkan::pipeline::IndirectDrawLayout;

use crate::counters::{Gauge, Metric, Site};
use crate::device::{lock, Core};
use crate::dgc::{self, Dgc};
use crate::error::{Error, Gap, IndirectCommandsKind};
use crate::fault::RangeKind;
use crate::floor::IndirectCommandLimits;
use crate::kernel_map::Lru;
use crate::memory::{self, Block};
use crate::pipeline::{address_word, FieldValue, PUSH_STAGES};
use crate::record::{self, Reach};
use crate::render::{DrawPlace, Draws, Raster, RenderCompiled, VERTEX};
use crate::submit::Held;
use crate::{Device, MemoryClass};

/// The most templates of reset commands a device keeps, one per record layout.
pub(crate) const TEMPLATES: usize = 16;
/// The most bytes of one template: `vkCmdUpdateBuffer`'s most.
pub(crate) const TEMPLATE_BYTES: u32 = 65536;
/// The most command layouts (`VkIndirectCommandsLayoutEXT`) a device keeps, one per shape of
/// tokens ([`Tokens`]).
pub(crate) const LAYOUTS: usize = 64;
/// The most execution shapes (a content, a raster state, a record layout and a command count)
/// whose layout and preprocess requirements a device keeps ([`Shape`]).
pub(crate) const SHAPES: usize = 256;
/// The most preprocess bytes one `vkCmdExecuteGeneratedCommandsEXT` takes, as
/// `vkGetGeneratedCommandsMemoryRequirementsEXT` answers for its commands: an execution of more is
/// one of each part of the most commands that stay at or under it ([`Shape::part`]), in order, the
/// last the rest, each its own preprocess in the execution's region, its sequences counted from the
/// part's first command (`PushField::IndirectBase`). Half of about 4 MiB, past which a driver was
/// measured to draw nothing of an execution or to lose the device (README). One rule on every
/// device; the bytes a command are the device's own answer.
pub(crate) const PART_BYTES: u64 = 2 << 20;
/// The most bytes of preprocess regions no execution holds that a device keeps for the next ones:
/// the regions of one frame of Apple's published rendering sample's 45,024 commands, which a
/// completed frame gives back for a later one to take (in a steady run only one frame's regions
/// are idle; those of the frames in flight are in use): its 11 executions' parts hold the commands'
/// own preprocess bytes (about 27 MB at the 592 bytes a command of 17 tokens measured on one
/// device) and at most one part of [`PART_BYTES`] more an execution (22 MB), with room for the
/// regions' rounding. A region made within a second of one freed past it is counted
/// ([`Metrics::preprocess_regions_remade`](crate::Metrics::preprocess_regions_remade)), and every
/// idle region goes once it has been idle for the runtime's idle age
/// ([`memory::IDLE_AGE_NANOS`]): at rest the pool holds nothing.
pub(crate) const IDLE_BYTES: u64 = 64 << 20;
/// The granule a preprocess region's bytes are rounded up to.
const REGION_GRANULE: u64 = 64 << 10;
/// The bounds of a buffer slot a command holds, which an execution pushes for it: its offset
/// before the address (2^31 - 1) and its length (`u32::MAX`), the widest a checked access passes,
/// and not the all-ones pair of a slot bound to nothing. An access through the slot reads and
/// writes at the address the command holds plus its offset, unchecked, as one through a pointer
/// loaded from memory does.
pub(crate) const UNCHECKED_BOUNDS: [u8; 8] = {
    let (before, len) = (0x7fff_ffff_u32.to_le_bytes(), u32::MAX.to_le_bytes());
    [
        before[0], before[1], before[2], before[3], len[0], len[1], len[2], len[3],
    ]
};

/// `MTLIndirectCommandType`: the kinds of command a buffer holds, Metal's bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct IndirectCommandType(u64);

impl IndirectCommandType {
    /// `MTLIndirectCommandTypeDraw`.
    pub const DRAW: IndirectCommandType = IndirectCommandType(1 << 0);
    /// `MTLIndirectCommandTypeDrawIndexed`: the one kind the runtime builds.
    pub const DRAW_INDEXED: IndirectCommandType = IndirectCommandType(1 << 1);
    /// `MTLIndirectCommandTypeDrawPatches`.
    pub const DRAW_PATCHES: IndirectCommandType = IndirectCommandType(1 << 2);
    /// `MTLIndirectCommandTypeDrawIndexedPatches`.
    pub const DRAW_INDEXED_PATCHES: IndirectCommandType = IndirectCommandType(1 << 3);
    /// `MTLIndirectCommandTypeConcurrentDispatch`.
    pub const CONCURRENT_DISPATCH: IndirectCommandType = IndirectCommandType(1 << 5);
    /// `MTLIndirectCommandTypeConcurrentDispatchThreads`.
    pub const CONCURRENT_DISPATCH_THREADS: IndirectCommandType = IndirectCommandType(1 << 6);
    /// `MTLIndirectCommandTypeDrawMeshThreadgroups`.
    pub const DRAW_MESH_THREADGROUPS: IndirectCommandType = IndirectCommandType(1 << 7);
    /// `MTLIndirectCommandTypeDrawMeshThreads`.
    pub const DRAW_MESH_THREADS: IndirectCommandType = IndirectCommandType(1 << 8);

    /// The type of Metal's bits `bits`.
    pub const fn from_bits(bits: u64) -> IndirectCommandType {
        IndirectCommandType(bits)
    }

    /// Metal's bits.
    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl std::ops::BitOr for IndirectCommandType {
    type Output = IndirectCommandType;

    fn bitor(self, other: IndirectCommandType) -> IndirectCommandType {
        IndirectCommandType(self.0 | other.0)
    }
}

/// `MTLIndirectCommandBufferDescriptor`, of the properties the runtime reads. Built: commands of
/// [`IndirectCommandType::DRAW_INDEXED`] alone that inherit the executing encoder's pipeline
/// state and bind their own buffers; every other answer is [`Gap::IndirectCommandBuffers`] at
/// [`Device::new_indirect_command_buffer`]. The render states the commands draw with are the
/// executing encoder's, whatever Metal's `inherit*` properties say, as an M4's commands draw
/// with them.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndirectCommandBufferDescriptor {
    /// `commandTypes`.
    pub command_types: IndirectCommandType,
    /// `inheritPipelineState`: commands draw with the executing encoder's pipeline state. Metal's
    /// default is false; the runtime builds true alone.
    pub inherit_pipeline_state: bool,
    /// `inheritBuffers`: commands draw with the executing encoder's buffers. The runtime builds
    /// false alone.
    pub inherit_buffers: bool,
    /// `maxVertexBufferBindCount`: the vertex buffer slots below which a command binds buffers,
    /// at most 31.
    pub max_vertex_buffer_bind_count: u32,
    /// `maxFragmentBufferBindCount`, as `max_vertex_buffer_bind_count`.
    pub max_fragment_buffer_bind_count: u32,
    /// `maxKernelBufferBindCount`: read by no render command.
    pub max_kernel_buffer_bind_count: u32,
}

impl IndirectCommandBufferDescriptor {
    /// A descriptor with Metal's defaults.
    pub fn new() -> IndirectCommandBufferDescriptor {
        IndirectCommandBufferDescriptor::default()
    }
}

/// `MTLIndirectCommandBuffer`: a buffer of indexed draw commands, the handle of its records.
/// Clones are the same buffer. Its records live while a handle, or a command buffer that uses it,
/// holds them.
#[derive(Clone)]
pub struct IndirectCommandBuffer {
    inner: Arc<IcbInner>,
}

struct IcbInner {
    core: Arc<Core>,
    records: Arc<Block>,
    layout: IndirectDrawLayout,
    capacity: u32,
}

impl Drop for IcbInner {
    fn drop(&mut self) {
        self.core.counters.fall(Gauge::IndirectCommandBuffers);
    }
}

impl IndirectCommandBuffer {
    /// `size`: the commands it holds, its `maxCommandCount`.
    pub fn size(&self) -> u32 {
        self.inner.capacity
    }

    /// `gpuResourceID`: the device address of its records' header, which a kernel's encoding
    /// reaches the records through (Metal states no value). The consumer writes it into an
    /// argument buffer's `command_buffer` member.
    pub fn gpu_resource_id(&self) -> u64 {
        self.inner.records.memory.address
    }

    /// Whether `core` is the device it was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(&self.inner.core, core)
    }

    /// Its records' block.
    pub(crate) fn records(&self) -> &Arc<Block> {
        &self.inner.records
    }

    /// Its records' layout.
    pub(crate) fn layout(&self) -> IndirectDrawLayout {
        self.inner.layout
    }
}

/// The device's state for indirect command buffers, where it enabled
/// `VK_EXT_device_generated_commands` (`floor::INDIRECT_COMMANDS`).
pub(crate) struct Indirect {
    pub(crate) dgc: Dgc,
    pub(crate) limits: IndirectCommandLimits,
    templates: Mutex<Lru<IndirectDrawLayout, Arc<Template>>>,
    layouts: Mutex<Lru<Tokens, Arc<CommandsLayout>>>,
    shapes: Mutex<Lru<(u64, Raster, IndirectDrawLayout, u32), Shape>>,
    /// The idle regions, the oldest first.
    regions: Mutex<Vec<Region>>,
    /// When the last region freed past [`IDLE_BYTES`] went, in nanoseconds of
    /// [`host_time`](crate::host_time); 0 before the first.
    evicted_at: AtomicU64,
}

/// The tokens of an execution of a content's commands of one record layout: each slot word the
/// content pushes, a token of its own (its offset in a command and its push offset), then the
/// index buffer, the sequence index at its push offset, and the indexed draw at the layout's
/// offsets; and the push offset of the part's first command, which an execution pushes before
/// each part.
///
/// A token of one 8-byte word each, never one token for a run of words: on NVIDIA's driver
/// (620.70.06, an RTX 2080) a 24-byte push constant token of an execution of more than 64
/// sequences carried an earlier sequence's bytes, where three 8-byte tokens of the same bytes
/// carried each sequence's own.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tokens {
    pub(crate) layout: IndirectDrawLayout,
    pub(crate) pushes: Vec<(u32, u32)>,
    pub(crate) sequence: u32,
    pub(crate) base: u32,
}

impl Tokens {
    /// The tokens a command takes.
    pub(crate) fn count(&self) -> u32 {
        self.pushes.len() as u32 + 3
    }
}

/// A `VkIndirectCommandsLayoutEXT`, destroyed once neither the device's cache nor a command
/// buffer that executed through it holds it.
pub(crate) struct CommandsLayout {
    handle: u64,
    device: vk::Device,
    destroy: dgc::DestroyLayout,
}

impl CommandsLayout {
    /// Its `VkIndirectCommandsLayoutEXT`.
    pub(crate) fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for CommandsLayout {
    fn drop(&mut self) {
        // SAFETY: the layout is the device's, which is live while a cache or a command buffer
        // holds it, and no submitted command that executes through it is pending: every command
        // buffer that did holds it until it retires.
        unsafe { (self.destroy)(self.device, self.handle, std::ptr::null()) };
    }
}

/// An execution shape's layout, its part and a part's preprocess requirements, with the inherited
/// pipeline.
#[derive(Clone)]
pub(crate) struct Shape {
    pub(crate) layout: Arc<CommandsLayout>,
    /// The push offset of the part's first command ([`Tokens::base`]).
    pub(crate) base: u32,
    /// The commands of a part: the most whose preprocess bytes stay at or under [`PART_BYTES`].
    pub(crate) part: u32,
    preprocess: vk::MemoryRequirements,
}

impl Shape {
    /// The preprocess bytes a part of the shape takes.
    pub(crate) fn preprocess_size(&self) -> u64 {
        self.preprocess.size
    }

    /// The bytes from one part's preprocess to the next in an execution's region: its size at
    /// its alignment.
    pub(crate) fn part_stride(&self) -> u64 {
        (self.preprocess.size).next_multiple_of(self.preprocess.alignment.max(1))
    }
}

/// A preprocess region: a buffer of its own, with
/// `VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT`, on its own allocation.
pub(crate) struct Region {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_type: u32,
    address: u64,
    size: u64,
    /// When it last became idle, in nanoseconds of [`host_time`](crate::host_time).
    idle_since: u64,
}

impl Region {
    /// It, for the block-maker thread to free ([`memory::Loose`]).
    pub(crate) fn into_loose(self) -> memory::Loose {
        memory::Loose::Preprocess {
            buffer: self.buffer,
            memory: self.memory,
            bytes: self.size,
        }
    }

    /// Whether the region holds `need`, from its first byte at `need`'s alignment.
    fn holds(&self, need: &vk::MemoryRequirements) -> bool {
        let start = self.address.next_multiple_of(need.alignment.max(1)) - self.address;
        need.memory_type_bits & (1 << self.memory_type) != 0
            && start.saturating_add(need.size) <= self.size
    }

    /// Frees it.
    ///
    /// # Safety
    ///
    /// No pending work uses it.
    unsafe fn destroy(self, core: &Core) {
        (core.spans).freed(self.buffer, core.pool.in_flight(), None);
        // SAFETY: by the caller.
        unsafe {
            core.device.destroy_buffer(self.buffer, None);
            memory::free(&core.device, self.memory);
        }
        core.counters.fall(Gauge::Blocks);
    }
}

/// What a command buffer that executed commands holds until it retires: the execution's layout
/// and its preprocess region, which goes back to the device's pool then.
pub(crate) struct Executed {
    core: Arc<Core>,
    _layout: Arc<CommandsLayout>,
    region: Option<Region>,
}

impl Drop for Executed {
    fn drop(&mut self) {
        if let (Some(indirect), Some(region)) = (&self.core.indirect, self.region.take()) {
            indirect.give_back(&self.core, region);
        }
    }
}

/// A template of reset commands of one layout ([`Indirect::template`]): `commands` of them, from
/// the block's first byte.
pub(crate) struct Template {
    block: Arc<Block>,
    commands: u32,
}

impl Indirect {
    pub(crate) fn new(dgc: Dgc, limits: IndirectCommandLimits) -> Indirect {
        Indirect {
            dgc,
            limits,
            templates: Mutex::new(Lru::new(TEMPLATES)),
            layouts: Mutex::new(Lru::new(LAYOUTS)),
            shapes: Mutex::new(Lru::new(SHAPES)),
            regions: Mutex::new(Vec::new()),
            evicted_at: AtomicU64::new(0),
        }
    }

    /// The shape of an execution of at most `capacity` commands of `layout` with `compiled`'s
    /// `pipeline`, made under `raster`: the device's, or one made now, its layout the device's for
    /// its tokens ([`tokens`]) or one made now, and its part and a part's preprocess requirements
    /// queried with the pipeline ([`part`](Self::part)), which an execution with another of the
    /// content's pipelines does not reuse.
    ///
    /// Err: [`Gap::IndirectCommandBuffers`] where [`tokens`] answers a kind, and of
    /// [`IndirectCommandsKind::Tokens`] for more tokens than the device takes; [`Error::Vulkan`]
    /// where the layout cannot be made.
    pub(crate) fn shape(
        &self,
        core: &Core,
        (compiled, raster, pipeline): (&RenderCompiled, Raster, vk::Pipeline),
        layout: IndirectDrawLayout,
        capacity: u32,
    ) -> Result<Shape, Error> {
        let key = (compiled.id, raster, layout, capacity);
        if let Some(shape) = lock(&self.shapes).get(&key) {
            return Ok(shape);
        }
        let refuse = |kind| Error::Unsupported(Gap::IndirectCommandBuffers { kind });
        let tokens = tokens(&compiled.draws, layout).map_err(refuse)?;
        if tokens.count() > self.limits.tokens {
            return Err(refuse(IndirectCommandsKind::Tokens));
        }
        let commands = self.layout(core, &tokens)?;
        let (part, preprocess) = self.part(core, &commands, pipeline, capacity);
        let shape = Shape {
            layout: commands,
            base: tokens.base,
            part,
            preprocess,
        };
        let (kept, gone) = lock(&self.shapes).keep(key, shape);
        drop(gone);
        Ok(kept)
    }

    /// The commands of a part of an execution of at most `capacity` commands of `commands` with
    /// `pipeline`, and a part's preprocess requirements: every command where the device's
    /// preprocess bytes for them (`vkGetGeneratedCommandsMemoryRequirementsEXT`) stay at or under
    /// [`PART_BYTES`], else about the most that do. The device is asked from one command up,
    /// doubling while the answer stays at or under the bound, so no answer for far more commands
    /// than a part takes is trusted (one driver's answer grows with the square of the count and
    /// wraps at 32 bits); past the last count that stayed under it, the device is asked again at
    /// fewer, in proportion to the last answer's bytes, until the answer is at or under it.
    pub(crate) fn part(
        &self,
        core: &Core,
        commands: &CommandsLayout,
        pipeline: vk::Pipeline,
        capacity: u32,
    ) -> (u32, vk::MemoryRequirements) {
        let info = dgc::PipelineInfo::new(pipeline);
        let ask = |sequences: u32| {
            let asked = dgc::MemoryInfo::new(&info, commands.handle, sequences);
            let mut requirements: vk::MemoryRequirements2<'static> =
                vk::MemoryRequirements2::default();
            // SAFETY: the layout and the pipeline are the device's, and the structures are valid
            // for the call.
            unsafe {
                (self.dgc.memory_requirements)(core.device.handle(), &asked, &mut requirements)
            };
            requirements.memory_requirements
        };
        let (mut part, mut need) = (1, ask(1));
        while part < capacity {
            let next = part.saturating_mul(2).min(capacity);
            let more = ask(next);
            if more.size <= PART_BYTES {
                (part, need) = (next, more);
                continue;
            }
            let (mut over, mut bytes) = (next, more.size);
            loop {
                let fewer = u128::from(PART_BYTES) * u128::from(over) / u128::from(bytes);
                let count = u32::try_from(fewer).map_or(over - 1, |fewer| fewer.min(over - 1));
                if count <= part {
                    break;
                }
                let asked = ask(count);
                if asked.size <= PART_BYTES {
                    (part, need) = (count, asked);
                    break;
                }
                (over, bytes) = (count, asked.size);
            }
            break;
        }
        (part, need)
    }

    /// The command layout of `tokens`: the device's, or one made now, through the device's
    /// pipeline layout, for vertex and fragment stages, its push tokens' ranges of
    /// [`PUSH_STAGES`], the layout's one push range's stages.
    pub(crate) fn layout(
        &self,
        core: &Core,
        tokens: &Tokens,
    ) -> Result<Arc<CommandsLayout>, Error> {
        if let Some(layout) = lock(&self.layouts).get(tokens) {
            return Ok(layout);
        }
        let range = |offset, size| dgc::PushConstantToken {
            update_range: vk::PushConstantRange {
                stage_flags: PUSH_STAGES,
                offset,
                size,
            },
        };
        let pushes: Vec<dgc::PushConstantToken> = (tokens.pushes.iter())
            .map(|&(_, push)| range(push, IndirectDrawLayout::SLOT_BYTES))
            .collect();
        let sequence = range(tokens.sequence, 4);
        let index = dgc::IndexBufferToken {
            mode: dgc::INPUT_MODE_VULKAN,
        };
        let records = tokens.layout;
        let mut list: Vec<dgc::LayoutToken> = (tokens.pushes.iter().zip(&pushes))
            .map(|(&(at, ..), push)| {
                dgc::LayoutToken::new(
                    dgc::TOKEN_PUSH_CONSTANT,
                    std::ptr::from_ref(push).cast(),
                    at,
                )
            })
            .collect();
        list.push(dgc::LayoutToken::new(
            dgc::TOKEN_INDEX_BUFFER,
            std::ptr::from_ref(&index).cast(),
            records.index(),
        ));
        list.push(dgc::LayoutToken::new(
            dgc::TOKEN_SEQUENCE_INDEX,
            std::ptr::from_ref(&sequence).cast(),
            records.sequence(),
        ));
        list.push(dgc::LayoutToken::new(
            dgc::TOKEN_DRAW_INDEXED,
            std::ptr::null(),
            records.draw(),
        ));
        let info = dgc::LayoutCreateInfo::new(records.stride(), core.layout, &list);
        let mut handle = 0;
        let device = core.device.handle();
        // SAFETY: the structures, and every token's data, live for the call; the pipeline layout
        // is the device's.
        let result =
            unsafe { (self.dgc.create_layout)(device, &info, std::ptr::null(), &mut handle) };
        if result != vk::Result::SUCCESS {
            return Err(Error::Vulkan {
                call: "vkCreateIndirectCommandsLayoutEXT",
                result,
            });
        }
        let made = Arc::new(CommandsLayout {
            handle,
            device,
            destroy: self.dgc.destroy_layout,
        });
        let (kept, gone) = lock(&self.layouts).keep(tokens.clone(), made);
        drop(gone);
        Ok(kept)
    }

    /// A preprocess region for `parts` parts of `shape`, one after another
    /// ([`Shape::part_stride`]), held by what the call answers until the command buffer it
    /// records into retires: an idle one that holds it at most twice over, or one made now of its
    /// bytes rounded up to 64 KiB, in a memory type its requirements and the buffer's allow, a
    /// device-local one first. None where the shape needs no bytes. Nothing waits and no legal
    /// execution is refused for the pool's sake: only the device's memory bounds what is in use.
    ///
    /// Err: [`Error::Vulkan`] where the buffer or its memory cannot be made;
    /// [`Limit::MemoryAllocations`](crate::Limit::MemoryAllocations).
    pub(crate) fn executed(
        &self,
        core: &Arc<Core>,
        shape: &Shape,
        parts: u32,
    ) -> Result<(Executed, u64), Error> {
        let need = vk::MemoryRequirements {
            size: shape.part_stride() * u64::from(parts),
            ..shape.preprocess
        };
        let region = match need.size {
            0 => None,
            _ => Some(self.take(core, &need)?),
        };
        let address = (region.as_ref()).map_or(0, |region| {
            region.address.next_multiple_of(need.alignment.max(1))
        });
        let executed = Executed {
            core: Arc::clone(core),
            _layout: Arc::clone(&shape.layout),
            region,
        };
        Ok((executed, address))
    }

    /// [`executed`](Self::executed)'s region.
    fn take(&self, core: &Core, need: &vk::MemoryRequirements) -> Result<Region, Error> {
        let most = need.size.saturating_mul(2).max(REGION_GRANULE);
        let found = {
            let mut idle = lock(&self.regions);
            let best = (idle.iter().enumerate())
                .filter(|(_, region)| region.holds(need) && region.size <= most)
                .min_by_key(|(_, region)| region.size)
                .map(|(at, _)| at);
            best.map(|at| idle.remove(at))
        };
        let region = match found {
            Some(region) => {
                core.counters
                    .fall_by(Gauge::PreprocessIdleBytes, region.size);
                region
            }
            None => {
                let evicted = self.evicted_at.load(Ordering::Relaxed);
                if evicted != 0 && memory::remade(evicted, crate::host_time()) {
                    core.counters.count(Metric::PreprocessRegionsRemade);
                }
                let size = (need.size + need.alignment).next_multiple_of(REGION_GRANULE);
                // SAFETY: the device is live.
                unsafe { make_region(core, size, need.memory_type_bits) }?
            }
        };
        core.counters
            .rise_by(Gauge::PreprocessUsedBytes, region.size);
        Ok(region)
    }

    /// Takes back `region` from a retired command buffer: idle from now, the oldest idle regions
    /// given to the block-maker thread to free while the idle bytes pass [`IDLE_BYTES`], and the
    /// block-maker thread woken to free each one once it has been idle for the idle age
    /// ([`due`](Self::due)). No driver free runs here, on the completion thread.
    fn give_back(&self, core: &Core, mut region: Region) {
        core.counters
            .fall_by(Gauge::PreprocessUsedBytes, region.size);
        core.counters
            .rise_by(Gauge::PreprocessIdleBytes, region.size);
        region.idle_since = crate::host_time();
        let mut freed = Vec::new();
        {
            let mut idle = lock(&self.regions);
            idle.push(region);
            let mut bytes: u64 = idle.iter().map(|region| region.size).sum();
            while bytes > IDLE_BYTES && !idle.is_empty() {
                let oldest = idle.remove(0);
                bytes -= oldest.size;
                freed.push(oldest);
            }
        }
        if !freed.is_empty() {
            (self.evicted_at).store(crate::host_time(), Ordering::Relaxed);
        }
        for region in freed {
            // SAFETY: an idle region is in no pending work.
            unsafe { core.pool.give_back_loose(region.into_loose()) };
        }
        core.pool.nudge();
    }

    /// On the block-maker thread: the oldest idle region, taken out, where it has been idle for
    /// the idle age ([`memory::IDLE_AGE_NANOS`]) at `now`; else when it will have been (`Err`).
    /// `None` where no region is idle.
    pub(crate) fn due(&self, now: u64) -> Option<Result<Region, u64>> {
        let mut idle = lock(&self.regions);
        let since = idle.first()?.idle_since;
        Some(match memory::idle_due(since, now) {
            None => Ok(idle.remove(0)),
            Some(at) => Err(at),
        })
    }

    /// The template of reset commands of `layout`: the device's, or one made now, written in the
    /// next commit's upload, as many commands of the layout as [`TEMPLATE_BYTES`] holds (one at
    /// least). Each command's slot words are the nil buffer's (`Ring::nil`), and its index token
    /// binds the zero range (`Ring::zero`) with no bytes.
    ///
    /// Err: what taking its block, the ring's two ranges or the upload return.
    pub(crate) fn template(
        &self,
        core: &Arc<Core>,
        layout: IndirectDrawLayout,
    ) -> Result<Arc<Template>, Error> {
        if let Some(template) = lock(&self.templates).get(&layout) {
            return Ok(template);
        }
        let stride = layout.stride();
        let commands = (TEMPLATE_BYTES / stride).max(1);
        let memory = core.take_memory(
            MemoryClass::Private,
            u64::from(commands * stride),
            Site::PrivateTake,
        )?;
        let block = Arc::new(Block::new(memory, Arc::clone(core)));
        let nil = address_word(core.ring.nil(core)?);
        let command = layout.reset_command(nil, core.ring.zero(core)?);
        let bytes = command.repeat(commands as usize);
        let (buffer, base) = block.memory.range(0);
        let held = [Held::Block(Arc::clone(&block))];
        core.work.upload(core, None, held, |device, upload| {
            // SAFETY: the upload is recording; the block holds the bytes, at most 65,536 of
            // them, a multiple of 8 at a multiple of 4.
            unsafe { device.cmd_update_buffer(upload, buffer, base, &bytes) };
        })?;
        let template = Arc::new(Template { block, commands });
        // A template another thread kept first is the one kept; this one goes once its upload
        // retires.
        let (kept, gone) = lock(&self.templates).keep(layout, template);
        drop(gone);
        Ok(kept)
    }

    /// Forgets every template, shape and layout, and frees every idle region, at the device's
    /// end, once no command buffer holds any.
    pub(crate) fn clear(&self, core: &Core) {
        let templates = lock(&self.templates).clear();
        let shapes = lock(&self.shapes).clear();
        let layouts = lock(&self.layouts).clear();
        drop((templates, shapes, layouts));
        let idle: Vec<Region> = lock(&self.regions).drain(..).collect();
        for region in idle {
            core.counters
                .fall_by(Gauge::PreprocessIdleBytes, region.size);
            // SAFETY: the device's threads are joined and its command buffers retired.
            unsafe { region.destroy(core) };
        }
    }

    /// For the tests: the templates, shapes and layouts held.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn held(&self) -> (usize, usize, usize) {
        (
            lock(&self.templates).len(),
            lock(&self.shapes).len(),
            lock(&self.layouts).len(),
        )
    }
}

/// The tokens of an execution of `draws`' content's commands of `layout` ([`Tokens`]): a push
/// constant token for each slot word the content pushes of a slot the command holds, then the
/// index buffer, the sequence index at the content's `IndirectSequence` field and the indexed
/// draw; the part's first command goes at its `IndirectBase` field.
///
/// Err: [`IndirectCommandsKind::UnflaggedState`] for a content whose vertex function reads no
/// execution range; [`IndirectCommandsKind::DrawInput`] for one that reads a slot the command
/// binds outside its push constants, its command outside them, a fixed-function vertex binding,
/// or that must draw at base instance 0.
pub(crate) fn tokens(
    draws: &Draws,
    layout: IndirectDrawLayout,
) -> Result<Tokens, IndirectCommandsKind> {
    let mut words = Vec::new();
    let (mut sequence, mut base, mut range) = (None, None, false);
    for &(place, table, value) in &draws.fields {
        match value {
            FieldValue::BufferAddress { slot, .. } => {
                let (count, at) = match table {
                    VERTEX => (layout.vertex_slots(), layout.vertex_slot(slot)),
                    _ => (layout.fragment_slots(), layout.fragment_slot(slot)),
                };
                if slot >= count {
                    continue;
                }
                match place {
                    DrawPlace::Push(push) => words.push((at, push)),
                    DrawPlace::Block(..) => return Err(IndirectCommandsKind::DrawInput),
                }
            }
            FieldValue::IndirectSequence => match place {
                DrawPlace::Push(push) => sequence = Some(push),
                DrawPlace::Block(..) => return Err(IndirectCommandsKind::DrawInput),
            },
            FieldValue::IndirectBase => match place {
                DrawPlace::Push(push) => base = Some(push),
                DrawPlace::Block(..) => return Err(IndirectCommandsKind::DrawInput),
            },
            FieldValue::IndirectRange => range = true,
            _ => {}
        }
    }
    let (Some(sequence), Some(base), true) = (sequence, base, range) else {
        return Err(IndirectCommandsKind::UnflaggedState);
    };
    if !draws.bindings.is_empty() || draws.first_instance_zero {
        return Err(IndirectCommandsKind::DrawInput);
    }
    words.sort_unstable();
    Ok(Tokens {
        layout,
        pushes: words,
        sequence,
        base,
    })
}

/// The copies of `template`'s commands over commands `range` of `records` of `layout`: one
/// region for each run of as many commands as the template holds.
pub(crate) fn reset_regions(
    template: &Template,
    layout: IndirectDrawLayout,
    records: &Block,
    range: Range<u32>,
) -> (vk::Buffer, vk::Buffer, Vec<vk::BufferCopy>) {
    let (source, source_base) = template.block.memory.range(0);
    let (destination, base) = records.memory.range(0);
    let stride = u64::from(layout.stride());
    let regions = (range.clone())
        .step_by(template.commands as usize)
        .map(|first| {
            let commands = (range.end - first).min(template.commands);
            vk::BufferCopy {
                src_offset: source_base,
                dst_offset: base + layout.command(first),
                size: u64::from(commands) * stride,
            }
        })
        .collect();
    (source, destination, regions)
}

impl Template {
    /// Its block, which a copy from it keeps.
    pub(crate) fn block(&self) -> &Arc<Block> {
        &self.block
    }
}

/// A preprocess region of `size` bytes in a memory type of `bits` and the buffer's own.
///
/// # Safety
///
/// The device is live.
unsafe fn make_region(core: &Core, size: u64, bits: u32) -> Result<Region, Error> {
    let device = &core.device;
    let mut usage = vk::BufferUsageFlags2CreateInfoKHR::default()
        .usage(dgc::PREPROCESS_USAGE | vk::BufferUsageFlags2KHR::SHADER_DEVICE_ADDRESS);
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut usage);
    // SAFETY: the device is live, and the create info valid.
    let buffer =
        unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))?;
    // SAFETY: the buffer is the device's.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let allowed = bits & requirements.memory_type_bits;
    let types = (0..u32::BITS).filter(|&index| allowed & (1 << index) != 0);
    let Some(memory_type) = types.min_by_key(|&index| core.device_local & (1 << index) == 0) else {
        // SAFETY: the buffer is unused.
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(Error::Vulkan {
            call: "vkGetGeneratedCommandsMemoryRequirementsEXT",
            result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        });
    };
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size.max(size))
        .memory_type_index(memory_type)
        .push_next(&mut flags);
    // SAFETY: the memory type is one the buffer allows, and `bufferDeviceAddress` is enabled.
    let memory = match unsafe {
        memory::allocate(
            device,
            &core.counters,
            core.max_allocations,
            &allocate,
            None,
        )
    } {
        Ok(memory) => memory,
        Err(error) => {
            // SAFETY: the buffer is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    let region = Region {
        buffer,
        memory,
        memory_type,
        address: 0,
        size,
        idle_since: 0,
    };
    // SAFETY: the memory is the buffer's type and size, and unbound.
    if let Err(result) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        // SAFETY: nothing uses it.
        unsafe { region.destroy(core) };
        return Err(Error::Vulkan {
            call: "vkBindBufferMemory",
            result,
        });
    }
    let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    // SAFETY: the buffer is bound and of `SHADER_DEVICE_ADDRESS` usage.
    let address = unsafe { device.get_buffer_device_address(&address_info) };
    (core.spans).made(buffer, address, size, RangeKind::Preprocess);
    Ok(Region { address, ..region })
}

impl Device {
    /// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`: a buffer of
    /// `max_command_count` indexed draw commands, its records in a Private block the runtime
    /// takes (the options' storage mode is not read: the CPU encodes no command). Its commands are
    /// reset, each drawing nothing, by copies from the device's template of its layout in the
    /// upload of the next commit, ahead of every command buffer committed after the call; its
    /// header is written there too.
    ///
    /// Err: [`Gap::IndirectCommandBuffers`] of [`IndirectCommandsKind::Extension`] on a device
    /// without the extension, [`IndirectCommandsKind::CommandTypes`] for types other than indexed
    /// draws alone, [`IndirectCommandsKind::CommandPipelines`] for commands that set their own
    /// pipeline state, [`IndirectCommandsKind::InheritedBuffers`] for commands that draw with the
    /// encoder's buffers, [`IndirectCommandsKind::BindCount`] for more than 31 vertex or fragment
    /// buffers, [`IndirectCommandsKind::Capacity`] for a count of 0 or past the device's
    /// `maxIndirectSequenceCount`; [`Refusal::InScheduledHandler`](crate::Refusal::InScheduledHandler);
    /// [`Error::Quiesced`]; what taking the records' block returns; [`Error::Vulkan`] where the
    /// upload's command buffer cannot be made.
    pub fn new_indirect_command_buffer(
        &self,
        descriptor: &IndirectCommandBufferDescriptor,
        max_command_count: u32,
    ) -> Result<IndirectCommandBuffer, Error> {
        let core = self.core();
        if core.closed() {
            return Err(Error::Quiesced);
        }
        crate::command::not_in_scheduled_handler()?;
        let refuse = |kind| Err(Error::Unsupported(Gap::IndirectCommandBuffers { kind }));
        let Some(indirect) = &core.indirect else {
            return refuse(IndirectCommandsKind::Extension);
        };
        if descriptor.command_types != IndirectCommandType::DRAW_INDEXED {
            return refuse(IndirectCommandsKind::CommandTypes);
        }
        if !descriptor.inherit_pipeline_state {
            return refuse(IndirectCommandsKind::CommandPipelines);
        }
        if descriptor.inherit_buffers {
            return refuse(IndirectCommandsKind::InheritedBuffers);
        }
        let Some(layout) = IndirectDrawLayout::new(
            descriptor.max_vertex_buffer_bind_count,
            descriptor.max_fragment_buffer_bind_count,
        ) else {
            return refuse(IndirectCommandsKind::BindCount);
        };
        if max_command_count == 0 || max_command_count > indirect.limits.sequences {
            return refuse(IndirectCommandsKind::Capacity);
        }
        let template = indirect.template(core, layout)?;
        let memory = core.take_memory(
            MemoryClass::Private,
            layout.bytes(max_command_count),
            Site::PrivateTake,
        )?;
        let records = Arc::new(Block::new(memory, Arc::clone(core)));
        let header = layout.header(max_command_count, core.ring.zero(core)?);
        let (source, destination, regions) =
            reset_regions(&template, layout, &records, 0..max_command_count);
        let (_, base) = records.memory.range(0);
        let held = [
            Held::Block(Arc::clone(&records)),
            Held::Block(Arc::clone(template.block())),
        ];
        core.work.upload(core, None, held, |device, upload| {
            // SAFETY: the upload is recording; the header and every region lie in their blocks,
            // which it keeps. The barrier orders the template's own write, in this upload or an
            // earlier one, before the copies read it.
            unsafe {
                device.cmd_update_buffer(upload, destination, base, &header);
                record::barrier(device, upload, Reach::Commands);
                core.counters.count(Metric::UploadBarriers);
                device.cmd_copy_buffer(upload, source, destination, &regions);
            }
        })?;
        core.counters.rise(Gauge::IndirectCommandBuffers);
        Ok(IndirectCommandBuffer {
            inner: Arc::new(IcbInner {
                core: Arc::clone(core),
                records,
                layout,
                capacity: max_command_count,
            }),
        })
    }
}
