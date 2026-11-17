//! Compute command encoders: their binding state, their dispatches, and where barriers go.

use crate::buffer::Buffer;
use crate::command::{Building, CommandBuffer};
use crate::counters::Metric;
use crate::error::{Error, Gap, Limit, Refusal};
use crate::hazard::Access;
use crate::pipeline::{ComputePipelineState, Constant, FieldValue, Place, Variant, PUSH_BYTES};
use crate::record::{self, Reach};
use crate::ring::CHUNK;
use ash::vk;
use std::ops::BitOr;
use std::sync::Arc;

/// Metal's count of buffer argument table entries per stage.
const BUFFER_SLOTS: u32 = 31;

/// `MTLDispatchType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchType {
    /// `MTLDispatchTypeSerial`: each dispatch runs after the ones before it that it depends on;
    /// the runtime places a barrier at each hazard.
    Serial,
    /// `MTLDispatchTypeConcurrent`: dispatches may run at once; barriers go only where
    /// [`ComputeCommandEncoder::memory_barrier`] asks for one.
    Concurrent,
}

/// `MTLResourceUsage`: an option set of [`READ`](Self::READ), [`WRITE`](Self::WRITE) and
/// [`SAMPLE`](Self::SAMPLE), combined with `|`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceUsage(u32);

impl ResourceUsage {
    /// `MTLResourceUsageRead`.
    pub const READ: ResourceUsage = ResourceUsage(1);
    /// `MTLResourceUsageWrite`.
    pub const WRITE: ResourceUsage = ResourceUsage(2);
    /// `MTLResourceUsageSample`: a read, for hazards.
    pub const SAMPLE: ResourceUsage = ResourceUsage(4);

    /// The access the usage declares, or None for no usage.
    fn access(self) -> Option<Access> {
        let reads = self.0 & (Self::READ.0 | Self::SAMPLE.0) != 0;
        let writes = self.0 & Self::WRITE.0 != 0;
        match (reads, writes) {
            (true, true) => Some(Access::ReadWrite),
            (false, true) => Some(Access::Write),
            (true, false) => Some(Access::Read),
            (false, false) => None,
        }
    }
}

impl BitOr for ResourceUsage {
    type Output = ResourceUsage;

    fn bitor(self, other: ResourceUsage) -> ResourceUsage {
        ResourceUsage(self.0 | other.0)
    }
}

/// `MTLResource`, for [`ComputeCommandEncoder::use_resource`].
#[non_exhaustive]
#[derive(Clone, Copy)]
pub enum Resource<'a> {
    /// An `MTLBuffer`.
    Buffer(&'a Buffer),
}

/// What a buffer slot holds.
#[derive(Clone, Default)]
enum Slot {
    #[default]
    Empty,
    Buffer {
        buffer: Buffer,
        offset: u64,
    },
    /// The address of a `set_bytes` copy in the parameter ring.
    Bytes(u64),
}

/// The grid of a dispatch.
enum Grid<'a> {
    /// Threadgroups per grid, and the grid in threads.
    Direct([u32; 3], [u32; 3]),
    /// The arguments' buffer and offset.
    Indirect(&'a Buffer, u64),
}

/// `MTLComputeCommandEncoder`: records each dispatch into its command buffer at once.
///
/// What a dispatch touches is its pipeline's bound slots, with the access the translator states
/// for each, and the resources declared by [`use_resource`](Self::use_resource); the indirect
/// arguments' buffer is a read. A serial encoder places one global memory barrier before a
/// dispatch that reads or writes a resource written since the last barrier, or writes one read
/// since it; a pipeline that reads through an address outside its slots conflicts with any write
/// since the last barrier, and one that writes so conflicts with everything and owes a barrier
/// before the next dispatch. A concurrent encoder places a barrier only at
/// [`memory_barrier`](Self::memory_barrier). Every resource takes part, whatever its tracking
/// mode, as serial dispatch is serial order in Metal. An encoder that wrote anything owes one
/// barrier at its end, which the next command, or the segment's own end, records.
///
/// Parameters: a dispatch's parameter block (bound addresses plus offsets, `set_bytes` addresses
/// and grid values) is pushed as push constants up to 128 bytes; a longer one is written into the
/// parameter ring and its address is pushed.
///
/// Owned: one thread encodes at a time. Dropping it ends it, as
/// [`end_encoding`](Self::end_encoding) does.
pub struct ComputeCommandEncoder {
    command_buffer: CommandBuffer,
    /// What the command buffer records, held while the encoder is open.
    building: Option<Building>,
    serial: bool,
    pipeline: Option<ComputePipelineState>,
    slots: Vec<Slot>,
    /// Threadgroup memory lengths by index, as set.
    threadgroup_lengths: Vec<(u32, u32)>,
    /// `use_resource` declarations, by resource id.
    declared: Vec<(u32, Access)>,
    /// The threadgroup size, whether it was an indirect dispatch of a kernel that tells one apart,
    /// and the `VkPipeline` of the last dispatch, while the state and the threadgroup memory
    /// lengths are unchanged.
    last: Option<([u32; 3], bool, vk::Pipeline)>,
    /// The parameters being filled, kept to reuse their bytes.
    parameters: Parameters,
    /// Whether a dispatch of the encoder wrote anything.
    wrote: bool,
}

impl CommandBuffer {
    /// `computeCommandEncoderWithDispatchType:`; `computeCommandEncoder` is this call with
    /// [`DispatchType::Serial`].
    ///
    /// Err: [`Refusal::Committed`]; [`Refusal::EncoderOpen`] while another encoder of the command
    /// buffer is open.
    pub fn compute_command_encoder(
        &self,
        dispatch_type: DispatchType,
    ) -> Result<ComputeCommandEncoder, Error> {
        let mut building = self.open_encoder()?;
        let core = Arc::clone(building.device.core());
        core.counters.count(Metric::Passes);
        if dispatch_type == DispatchType::Concurrent {
            // Its dispatches take no barrier among themselves, so whatever earlier encoders read
            // or wrote is ordered before its first dispatch.
            building.owe_if_touched();
        }
        Ok(ComputeCommandEncoder {
            command_buffer: self.clone(),
            building: Some(building),
            serial: dispatch_type == DispatchType::Serial,
            pipeline: None,
            slots: vec![Slot::Empty; BUFFER_SLOTS as usize],
            threadgroup_lengths: Vec::new(),
            declared: Vec::new(),
            last: None,
            parameters: Parameters::default(),
            wrote: false,
        })
    }
}

impl ComputeCommandEncoder {
    fn building(&mut self) -> &mut Building {
        self.building
            .as_mut()
            .expect("an encoder holds its command buffer's recording until it ends")
    }

    /// `setComputePipelineState:`. The command buffer keeps the state until it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's state.
    pub fn set_compute_pipeline_state(
        &mut self,
        state: &ComputePipelineState,
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(state.inner.core(), self.building().device.core()) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        if self.pipeline.as_ref().is_some_and(|set| set.is(state)) {
            return Ok(());
        }
        self.building().keep_pipeline(state);
        self.pipeline = Some(state.clone());
        self.last = None;
        Ok(())
    }

    /// `setBuffer:offset:atIndex:` (`setBuffers:offsets:withRange:` is a loop of it). `None`
    /// unbinds the slot. The command buffer keeps the buffer's block until it retires.
    ///
    /// Err: [`Refusal::BufferIndex`] for an index of 31 or more; [`Refusal::OtherDevice`] for
    /// another device's buffer.
    pub fn set_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        let slot = self.slot(index)?;
        let bound = match buffer {
            Some(buffer) => {
                let building = self.building();
                if !buffer.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                building.keep(buffer.held_block());
                Slot::Buffer {
                    buffer: buffer.clone(),
                    offset,
                }
            }
            None => Slot::Empty,
        };
        self.slots[slot] = bound;
        Ok(())
    }

    /// `setBufferOffset:atIndex:`.
    ///
    /// Err: [`Refusal::BufferIndex`] for an index of 31 or more; [`Refusal::NoBufferAtIndex`]
    /// where no buffer is bound there.
    pub fn set_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        let slot = self.slot(index)?;
        match &mut self.slots[slot] {
            Slot::Buffer { offset: bound, .. } => {
                *bound = offset;
                Ok(())
            }
            _ => Err(Error::Refused(Refusal::NoBufferAtIndex { index })),
        }
    }

    /// `setBytes:length:atIndex:`: the bytes are copied into the parameter ring now, as Metal
    /// copies them, and the slot's address is the copy's.
    ///
    /// Err: [`Refusal::BufferIndex`] for an index of 31 or more; [`Limit::BytesLength`] past one
    /// ring chunk (64 KiB; Metal documents the call for data under 4 KB); [`Error::Vulkan`] where
    /// the ring cannot grow.
    pub fn set_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        let slot = self.slot(index)?;
        if bytes.len() as u64 > CHUNK {
            return Err(Error::Limit(Limit::BytesLength {
                len: bytes.len() as u64,
                max: CHUNK,
            }));
        }
        let address = self.building().write_ring(bytes)?;
        self.slots[slot] = Slot::Bytes(address);
        Ok(())
    }

    /// `setThreadgroupMemoryLength:atIndex:`: a specialization constant of the pipeline.
    ///
    /// Err: [`Refusal::BufferIndex`] for an index of 31 or more.
    pub fn set_threadgroup_memory_length(&mut self, length: u32, index: u32) -> Result<(), Error> {
        self.slot(index)?;
        match self
            .threadgroup_lengths
            .iter_mut()
            .find(|(set, _)| *set == index)
        {
            Some((_, set)) => *set = length,
            None => self.threadgroup_lengths.push((index, length)),
        }
        self.last = None;
        Ok(())
    }

    /// `useResource:usage:` (`useResources:count:usage:` is a loop of it): the resource joins every
    /// later dispatch's accesses. The command buffer keeps a buffer's block until it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's resource.
    pub fn use_resource(
        &mut self,
        resource: Resource<'_>,
        usage: ResourceUsage,
    ) -> Result<(), Error> {
        let Resource::Buffer(buffer) = resource;
        let building = self.building();
        if !buffer.is_of(building.device.core()) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        building.keep(buffer.held_block());
        let Some(access) = usage.access() else {
            return Ok(());
        };
        let id = buffer.id();
        match self
            .declared
            .iter_mut()
            .find(|(declared, _)| *declared == id)
        {
            Some((_, declared)) => *declared = merge(*declared, access),
            None => self.declared.push((id, access)),
        }
        Ok(())
    }

    /// `memoryBarrierWithScope:` and `memoryBarrierWithResources:count:`: one global memory
    /// barrier either way, before the next dispatch.
    pub fn memory_barrier(&mut self) {
        self.building().owe();
    }

    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    ///
    /// Err: [`Refusal::NoPipeline`]; [`Refusal::ThreadgroupSize`]; [`Limit::ThreadgroupShape`];
    /// [`Limit::GroupCount`] for a count above the device's `maxComputeWorkGroupCount`;
    /// [`Error::Vulkan`] where the pipeline's compile, the ring or recording fails.
    pub fn dispatch_threadgroups(
        &mut self,
        threadgroups_per_grid: [u32; 3],
        threads_per_threadgroup: [u32; 3],
    ) -> Result<(), Error> {
        let grid = threadgroups_per_grid
            .iter()
            .zip(threads_per_threadgroup)
            .map(|(&groups, size)| groups.saturating_mul(size));
        let grid: Vec<u32> = grid.collect();
        self.dispatch(
            Grid::Direct(threadgroups_per_grid, [grid[0], grid[1], grid[2]]),
            threads_per_threadgroup,
        )
    }

    /// `dispatchThreads:threadsPerThreadgroup:`: the grid in whole threadgroups, with the grid's
    /// extent a parameter of the kernel.
    ///
    /// Err: as [`dispatch_threadgroups`](Self::dispatch_threadgroups);
    /// [`Gap::PartialThreadgroups`] for a grid that is not a whole number of threadgroups on some
    /// axis.
    pub fn dispatch_threads(
        &mut self,
        threads_per_grid: [u32; 3],
        threads_per_threadgroup: [u32; 3],
    ) -> Result<(), Error> {
        let mut groups = [0; 3];
        for axis in 0..3 {
            groups[axis] = match threads_per_threadgroup[axis] {
                0 => 0,
                size if threads_per_grid[axis] % size != 0 => {
                    return Err(Error::Unsupported(Gap::PartialThreadgroups {
                        threads_per_grid,
                        threads_per_threadgroup,
                    }))
                }
                size => threads_per_grid[axis] / size,
            };
        }
        self.dispatch(
            Grid::Direct(groups, threads_per_grid),
            threads_per_threadgroup,
        )
    }

    /// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`:
    /// `vkCmdDispatchIndirect` on the buffer, which counts as a read. The GPU reads the counts;
    /// the runtime never does, so a count above the device's limit is not refused (a known gap).
    ///
    /// Err: as [`dispatch_threadgroups`](Self::dispatch_threadgroups); [`Refusal::OtherDevice`];
    /// [`Refusal::IndirectOffset`] for an offset not a multiple of 4 or without 12 bytes after
    /// it in the buffer.
    pub fn dispatch_threadgroups_with_indirect_buffer(
        &mut self,
        buffer: &Buffer,
        offset: u64,
        threads_per_threadgroup: [u32; 3],
    ) -> Result<(), Error> {
        if !buffer.is_of(self.building().device.core()) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        if offset % 4 != 0
            || offset
                .checked_add(12)
                .is_none_or(|end| end > buffer.length())
        {
            return Err(Error::Refused(Refusal::IndirectOffset {
                offset,
                length: buffer.length(),
            }));
        }
        self.building().keep(buffer.held_block());
        self.dispatch(Grid::Indirect(buffer, offset), threads_per_threadgroup)
    }

    /// `endEncoding`: an encoder that wrote anything owes one barrier, which the next command or
    /// the segment's end records.
    pub fn end_encoding(mut self) {
        self.end();
    }

    fn end(&mut self) {
        if let Some(mut building) = self.building.take() {
            if self.wrote {
                building.owe();
            }
            self.command_buffer.close_encoder(building);
        }
    }

    /// The slot of buffer index `index`.
    fn slot(&self, index: u32) -> Result<usize, Error> {
        if index < BUFFER_SLOTS {
            Ok(index as usize)
        } else {
            Err(Error::Refused(Refusal::BufferIndex { index }))
        }
    }

    fn dispatch(&mut self, grid: Grid<'_>, size: [u32; 3]) -> Result<(), Error> {
        let pipeline = self
            .pipeline
            .clone()
            .ok_or(Error::Refused(Refusal::NoPipeline))?;
        let interface = &pipeline.inner.interface;
        let total = size.iter().map(|&axis| u64::from(axis)).product::<u64>();
        if total == 0 || total > u64::from(interface.max_total_threads) {
            return Err(Error::Refused(Refusal::ThreadgroupSize {
                size,
                max: interface.max_total_threads,
            }));
        }
        let core = Arc::clone(self.building().device.core());
        let limits = &core.limits;
        if size
            .iter()
            .zip(limits.max_group_size)
            .any(|(&axis, max)| axis > max)
        {
            return Err(Error::Limit(Limit::ThreadgroupShape {
                size,
                max: limits.max_group_size,
            }));
        }
        if let Grid::Direct(groups, _) = grid {
            if groups
                .iter()
                .zip(limits.max_group_count)
                .any(|(&count, max)| count > max)
            {
                return Err(Error::Limit(Limit::GroupCount {
                    count: groups,
                    max: limits.max_group_count,
                }));
            }
            // No thread runs: nothing is recorded, so a zero grid stays the kernel's mark of an
            // indirect dispatch.
            if groups.contains(&0) {
                return Ok(());
            }
        }
        // Only a kernel that reads its grid from the dispatch region tells an indirect dispatch
        // apart.
        let indirect = matches!(grid, Grid::Indirect(..))
            && interface
                .constants
                .iter()
                .any(|(_, set)| matches!(set, Constant::Region));
        let vk_pipeline = match self.last {
            Some((last, last_indirect, made)) if last == size && last_indirect == indirect => made,
            _ => {
                let variant = Variant {
                    size,
                    threadgroup_lengths: interface
                        .threadgroups
                        .iter()
                        .map(|&index| self.threadgroup_length(index))
                        .collect(),
                    indirect,
                };
                pipeline.inner.pipeline(&variant)?
            }
        };
        self.last = Some((size, indirect, vk_pipeline));
        let serial = self.serial;
        let wrote = {
            let accesses = Accesses {
                slots: &self.slots,
                declared: &self.declared,
                indirect: match grid {
                    Grid::Indirect(buffer, _) => Some(buffer.id()),
                    Grid::Direct(..) => None,
                },
            };
            let building = self
                .building
                .as_mut()
                .expect("an encoder holds its command buffer's recording until it ends");
            building.dispatch_hazards(serial, interface, &accesses)?
        };
        self.wrote |= wrote;
        self.fill_parameters(&pipeline, &grid, size);
        let building = self
            .building
            .as_mut()
            .expect("an encoder holds its command buffer's recording until it ends");
        building.record_dispatch(vk_pipeline, &mut self.parameters, &grid)
    }

    /// The length set for threadgroup memory index `index`, 0 where none was.
    fn threadgroup_length(&self, index: u32) -> u32 {
        self.threadgroup_lengths
            .iter()
            .find(|(set, _)| *set == index)
            .map_or(0, |(_, length)| *length)
    }

    /// Fills the push window and the ring block of a dispatch of `pipeline` over `grid` in
    /// threadgroups of `size`, each axis 1 to 1,024. The block's address is written by
    /// [`Building::record_dispatch`] once the block is in the ring.
    fn fill_parameters(
        &mut self,
        pipeline: &ComputePipelineState,
        grid: &Grid<'_>,
        size: [u32; 3],
    ) {
        let interface = &pipeline.inner.interface;
        let parameters = &mut self.parameters;
        parameters.push.clear();
        parameters
            .push
            .resize(interface.push_len.next_multiple_of(4) as usize, 0);
        parameters.block.clear();
        parameters.block.resize(interface.block_len as usize, 0);
        for field in &interface.fields {
            let (bytes, at) = match field.place {
                Place::Push(offset) => (&mut parameters.push, offset as usize),
                Place::Block(offset) => (&mut parameters.block, offset as usize),
            };
            match field.value {
                FieldValue::BufferAddress(slot) => {
                    let address = match self.slots.get(slot as usize) {
                        Some(Slot::Buffer { buffer, offset }) => {
                            buffer.gpu_address().wrapping_add(*offset)
                        }
                        Some(Slot::Bytes(address)) => *address,
                        Some(Slot::Empty) | None => 0,
                    };
                    bytes[at..at + 8].copy_from_slice(&address.to_le_bytes());
                }
                FieldValue::DispatchRegion => {
                    let threads = match grid {
                        Grid::Direct(_, threads) => *threads,
                        Grid::Indirect(..) => [0; 3],
                    };
                    let packed = (size[0] - 1) | (size[1] - 1) << 10 | (size[2] - 1) << 20;
                    let words = [0, 0, 0, packed, threads[0], threads[1], threads[2], 0];
                    for (index, word) in words.iter().enumerate() {
                        let at = at + index * 4;
                        bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                    }
                }
            }
        }
        parameters.block_address = (interface.block_len > 0).then_some(interface.block_address);
    }
}

/// A dispatch's parameters: the push window's bytes, and the ring block's with where its address
/// goes in the window.
#[derive(Default)]
struct Parameters {
    push: Vec<u8>,
    block: Vec<u8>,
    block_address: Option<u32>,
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        self.end();
    }
}

/// The union of two declared accesses.
fn merge(a: Access, b: Access) -> Access {
    match (a.reads() || b.reads(), a.writes() || b.writes()) {
        (true, true) => Access::ReadWrite,
        (false, true) => Access::Write,
        _ => Access::Read,
    }
}

/// What a dispatch touches besides its pipeline's own facts.
pub(crate) struct Accesses<'a> {
    slots: &'a [Slot],
    declared: &'a [(u32, Access)],
    indirect: Option<u32>,
}

impl Accesses<'_> {
    /// Each resource the dispatch touches, by id, with its access.
    pub(crate) fn each<'b>(
        &'b self,
        bound: &'b [(u32, Access)],
    ) -> impl Iterator<Item = (u32, Access)> + 'b {
        let slots =
            bound
                .iter()
                .filter_map(|&(slot, access)| match self.slots.get(slot as usize) {
                    Some(Slot::Buffer { buffer, .. }) => Some((buffer.id(), access)),
                    _ => None,
                });
        slots
            .chain(self.declared.iter().copied())
            .chain(self.indirect.map(|id| (id, Access::Read)))
    }
}

impl Building {
    /// Places a dispatch's barrier where it needs one, and adds what it touches to the sets.
    /// True where the dispatch writes.
    fn dispatch_hazards(
        &mut self,
        serial: bool,
        interface: &crate::pipeline::Interface,
        accesses: &Accesses<'_>,
    ) -> Result<bool, Error> {
        let command = self.command()?;
        let core = Arc::clone(self.device.core());
        let hazards = self.hazards();
        if serial {
            let conflict = accesses
                .each(&interface.buffers)
                .any(|(id, access)| hazards.conflicts(id, access))
                || (interface.reads_unbound && hazards.wrote())
                || (interface.writes_unbound && hazards.touched());
            if conflict {
                record::barrier(&core.device, command, Reach::Commands);
                core.counters.count(Metric::Barriers);
                hazards.clear();
            }
        }
        let mut wrote = false;
        for (id, access) in accesses.each(&interface.buffers) {
            hazards.mark(id, access);
            wrote |= access.writes();
        }
        if interface.reads_unbound {
            hazards.mark_wild_read();
        }
        if interface.writes_unbound {
            hazards.mark_wild_write();
            wrote = true;
            if serial {
                // Such a write conflicts with everything after it.
                self.owe();
            }
        }
        Ok(wrote)
    }

    /// Binds `pipeline` where it is not bound, writes the parameters' ring block and its address,
    /// pushes the window, and records the dispatch.
    fn record_dispatch(
        &mut self,
        pipeline: vk::Pipeline,
        parameters: &mut Parameters,
        grid: &Grid<'_>,
    ) -> Result<(), Error> {
        let core = Arc::clone(self.device.core());
        let device = &core.device;
        if let Some(at) = parameters.block_address {
            let address = self.write_ring(&parameters.block)?;
            let at = at as usize;
            parameters.push[at..at + 8].copy_from_slice(&address.to_le_bytes());
        }
        let push = &parameters.push[..];
        debug_assert!(push.len() <= PUSH_BYTES as usize);
        let command = self.command()?;
        // SAFETY: the command buffer is recording outside any render pass; the pipeline and the
        // layout are the device's, and the push range of the layout spans every stage's first 128
        // bytes.
        unsafe {
            if self.bind(pipeline) {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            }
            if !push.is_empty() {
                device.cmd_push_constants(command, core.layout, vk::ShaderStageFlags::ALL, 0, push);
            }
            match *grid {
                Grid::Direct(groups, _) => {
                    device.cmd_dispatch(command, groups[0], groups[1], groups[2]);
                }
                Grid::Indirect(buffer, offset) => {
                    let block = &buffer.held_block().memory;
                    device.cmd_dispatch_indirect(command, block.buffer, buffer.offset() + offset);
                }
            }
        }
        Ok(())
    }
}
