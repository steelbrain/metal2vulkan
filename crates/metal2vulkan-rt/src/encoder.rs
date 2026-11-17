//! Compute command encoders: their binding state, their dispatches, and where barriers go.

use crate::buffer::Buffer;
use crate::command::{Building, CommandBuffer};
use crate::counters::Metric;
use crate::error::{Error, Gap, Limit, Refusal};
use crate::hazard::Access;
use crate::pipeline::{
    tile, ComputePipelineState, FieldValue, Form, Interface, Kernel, Place, Variant, PUSH_BYTES,
};
use crate::record::Reach;
use crate::ring::CHUNK;
use crate::sampler::SamplerState;
use crate::texture::Texture;
use ash::vk;
use metal2vulkan::metal::format::integer_channels_word;
use metal2vulkan::pipeline::{nonuniform_region_bytes, region_bytes};
use std::ops::BitOr;
use std::sync::Arc;

/// Metal's count of buffer argument table entries per stage.
const BUFFER_SLOTS: u32 = 31;
/// Metal's count of texture argument table entries per stage.
const TEXTURE_SLOTS: u32 = 128;
/// Metal's count of sampler argument table entries per stage.
const SAMPLER_SLOTS: u32 = 16;

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
    /// An `MTLTexture`.
    Texture(&'a Texture),
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
    /// Threadgroups per grid at the call's threadgroup size, the grid in threads, and whether the
    /// last threadgroup along some axis is partial (a `dispatchThreads` grid that is not a whole
    /// number of threadgroups).
    Direct {
        groups: [u32; 3],
        threads: [u32; 3],
        ragged: bool,
    },
    /// The arguments' buffer and offset.
    Indirect(&'a Buffer, u64),
}

/// What one `vkCmdDispatch*` launches.
enum Launch<'a> {
    /// Workgroups.
    Direct([u32; 3]),
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
/// barrier at its end, which the next command, or the segment's own end, records. A barrier
/// between dispatches orders the dispatches after it after the dispatches before it, and reaches
/// nothing else, as nothing else in the command buffer touches memory; it orders indirect
/// argument reads only where the dispatch after it is indirect, and an indirect dispatch after a
/// write no barrier since has ordered before them gets one that does.
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
    /// The texture argument table.
    textures: Vec<Option<Texture>>,
    /// The sampler argument table.
    samplers: Vec<Option<SamplerState>>,
    /// Threadgroup memory lengths by index, as set.
    threadgroup_lengths: Vec<(u32, u32)>,
    /// `use_resource` declarations, by resource id.
    declared: Vec<(u32, Access)>,
    /// A `use_resource` declaration writes a Shared buffer.
    declared_shared_write: bool,
    /// The module, workgroup size and `VkPipeline` of the last dispatch, while the state and the
    /// threadgroup memory lengths are unchanged.
    last: Option<(Module, [u32; 3], vk::Pipeline)>,
    /// The parameters being filled, kept to reuse their bytes.
    parameters: Parameters,
    /// Whether a dispatch of the encoder wrote anything.
    wrote: bool,
}

/// Which of its state's modules a dispatch runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Module {
    /// The state's whole module.
    Whole,
    /// An observing kernel's partial-threadgroups variant.
    Partial,
    /// The emulated-SIMD-group variant.
    Emulated,
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
            textures: vec![None; TEXTURE_SLOTS as usize],
            samplers: vec![None; SAMPLER_SLOTS as usize],
            threadgroup_lengths: Vec::new(),
            declared: Vec::new(),
            declared_shared_write: false,
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

    /// `setTexture:atIndex:` (`setTextures:withRange:` is a loop of it). `None` unbinds the slot.
    /// A dispatch pushes the texture's `gpuResourceID` for each texture slot its kernel reads,
    /// and the channel widths an integer write saturates to where its kernel takes them. The
    /// command buffer keeps the texture until it retires.
    ///
    /// Err: [`Refusal::TextureArgumentIndex`] for an index of 128 or more;
    /// [`Refusal::OtherDevice`] for another device's texture.
    pub fn set_texture(&mut self, texture: Option<&Texture>, index: u32) -> Result<(), Error> {
        if index >= TEXTURE_SLOTS {
            return Err(Error::Refused(Refusal::TextureArgumentIndex { index }));
        }
        if let Some(texture) = texture {
            let building = self.building();
            if !texture.is_of(building.device.core()) {
                return Err(Error::Refused(Refusal::OtherDevice));
            }
            building.keep_texture(texture);
        }
        self.textures[index as usize] = texture.cloned();
        Ok(())
    }

    /// `setSamplerState:atIndex:` (`setSamplerStates:withRange:` is a loop of it). `None`
    /// unbinds the slot. A dispatch pushes the state's `gpuResourceID` for each sampler slot its
    /// kernel samples through. The command buffer keeps the state until it retires.
    ///
    /// Err: [`Refusal::SamplerArgumentIndex`] for an index of 16 or more;
    /// [`Refusal::OtherDevice`] for another device's state.
    pub fn set_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        if index >= SAMPLER_SLOTS {
            return Err(Error::Refused(Refusal::SamplerArgumentIndex { index }));
        }
        if let Some(sampler) = sampler {
            let building = self.building();
            if !sampler.is_of(building.device.core()) {
                return Err(Error::Refused(Refusal::OtherDevice));
            }
            building.keep_sampler(sampler);
        }
        self.samplers[index as usize] = sampler.cloned();
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
    /// later dispatch's accesses. The command buffer keeps a buffer's block, or a texture, until
    /// it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's resource.
    pub fn use_resource(
        &mut self,
        resource: Resource<'_>,
        usage: ResourceUsage,
    ) -> Result<(), Error> {
        let building = self.building();
        let (id, shared) = match resource {
            Resource::Buffer(buffer) => {
                if !buffer.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                building.keep(buffer.held_block());
                (buffer.id(), buffer.is_shared())
            }
            Resource::Texture(texture) => {
                if !texture.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                building.keep_texture(texture);
                // The host reads a texture's bytes only through a copy that makes them visible
                // to it itself (`Texture::get_bytes`).
                (texture.id(), false)
            }
        };
        let Some(access) = usage.access() else {
            return Ok(());
        };
        self.declared_shared_write |= access.writes() && shared;
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
    /// A count above the device's `maxComputeWorkGroupCount` is split into dispatches within it,
    /// each reading Metal's whole grid from its region.
    ///
    /// A threadgroup that is not whole subgroups of a kernel whose tier runs only whole ones (32
    /// threads where each SIMD group is a 32-lane cluster of a 64-lane subgroup included) runs
    /// the kernel's emulated-SIMD-group module, waiting for the reaper's lowering of it where
    /// that is not done.
    ///
    /// Err: [`Refusal::NoPipeline`]; [`Refusal::ThreadgroupSize`]; [`Limit::ThreadgroupShape`];
    /// [`Limit::ThreadgroupMemory`]; [`Gap::PartialSimdgroups`] for a threadgroup that is not
    /// whole SIMD groups of a kernel whose tier needs them and that has no emulated module;
    /// [`Error::Quiesced`] where the device was quiesced before that module was lowered;
    /// [`Error::Vulkan`] where the pipeline's compile, the ring or recording fails.
    pub fn dispatch_threadgroups(
        &mut self,
        threadgroups_per_grid: [u32; 3],
        threads_per_threadgroup: [u32; 3],
    ) -> Result<(), Error> {
        let threads = std::array::from_fn(|axis| {
            threadgroups_per_grid[axis].saturating_mul(threads_per_threadgroup[axis])
        });
        self.dispatch(
            Grid::Direct {
                groups: threadgroups_per_grid,
                threads,
                ragged: false,
            },
            threads_per_threadgroup,
        )
    }

    /// `dispatchThreads:threadsPerThreadgroup:`: the threadgroups the grid's threads fill, the
    /// last along an axis running only the threads left, with the grid's extent a parameter of
    /// the kernel. A kernel that observes its threadgroup partition runs a grid that is not a
    /// whole number of threadgroups on its partial-threadgroups variant.
    ///
    /// Err: as [`dispatch_threadgroups`](Self::dispatch_threadgroups);
    /// [`Gap::PartialThreadgroups`] for such a grid of a kernel whose partial-threadgroups
    /// variant the translator refused, or at a threadgroup that is not whole subgroups of a kernel
    /// whose tier runs only whole ones.
    pub fn dispatch_threads(
        &mut self,
        threads_per_grid: [u32; 3],
        threads_per_threadgroup: [u32; 3],
    ) -> Result<(), Error> {
        let groups = std::array::from_fn(|axis| {
            threads_per_grid[axis].div_ceil(threads_per_threadgroup[axis].max(1))
        });
        let ragged = (0..3).any(|axis| {
            threads_per_threadgroup[axis] != 0
                && !threads_per_grid[axis].is_multiple_of(threads_per_threadgroup[axis])
        });
        self.dispatch(
            Grid::Direct {
                groups,
                threads: threads_per_grid,
                ragged,
            },
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
        let inner = &pipeline.inner;
        let max = inner.whole.interface.max_total_threads;
        let total = size.iter().map(|&axis| u64::from(axis)).product::<u64>();
        if total == 0 || total > u64::from(max) {
            return Err(Error::Refused(Refusal::ThreadgroupSize { size, max }));
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
        // What runs: the module, which of the state's it is, its workgroup size, and for a
        // direct dispatch Metal's grid, the workgroups, and whether the last along some axis is
        // partial.
        let partial_threadgroups = |threads| {
            Error::Unsupported(Gap::PartialThreadgroups {
                threads_per_grid: threads,
                threads_per_threadgroup: size,
            })
        };
        let (kernel, module, local, direct) = match (&grid, &inner.form) {
            (Grid::Indirect(..), _) => (&inner.whole, Module::Whole, size, None),
            (&Grid::Direct { threads, .. }, Form::Tiled) => {
                let local = tile(threads);
                let ragged = (0..3).any(|axis| !threads[axis].is_multiple_of(local[axis]));
                let groups = std::array::from_fn(|axis| threads[axis].div_ceil(local[axis]));
                (
                    &inner.whole,
                    Module::Whole,
                    local,
                    Some((threads, groups, ragged)),
                )
            }
            (
                &Grid::Direct {
                    groups,
                    threads,
                    ragged: false,
                },
                Form::Sized { .. },
            ) => (
                &inner.whole,
                Module::Whole,
                size,
                Some((threads, groups, false)),
            ),
            (
                &Grid::Direct {
                    groups,
                    threads,
                    ragged: true,
                },
                Form::Sized { partial },
            ) => match partial {
                Some(kernel) => (
                    &**kernel,
                    Module::Partial,
                    size,
                    Some((threads, groups, true)),
                ),
                None => return Err(partial_threadgroups(threads)),
            },
        };
        // A threadgroup that is not whole groups of a module that runs only whole ones runs the
        // emulated module; the partial-threadgroups variant has none.
        let threads = local.iter().map(|&axis| u64::from(axis)).product::<u64>();
        let (kernel, module) = match kernel
            .interface
            .multiples
            .iter()
            .find(|&&multiple| !threads.is_multiple_of(u64::from(multiple.max(1))))
        {
            None => (kernel, module),
            Some(_) if module == Module::Partial => {
                return Err(partial_threadgroups(
                    direct.map_or([0; 3], |(threads, ..)| threads),
                ))
            }
            Some(&width) => (inner.emulated_kernel(size, width)?, Module::Emulated),
        };
        // No thread runs: nothing is recorded, so a zero grid stays the kernel's mark of an
        // indirect dispatch.
        if direct.is_some_and(|(_, groups, _)| groups.contains(&0)) {
            return Ok(());
        }
        let vk_pipeline = match self.last {
            Some((last, at, made)) if last == module && at == local => made,
            _ => {
                let variant = Variant {
                    size: local,
                    threadgroup_lengths: kernel
                        .interface
                        .threadgroups
                        .iter()
                        .map(|&index| self.threadgroup_length(index))
                        .collect(),
                };
                let length = kernel
                    .interface
                    .memory
                    .bytes(&variant.threadgroup_lengths, threads);
                if length > u64::from(limits.max_shared_memory) {
                    return Err(Error::Limit(Limit::ThreadgroupMemory {
                        length,
                        max: limits.max_shared_memory,
                    }));
                }
                kernel.pipeline(&core, &variant)?
            }
        };
        self.last = Some((module, local, vk_pipeline));
        let serial = self.serial;
        // A write the host may read once the command buffer completes: to a Shared buffer bound or
        // declared, or through an address outside the dispatch's resources.
        let host = kernel.interface.writes_unbound
            || self.declared_shared_write
            || (kernel.interface.buffers.iter()).any(|&(slot, access)| {
                access.writes()
                    && matches!(
                        self.slots.get(slot as usize),
                        Some(Slot::Buffer { buffer, .. }) if buffer.is_shared()
                    )
            });
        let wrote = {
            let accesses = Accesses {
                slots: &self.slots,
                textures: &self.textures,
                declared: &self.declared,
                indirect: match grid {
                    Grid::Indirect(buffer, _) => Some(buffer.id()),
                    Grid::Direct { .. } => None,
                },
            };
            let building = self
                .building
                .as_mut()
                .expect("an encoder holds its command buffer's recording until it ends");
            let wrote = building.dispatch_hazards(serial, &kernel.interface, &accesses)?;
            // Marked after the hazards' barrier, which may have split the recording at a signal:
            // the mark is the dispatch's own segment's.
            if host {
                building.wrote_host();
            }
            wrote
        };
        self.wrote |= wrote;
        let region = |base, threads, ragged| {
            match ragged {
                true => nonuniform_region_bytes(base, threads, local),
                false => region_bytes(base, threads, local),
            }
            .ok_or(Error::Refused(Refusal::ThreadgroupSize { size, max }))
        };
        match (direct, grid) {
            (Some((threads, groups, ragged)), _) => {
                // A count past the device's limit on an axis is split: each piece starts at its
                // first threadgroup in Metal's grid, which its region carries.
                let limit = limits.max_group_count;
                for z in (0..groups[2]).step_by(limit[2] as usize) {
                    for y in (0..groups[1]).step_by(limit[1] as usize) {
                        for x in (0..groups[0]).step_by(limit[0] as usize) {
                            let base = [x, y, z];
                            let count = std::array::from_fn(|axis| {
                                limit[axis].min(groups[axis] - base[axis])
                            });
                            self.fill_parameters(
                                kernel,
                                &inner.numbered.ids,
                                &region(base, threads, ragged)?,
                            );
                            self.record(vk_pipeline, Launch::Direct(count))?;
                        }
                    }
                }
                Ok(())
            }
            (None, Grid::Indirect(buffer, offset)) => {
                self.fill_parameters(kernel, &inner.numbered.ids, &region([0; 3], [0; 3], false)?);
                self.record(vk_pipeline, Launch::Indirect(buffer, offset))
            }
            (None, Grid::Direct { .. }) => Ok(()),
        }
    }

    /// Records one launch of the parameters filled last.
    fn record(&mut self, pipeline: vk::Pipeline, launch: Launch<'_>) -> Result<(), Error> {
        let building = self
            .building
            .as_mut()
            .expect("an encoder holds its command buffer's recording until it ends");
        building.record_dispatch(pipeline, &mut self.parameters, &launch)
    }

    /// The length set for threadgroup memory index `index`, 0 where none was.
    fn threadgroup_length(&self, index: u32) -> u32 {
        self.threadgroup_lengths
            .iter()
            .find(|(set, _)| *set == index)
            .map_or(0, |(_, length)| *length)
    }

    /// Fills the push window and the ring block of one launch of `kernel` whose region is
    /// `region`, the `DispatchRegion` push field's bytes, and whose state gave its numbered
    /// samplers `sampler_ids`. The block's address is written by [`Building::record_dispatch`]
    /// once the block is in the ring.
    fn fill_parameters(&mut self, kernel: &Kernel, sampler_ids: &[u32], region: &[u8; 32]) {
        let interface: &Interface = &kernel.interface;
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
                FieldValue::TextureId(slot) => {
                    let id = match self.textures.get(slot as usize) {
                        Some(Some(texture)) => texture.gpu_resource_id().to_bits(),
                        _ => 0,
                    };
                    bytes[at..at + 8].copy_from_slice(&id.to_le_bytes());
                }
                FieldValue::SamplerId(slot) => {
                    let id = match self.samplers.get(slot as usize) {
                        Some(Some(sampler)) => sampler.gpu_resource_id().to_bits(),
                        _ => 0,
                    };
                    bytes[at..at + 8].copy_from_slice(&id.to_le_bytes());
                }
                FieldValue::IntegerChannels(slot) => {
                    let word = match self.textures.get(slot as usize) {
                        Some(Some(texture)) => {
                            integer_channels_word(texture.descriptor().pixel_format)
                        }
                        _ => 0,
                    };
                    bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
                FieldValue::ConstexprSampler(place) => {
                    let index = sampler_ids.get(place).copied().unwrap_or(0);
                    bytes[at..at + 4].copy_from_slice(&index.to_le_bytes());
                }
                FieldValue::DispatchRegion => bytes[at..at + 32].copy_from_slice(region),
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
    textures: &'a [Option<Texture>],
    declared: &'a [(u32, Access)],
    indirect: Option<u32>,
}

impl Accesses<'_> {
    /// Each resource the dispatch of a kernel of `interface` touches, by id, with its access.
    pub(crate) fn each<'b>(
        &'b self,
        interface: &'b Interface,
    ) -> impl Iterator<Item = (u32, Access)> + 'b {
        let slots = interface.buffers.iter().filter_map(|&(slot, access)| {
            match self.slots.get(slot as usize) {
                Some(Slot::Buffer { buffer, .. }) => Some((buffer.id(), access)),
                _ => None,
            }
        });
        let textures = interface.textures.iter().filter_map(|&(slot, access)| {
            let texture = self.textures.get(slot as usize)?.as_ref()?;
            Some((texture.id(), access))
        });
        slots
            .chain(textures)
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
        interface: &Interface,
        accesses: &Accesses<'_>,
    ) -> Result<bool, Error> {
        let indirect = accesses.indirect.is_some();
        let reach = Reach::Dispatches { indirect };
        let command = self.command(reach)?;
        let hazards = self.hazards();
        let conflict = serial
            && (accesses
                .each(interface)
                .any(|(id, access)| hazards.conflicts(id, access))
                || (interface.reads_unbound && hazards.wrote())
                || (interface.writes_unbound && hazards.touched()));
        // In either kind of encoder: the barriers since a write may have left indirect argument
        // reads out.
        if conflict || (indirect && self.args_unordered()) {
            self.barrier(command, reach);
            self.hazards().clear();
        }
        let hazards = self.hazards();
        let mut wrote = false;
        for (id, access) in accesses.each(interface) {
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
        if wrote {
            self.wrote();
        }
        Ok(wrote)
    }

    /// Binds `pipeline` where it is not bound, writes the parameters' ring block and its address,
    /// pushes the window, and records the dispatch.
    fn record_dispatch(
        &mut self,
        pipeline: vk::Pipeline,
        parameters: &mut Parameters,
        launch: &Launch<'_>,
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
        let command = self.command(Reach::Dispatches {
            indirect: matches!(launch, Launch::Indirect(..)),
        })?;
        // Set 1, which every pipeline's layout shares, is bound once in each command buffer,
        // before its first pipeline; it stays bound across the pipelines after it.
        let first = self.unbound();
        // SAFETY: the command buffer is recording outside any render pass; the pipeline, the
        // layout and set 1 are the device's, set 1 of that layout and update-after-bind, and the
        // push range of the layout spans every stage's first 128 bytes.
        unsafe {
            if first {
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    core.layout,
                    1,
                    &[core.arrays.set()],
                    &[],
                );
            }
            if self.bind(pipeline) {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            }
            if !push.is_empty() {
                device.cmd_push_constants(command, core.layout, vk::ShaderStageFlags::ALL, 0, push);
            }
            match *launch {
                Launch::Direct(groups) => {
                    device.cmd_dispatch(command, groups[0], groups[1], groups[2]);
                }
                Launch::Indirect(buffer, offset) => {
                    let block = &buffer.held_block().memory;
                    device.cmd_dispatch_indirect(command, block.buffer, buffer.offset() + offset);
                }
            }
        }
        Ok(())
    }
}
