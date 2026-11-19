//! Compute command encoders: their binding state, their dispatches, and where barriers go.

use crate::buffer::Buffer;
use crate::command::{Building, CommandBuffer};
use crate::counters::Metric;
use crate::device::Core;
use crate::error::{Error, Gap, Limit, Refusal};
use crate::fence::Fence;
use crate::floor;
use crate::hazard::Access;
use crate::kept::{Handle, Lent, Uncounted};
use crate::memory::BlockId;
use crate::over;
use crate::pipeline::{
    address_word, ComputePipelineState, FieldValue, Form, Interface, Kernel, Place, Which,
    PUSH_BYTES, PUSH_STAGES, SQUARE,
};
use crate::record::Reach;
use crate::ring::CHUNK;
use crate::sampler::SamplerState;
use crate::texture::{Region, Texture};
use ash::vk;
use metal2vulkan::metal::format::integer_channels_word;
use metal2vulkan::pipeline::{
    grid_tile, nonuniform_region_bytes, region_bytes, DEEP_THREADGROUP, GRID_TILE_THREADS,
};
use std::ops::BitOr;
use std::sync::Arc;

/// Metal's count of buffer argument table entries per stage.
pub(crate) const BUFFER_SLOTS: u32 = 31;
/// Metal's count of texture argument table entries per stage.
pub(crate) const TEXTURE_SLOTS: u32 = 128;
/// Metal's count of sampler argument table entries per stage.
pub(crate) const SAMPLER_SLOTS: u32 = 16;

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
        buffer: Lent<Buffer>,
        offset: u64,
    },
    /// The address of a `set_bytes` copy in the parameter ring.
    Bytes(u64),
}

/// [`Refusal::BufferOffsetAlignment`] where `buffer`, bound at `offset` to buffer argument
/// `slot`, is at an address (its `gpuAddress` plus the offset) that is not a multiple of the
/// alignment the module takes the argument to have, `mask + 1` (`FieldValue::BufferAddress`'s
/// mask). A dispatch and a draw check each buffer their functions reach before they record
/// anything.
pub(crate) fn check_alignment(
    slot: u32,
    mask: u64,
    buffer: &Buffer,
    offset: u64,
) -> Result<(), Error> {
    if buffer.gpu_address().wrapping_add(offset) & mask != 0 {
        return Err(Error::Refused(Refusal::BufferOffsetAlignment {
            index: slot,
            offset,
            alignment: mask + 1,
        }));
    }
    Ok(())
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
    /// The device, borrowed from the recording's own handle in `building` with no count of its
    /// own: every use is while the encoder is open, and so while `building` holds that handle.
    core: Uncounted<Core>,
    /// What the command buffer records, held while the encoder is open.
    building: Option<Building>,
    serial: bool,
    /// The state set, and the objects of the argument tables below, borrowed from what the
    /// command buffer keeps (`crate::kept`).
    pipeline: Option<Lent<ComputePipelineState>>,
    slots: Vec<Slot>,
    /// The texture argument table.
    textures: Vec<Option<Lent<Texture>>>,
    /// The sampler argument table.
    samplers: Vec<Option<Lent<SamplerState>>>,
    /// Threadgroup memory lengths by index, as set.
    threadgroup_lengths: Vec<(u32, u32)>,
    /// The lengths of the last variant looked up, in its kernel's order, kept to reuse.
    lengths: Vec<u32>,
    /// `use_resource` declarations, by resource id: its own and its id in the hazard sets.
    declared: Vec<([u32; 2], Access)>,
    /// A `use_resource` declaration writes a Shared buffer.
    declared_shared_write: bool,
    /// The module, workgroup size and `VkPipeline` of the last dispatch, while the state and the
    /// threadgroup memory lengths are unchanged.
    last: Option<Last>,
    /// The parameters being filled, kept to reuse their bytes.
    parameters: Parameters,
    /// Whether a dispatch of the encoder wrote anything.
    wrote: bool,
    /// What a dispatch of the pipeline state reaches of the bound resources.
    facts: DispatchFacts,
    /// The stage-in region set ([`ComputeCommandEncoder::set_stage_in_region`]), as the bytes of
    /// its push field: the origin and the size, each a `uvec4` with a zero `w`.
    stage_in_region: Option<[u8; 32]>,
}

/// What a dispatch of the encoder's pipeline state reaches of its bound resources and
/// declarations, worked out at the first dispatch after either changes and kept until the next
/// change: whether its buffers are aligned as its kernel takes them, whether it reaches a Managed
/// buffer's device copy or writes a texture over one, and whether it writes memory the host may
/// read. A dispatch after no change runs none of the scans.
///
/// A change is the encoder's own call: a pipeline state set, or a buffer, an offset, bytes or a
/// texture bound, or a resource declared. Nothing outside the encoder changes what a fact holds:
/// each is of the objects bound (a buffer's device copy and storage, a texture's storage) and of
/// the kernel, which do not change. What can change outside it between two dispatches, a Managed
/// buffer's textures over its device copy and the textures over a buffer in images of their own
/// that live, each dispatch looks at again where a fact says it may matter.
#[derive(Default)]
struct DispatchFacts {
    /// The bound buffers are aligned as the state's kernels take them.
    aligned: bool,
    /// The module whose kernel the facts below are of, while they hold.
    module: Option<Module>,
    /// A buffer the kernel reaches has a device copy, or a texture it writes is over one.
    managed: bool,
    /// A write the host may read once the command buffer completes: to a Shared buffer or a
    /// texture over one, bound or declared, or through an address outside the dispatch's
    /// resources.
    host: bool,
    /// The nil buffer's address where a buffer argument the kernels reach is bound to nothing
    /// (`Ring::nil`), else 0, found with the alignments.
    nil: u64,
}

/// Which of its state's modules a dispatch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Module {
    /// The state's whole module.
    Whole,
    /// The whole module of a kernel that runs its grid in tiles, at `MultiRegion` 1.
    Tiles,
    /// An observing kernel's partial-threadgroups variant.
    Partial,
    /// The emulated-SIMD-group variant.
    Emulated,
}

/// The last dispatch's module and workgroup size, and its `VkPipeline`.
#[derive(Clone, Copy)]
struct Last {
    module: Module,
    local: [u32; 3],
    pipeline: vk::Pipeline,
}

/// A direct dispatch's grid: Metal's threadgroups, its threads, and whether the last threadgroup
/// along some axis is partial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Direct {
    pub(crate) groups: [u32; 3],
    pub(crate) threads: [u32; 3],
    pub(crate) ragged: bool,
}

/// What runs a dispatch ([`choose`]): which of the state's modules, its workgroup size, the
/// threadgroup size its regions carry, and for a direct dispatch Metal's grid, the workgroups, and whether the last along some axis is partial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Choice {
    pub(crate) module: Module,
    pub(crate) local: [u32; 3],
    pub(crate) nominal: [u32; 3],
    pub(crate) direct: Option<([u32; 3], [u32; 3], bool)>,
}

/// How a state of `form` runs a dispatch of threadgroup size `size`, direct over `direct` or
/// indirect, the same on every device:
/// - indirect: the whole module at its own size;
/// - an image-only kernel ([`Form::Square`]): the whole module at `MultiRegion` 0 in workgroups
///   of [`SQUARE`] over a grid one deep, and of its [`image_tile`] over a deeper one, the region
///   carrying them, flagged where they do not divide the grid;
/// - a tiling kernel ([`Form::Tiled`]): its tiles, workgroups of `GRID_TILE_THREADS`, each one
///   `grid_tile` tile of the grid, which its region carries;
/// - an observing kernel ([`Form::Sized`]): its own size, a ragged grid on its
///   partial-threadgroups variant.
///
/// A size deeper than `DEEP_THREADGROUP` runs as one row of its threads
/// ([`Constant::Deep`](crate::pipeline::Constant::Deep)), so every workgroup is within the
/// floor's `maxComputeWorkGroupSize`.
pub(crate) fn choose(form: &Form, direct: Option<Direct>, size: [u32; 3]) -> Choice {
    let Some(direct) = direct else {
        return Choice {
            module: Module::Whole,
            local: size,
            nominal: size,
            direct: None,
        };
    };
    let Direct {
        groups,
        threads,
        ragged,
    } = direct;
    let over = |shape: [u32; 3]| {
        let groups = std::array::from_fn(|axis| threads[axis].div_ceil(shape[axis]));
        let ragged = (0..3).any(|axis| !threads[axis].is_multiple_of(shape[axis]));
        Some((threads, groups, ragged))
    };
    let whole = |shape: [u32; 3]| Choice {
        module: Module::Whole,
        local: shape,
        nominal: shape,
        direct: over(shape),
    };
    match form {
        Form::Square if threads[2] == 1 => whole(SQUARE),
        Form::Square => whole(image_tile(threads, size)),
        Form::Tiled => {
            let tile = grid_tile(threads);
            Choice {
                module: Module::Tiles,
                local: [GRID_TILE_THREADS, 1, 1],
                nominal: tile,
                direct: over(tile),
            }
        }
        Form::Sized { .. } => Choice {
            module: match ragged {
                false => Module::Whole,
                true => Module::Partial,
            },
            local: size,
            nominal: size,
            direct: Some((threads, groups, ragged)),
        },
    }
}

/// The tile an image-only kernel's grid of `threads`, more than one deep, runs in at threadgroup
/// size `size` ([`Form::Square`]): sides `2^a, 2^b, 2^c` with `a + b + c = 8`, so that one
/// workgroup of [`GRID_TILE_THREADS`] is one tile. Of every such tile it is the first in one total
/// order: the fewest threads launched, `∏ ⌈threads_i / side_i⌉ · side_i`; then the fewest tiles
/// partly past the grid; then the depth nearest the threadgroup's, `|2^c − size[2]|` smallest,
/// the shallower of two as near; then the most square in `x` and `y`, `|a − b|` smallest; then
/// the wider. A pure function of the grid and the threadgroup size: threadgroups of 8x8x4 over a
/// grid that 8x8x4 tiles divide run in 8x8x4, and threadgroups one deep in 16x16x1 where 16x16x1
/// tiles divide the grid. Each tile is one of 45, a `VkPipeline` of the state's each.
pub(crate) fn image_tile(threads: [u32; 3], size: [u32; 3]) -> [u32; 3] {
    const BITS: u32 = GRID_TILE_THREADS.trailing_zeros();
    // The tiles of side `2^k` an extent of `n` takes, and those of them it fills.
    let counts = |n: u32, k: u32| ((u64::from(n) + (1 << k) - 1) >> k, u64::from(n) >> k);
    let depth = u64::from(size[2].max(1));
    let mut best = (
        [0; 3],
        (u128::MAX, u128::MAX, u64::MAX, u32::MAX, u32::MAX, u32::MAX),
    );
    for c in 0..=BITS {
        let (tz, wz) = counts(threads[2], c);
        let near = (1u64 << c).abs_diff(depth);
        for b in 0..=BITS - c {
            let a = BITS - c - b;
            let ((tx, wx), (ty, wy)) = (counts(threads[0], a), counts(threads[1], b));
            let tiles = u128::from(tx) * u128::from(ty) * u128::from(tz);
            let partly = tiles - u128::from(wx) * u128::from(wy) * u128::from(wz);
            let key = (tiles, partly, near, c, a.abs_diff(b), b);
            if key < best.1 {
                best = ([a, b, c], key);
            }
        }
    }
    best.0.map(|k| 1 << k)
}

// A workgroup one deep at most `DEEP_THREADGROUP`, or one row of at most Metal's 1,024
// threads, is within the floor's sizes.
const _: () = assert!(
    DEEP_THREADGROUP <= floor::WORKGROUP_SIZE[2]
        && floor::WORKGROUP_INVOCATIONS <= floor::WORKGROUP_SIZE[0]
);

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
        // SAFETY: the encoder holds `building`, whose device handle keeps the core, until it ends;
        // nothing uses the borrow after that.
        let core = unsafe { Uncounted::of(building.device.core()) };
        building.count(Metric::Passes);
        if dispatch_type == DispatchType::Concurrent {
            // Its dispatches take no barrier among themselves, so whatever earlier encoders read
            // or wrote is ordered before its first dispatch.
            building.owe_if_touched();
        }
        Ok(ComputeCommandEncoder {
            command_buffer: self.clone(),
            core,
            building: Some(building),
            serial: dispatch_type == DispatchType::Serial,
            pipeline: None,
            slots: vec![Slot::Empty; BUFFER_SLOTS as usize],
            textures: vec![None; TEXTURE_SLOTS as usize],
            samplers: vec![None; SAMPLER_SLOTS as usize],
            threadgroup_lengths: Vec::new(),
            lengths: Vec::new(),
            declared: Vec::new(),
            declared_shared_write: false,
            last: None,
            parameters: Parameters::default(),
            wrote: false,
            facts: DispatchFacts::default(),
            stage_in_region: None,
        })
    }
}

impl ComputeCommandEncoder {
    fn building(&mut self) -> &mut Building {
        recording(&mut self.building)
    }

    /// The pipeline state, a binding or a declaration changed: the next dispatch finds its facts
    /// again.
    fn changed(&mut self) {
        self.facts = DispatchFacts::default();
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
        // SAFETY: the encoder holds the borrow, and it ends before its command buffer commits.
        self.pipeline = Some(unsafe { self.building().lend_pipeline(state) });
        self.last = None;
        self.changed();
        Ok(())
    }

    /// `setBuffer:offset:atIndex:` (`setBuffers:offsets:withRange:` is a loop of it). `None`
    /// unbinds the slot. The command buffer keeps the buffer, and so its block, until it retires.
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
                if let Slot::Buffer {
                    buffer: held,
                    offset: at,
                } = &self.slots[slot]
                {
                    if held.address() == buffer.address() && *at == offset {
                        return Ok(());
                    }
                }
                let building = self.building();
                if !buffer.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                Slot::Buffer {
                    // SAFETY: the encoder's slot holds the borrow, and it ends before its command
                    // buffer commits.
                    buffer: unsafe { building.lend_buffer(buffer) },
                    offset,
                }
            }
            None => Slot::Empty,
        };
        self.slots[slot] = bound;
        self.changed();
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
        let held = &self.textures[index as usize];
        if held.as_deref().map(Handle::address) == texture.map(Handle::address) {
            return Ok(());
        }
        let lent = match texture {
            Some(texture) => {
                let building = self.building();
                if !texture.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                // SAFETY: the encoder's slot holds the borrow, and it ends before its command
                // buffer commits.
                Some(unsafe { building.lend_texture(texture) })
            }
            None => None,
        };
        self.textures[index as usize] = lent;
        self.changed();
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
        let held = &self.samplers[index as usize];
        if held.as_deref().map(Handle::address) == sampler.map(Handle::address) {
            return Ok(());
        }
        let lent = match sampler {
            Some(sampler) => {
                let building = self.building();
                if !sampler.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                // SAFETY: the encoder's slot holds the borrow, and it ends before its command
                // buffer commits.
                Some(unsafe { building.lend_sampler(sampler) })
            }
            None => None,
        };
        self.samplers[index as usize] = lent;
        Ok(())
    }

    /// `setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`
    /// (`setSamplerStates:lodMinClamps:lodMaxClamps:withRange:` is a loop of it): the state bound
    /// as [`set_sampler_state`](Self::set_sampler_state) binds it, where its samples read alike
    /// with its LOD clamps replaced by these: each bound, held to the levels a sample can read
    /// (`[0, 15]`; Metal's largest textures have 15 levels), is the state's own so held, as a
    /// clamp of 1000 over a state's `FLT_MAX` is.
    ///
    /// Err: [`Gap::LodClamps`] where they read otherwise (nothing is bound); as
    /// [`set_sampler_state`](Self::set_sampler_state).
    pub fn set_sampler_state_with_lod_clamps(
        &mut self,
        sampler: Option<&SamplerState>,
        lod_min_clamp: f32,
        lod_max_clamp: f32,
        index: u32,
    ) -> Result<(), Error> {
        crate::sampler::clamps_at_bind(sampler, lod_min_clamp, lod_max_clamp)?;
        self.set_sampler_state(sampler, index)
    }

    /// `setStageInRegion:`: the region a kernel's `[[grid_origin]]` and `[[grid_size]]` read,
    /// pushed at every later dispatch of the encoder as the origin and the size, each a `uvec4`
    /// with a zero `w`. A dispatch of a kernel that reads the region on an encoder that set none
    /// is [`Gap::NoStageInRegion`]. A kernel with a `[[stage_in]]` argument, which fetches by the
    /// region, is [`Gap::KernelInput`] at its state's creation.
    pub fn set_stage_in_region(&mut self, region: Region) {
        let mut bytes = [0; 32];
        let words =
            (region.origin.into_iter().chain([0])).chain(region.size.into_iter().chain([0]));
        for (word, at) in words.zip(bytes.as_chunks_mut::<4>().0) {
            *at = word.to_le_bytes();
        }
        self.stage_in_region = Some(bytes);
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
                self.changed();
                Ok(())
            }
            _ => Err(Error::Refused(Refusal::NoBufferAtIndex { index })),
        }
    }

    /// `setBytes:length:atIndex:`: the bytes are copied into the parameter ring now, as Metal
    /// copies them, and the slot's address is the copy's.
    ///
    /// Err: [`Refusal::BufferIndex`] for an index of 31 or more; [`Limit::BytesLength`] past one
    /// ring chunk (64 KiB; Metal documents the call for data under 4 KB);
    /// [`Limit::MemoryAllocations`] where the ring's next block would take the device past its
    /// allocation count; [`Error::Vulkan`] where the ring cannot grow.
    pub fn set_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        let slot = self.slot(index)?;
        if bytes.len() as u64 > CHUNK {
            return Err(Error::Limit(Limit::BytesLength {
                len: bytes.len() as u64,
                max: CHUNK,
            }));
        }
        let address = self.building().write_ring(bytes)?;
        // Bytes are aligned, of no Managed or Shared buffer: a dispatch's facts change only where
        // the slot held a buffer.
        if !matches!(self.slots[slot], Slot::Bytes(_)) {
            self.changed();
        }
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
    /// later dispatch's accesses. The command buffer keeps the buffer or the texture until it
    /// retires.
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
                building.keep(buffer);
                ([buffer.id(), buffer.tracked_id()], buffer.is_shared())
            }
            Resource::Texture(texture) => {
                if !texture.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                building.keep_texture(texture);
                // The host reads a texture's bytes through a copy that makes them visible to it
                // itself (`Texture::get_bytes`), but for a texture over a Shared buffer.
                (
                    [texture.hazard_id(), texture.tracked_id()],
                    texture.host_reads(),
                )
            }
        };
        let Some(access) = usage.access() else {
            return Ok(());
        };
        self.declared_shared_write |= access.writes() && shared;
        self.changed();
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

    /// `useHeap:` (`useHeaps:count:` is a loop of it), for the heap whose memory is block `heap`:
    /// a [`HeapRange`](crate::HeapRange)'s block, or the block a placement heap's buffers sit in.
    /// The command buffer keeps the block until it retires, as it keeps a buffer, and so its
    /// block, for [`use_resource`](Self::use_resource), so a heap the consumer gives back while the work
    /// runs stays valid under it. Metal states no retention for `useHeap:`; keeping is the safe
    /// reading. It declares no access: a dispatch touches its bound slots and the `use_resource`
    /// declarations, and a kernel that reaches a heap's resources through addresses outside its
    /// slots already orders against every write since the last barrier.
    ///
    /// Err: [`Error::NoSuchBlock`] for a number no handle names.
    pub fn use_heap(&mut self, heap: BlockId) -> Result<(), Error> {
        let building = self.building();
        let block = building.device.block(heap)?;
        building.keep_block(&block);
        Ok(())
    }

    /// `memoryBarrierWithScope:` and `memoryBarrierWithResources:count:`: one global memory
    /// barrier either way, before the next dispatch.
    pub fn memory_barrier(&mut self) {
        self.building().owe();
    }

    /// `updateFence:`: encoders that wait on `fence` after this one see what this encoder and the
    /// ones before it wrote. It records nothing ([`Fence`]): its barrier at the end already
    /// reaches them.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's fence; [`Gap::FenceOtherQueue`] for a
    /// fence another queue used first.
    pub fn update_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        fence.used_on(&self.core, self.command_buffer.queue())
    }

    /// `waitForFence:`: this encoder sees what the encoders up to `fence`'s last update wrote. It
    /// records nothing ([`Fence`]): the barrier before its first dispatch already orders them.
    ///
    /// Err: as [`update_fence`](Self::update_fence).
    pub fn wait_for_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        fence.used_on(&self.core, self.command_buffer.queue())
    }

    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    ///
    /// A count above the device's `maxComputeWorkGroupCount` is split into dispatches within it,
    /// each reading Metal's whole grid from its region.
    ///
    /// A kernel that writes no memory (`PipelineInterface::writes_memory`) has no effect a program
    /// can observe: once the threadgroup size and the bound buffers' alignment are checked, its
    /// dispatch, direct or indirect, records nothing, binds nothing and reads no indirect buffer
    /// ([`Metrics::dispatches_without_effects`](crate::Metrics::dispatches_without_effects)).
    ///
    /// A threadgroup that is not whole subgroups of a kernel whose tier runs only whole ones (32
    /// threads where each SIMD group is a 32-lane cluster of a 64-lane subgroup included) runs
    /// the kernel's emulated-SIMD-group module, waiting for the reaper's lowering of it where
    /// that is not done.
    ///
    /// Err: [`Refusal::NoPipeline`]; [`Refusal::ThreadgroupSize`];
    /// [`Refusal::BufferOffsetAlignment`] for a buffer the kernel reaches bound off its argument's
    /// alignment, which nothing records; [`Gap::SwizzledReadWrite`];
    /// [`Limit::ThreadgroupMemory`]; [`Gap::PartialSimdgroups`] for a threadgroup that is not
    /// whole SIMD groups of a kernel whose tier needs them and that has no emulated module;
    /// [`Error::Quiesced`] where the device was quiesced before that module was lowered;
    /// [`Limit::MemoryAllocations`] where the parameter ring needs a block past the device's
    /// allocation count; [`Error::Vulkan`] where the pipeline's compile, the ring or recording
    /// fails.
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
    /// it in the buffer; [`Refusal::ArgumentPlace`] for arguments at a byte of the buffer's
    /// block's allocation that is not a multiple of 4, as a buffer placed at such a byte has.
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
        let place = buffer.device_range().1 + offset;
        if !place.is_multiple_of(4) {
            return Err(Error::Refused(Refusal::ArgumentPlace {
                place,
                alignment: 4,
            }));
        }
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
        let Some(pipeline) = &self.pipeline else {
            return Err(Error::Refused(Refusal::NoPipeline));
        };
        let state = &pipeline.inner;
        let inner = &state.compiled;
        let max = inner.whole.interface.max_total_threads;
        let total = size.iter().map(|&axis| u64::from(axis)).product::<u64>();
        if total == 0 || total > u64::from(max) {
            return Err(Error::Refused(Refusal::ThreadgroupSize { size, max }));
        }
        // Every module of a state is the same function, with the same arguments and alignments,
        // so the whole module answers this check and the next, once after each change.
        if !self.facts.aligned {
            self.facts.nil = check_alignments(&self.core, &self.slots, &inner.whole.interface)?;
            for &slot in &inner.whole.interface.storage_reads {
                if matches!(self.textures.get(slot as usize), Some(Some(t)) if t.swizzled()) {
                    return Err(Error::Unsupported(Gap::SwizzledReadWrite { slot }));
                }
            }
            let interface = &inner.whole.interface;
            for &(slot, access) in &interface.textures {
                let stored = access != Access::Read || interface.storage_reads.contains(&slot);
                let unstored = match self.textures.get(slot as usize) {
                    Some(Some(texture)) if stored => texture.unstored(),
                    _ => None,
                };
                if let Some(pixel_format) = unstored {
                    return Err(Error::Unsupported(Gap::StorageElement {
                        slot,
                        pixel_format,
                    }));
                }
            }
            self.facts.aligned = true;
        }
        let core = &*self.core;
        let limits = &core.limits;
        // A kernel that writes no memory has no effect a program can observe: once checked as
        // Metal checks it, the dispatch records nothing, binds nothing and reads no indirect
        // buffer.
        if !inner.whole.interface.has_effects {
            recording(&mut self.building).count(Metric::DispatchesWithoutEffects);
            return Ok(());
        }
        // What a kernel reads of a stage-in region the encoder never set is not established.
        if inner.whole.interface.stage_in_region && self.stage_in_region.is_none() {
            return Err(Error::Unsupported(Gap::NoStageInRegion));
        }
        if let Grid::Indirect(buffer, _) = &grid {
            recording(&mut self.building).keep(buffer);
        }
        // What runs: the module, which of the state's it is, its workgroup size, the threadgroup
        // size its regions carry, and for a direct dispatch Metal's grid, the workgroups, and
        // whether the last along some axis is partial.
        let partial_threadgroups = |threads| {
            Error::Unsupported(Gap::PartialThreadgroups {
                threads_per_grid: threads,
                threads_per_threadgroup: size,
            })
        };
        let direct = match &grid {
            &Grid::Direct {
                groups,
                threads,
                ragged,
            } => Some(Direct {
                groups,
                threads,
                ragged,
            }),
            Grid::Indirect(..) => None,
        };
        let Choice {
            module,
            local,
            nominal,
            direct,
        } = choose(&inner.form, direct, size);
        let kernel = match module {
            Module::Partial => inner.later_kernel(Which::Partial, || {
                partial_threadgroups(direct.map_or([0; 3], |(threads, ..)| threads))
            })?,
            _ => &inner.whole,
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
            Some(&width) => (
                inner.later_kernel(Which::Emulated, || {
                    Error::Unsupported(Gap::PartialSimdgroups {
                        threads_per_threadgroup: size,
                        width,
                    })
                })?,
                Module::Emulated,
            ),
        };
        // No thread runs: nothing is recorded, so a zero grid stays the kernel's mark of an
        // indirect dispatch.
        if direct.is_some_and(|(_, groups, _)| groups.contains(&0)) {
            return Ok(());
        }
        let vk_pipeline = match self.last {
            Some(last) if last.module == module && last.local == local => last.pipeline,
            _ => {
                let lengths = &mut self.lengths;
                lengths.clear();
                lengths.extend(
                    (kernel.interface.threadgroups.iter())
                        .map(|&index| threadgroup_length(&self.threadgroup_lengths, index)),
                );
                let length = kernel.interface.memory.bytes(lengths, threads);
                if length > u64::from(limits.max_shared_memory) {
                    return Err(Error::Limit(Limit::ThreadgroupMemory {
                        length,
                        max: limits.max_shared_memory,
                    }));
                }
                kernel.pipeline(core, (local, lengths, module == Module::Tiles))?
            }
        };
        self.last = Some(Last {
            module,
            local,
            pipeline: vk_pipeline,
        });
        let interface = &kernel.interface;
        if self.facts.module != Some(module) {
            self.facts.managed = self.reaches_managed(interface);
            self.facts.host = self.writes_host(interface);
            self.facts.module = Some(module);
        }
        let wrote = {
            let accesses = Accesses {
                slots: &self.slots,
                textures: &self.textures,
                declared: &self.declared,
                indirect: match grid {
                    Grid::Indirect(buffer, _) => Some([buffer.id(), buffer.tracked_id()]),
                    Grid::Direct { .. } => None,
                },
            };
            let building = recording(&mut self.building);
            // Textures over a buffer in an image of their own take their copies first
            // (`crate::over`).
            let overs = (building.copies_over())
                .then(|| accesses.each(interface, false).collect::<Vec<_>>());
            if let Some(overs) = &overs {
                let unbound = interface.reads_unbound || interface.writes_unbound;
                building.overs_before(overs, unbound)?;
            }
            if self.facts.managed {
                // A binding of a Managed buffer with textures over its device copy takes their
                // copies back first, and one whose compiled interface writes it owes them their
                // copies in; a texture slot written owes its copy back
                // (`crate::over::ManagedOvers`).
                let mut backs = Vec::new();
                for &(slot, access) in &interface.buffers {
                    if let Some(Slot::Buffer { buffer, .. }) = self.slots.get(slot as usize) {
                        if buffer.managed_overs().is_some() {
                            backs.extend(over::backs(buffer, None, true));
                            if access.writes() {
                                building.wrote_managed(buffer);
                            }
                        }
                    }
                }
                building.managed_backs(backs)?;
                for &(slot, access) in &interface.textures {
                    if let (true, Some(Some(texture))) =
                        (access.writes(), self.textures.get(slot as usize))
                    {
                        building.wrote_texture_managed(texture);
                    }
                }
            }
            let wrote = building.dispatch_hazards(self.serial, interface, &accesses)?;
            match &overs {
                Some(overs) => building.overs_after(overs, interface.writes_unbound),
                None => building.overs_untracked(),
            }
            // Marked after the hazards' barrier, which may have split the recording at a signal:
            // the mark is the dispatch's own segment's.
            if self.facts.host {
                building.wrote_host();
            }
            wrote
        };
        self.wrote |= wrote;
        let region = |base, threads, ragged| {
            match ragged {
                true => nonuniform_region_bytes(base, threads, nominal),
                false => region_bytes(base, threads, nominal),
            }
            .ok_or(Error::Refused(Refusal::ThreadgroupSize { size, max }))
        };
        let ids = &state.numbered.ids;
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
                            let region = region(base, threads, ragged)?;
                            let bound = (&self.slots[..], &self.textures[..], &self.samplers[..]);
                            let regions = (&region, self.stage_in_region.as_ref());
                            self.parameters
                                .fill(kernel, (bound, self.facts.nil), ids, regions);
                            recording(&mut self.building).record_dispatch(
                                core,
                                vk_pipeline,
                                &mut self.parameters,
                                &Launch::Direct(count),
                            )?;
                        }
                    }
                }
                Ok(())
            }
            (None, Grid::Indirect(buffer, offset)) => {
                let region = region([0; 3], [0; 3], false)?;
                let bound = (&self.slots[..], &self.textures[..], &self.samplers[..]);
                let regions = (&region, self.stage_in_region.as_ref());
                self.parameters
                    .fill(kernel, (bound, self.facts.nil), ids, regions);
                recording(&mut self.building).record_dispatch(
                    core,
                    vk_pipeline,
                    &mut self.parameters,
                    &Launch::Indirect(buffer, offset),
                )
            }
            (None, Grid::Direct { .. }) => Ok(()),
        }
    }

    /// Whether a dispatch of a kernel of `interface` reaches a bound buffer that has a device
    /// copy, or writes a bound texture over one: the only dispatches that may owe a Managed
    /// buffer's textures over its device copy their copies.
    fn reaches_managed(&self, interface: &Interface) -> bool {
        (interface.buffers.iter()).any(|&(slot, _)| {
            matches!(
                self.slots.get(slot as usize),
                Some(Slot::Buffer { buffer, .. }) if buffer.held().1.is_some()
            )
        }) || (interface.textures.iter()).any(|&(slot, access)| {
            access.writes()
                && matches!(
                    self.textures.get(slot as usize),
                    Some(Some(texture)) if texture.managed_over().is_some()
                )
        })
    }

    /// Whether a dispatch of a kernel of `interface` writes what the host may read once the
    /// command buffer completes: a Shared buffer or a texture over one, bound or declared, or
    /// memory through an address outside the dispatch's resources.
    fn writes_host(&self, interface: &Interface) -> bool {
        interface.writes_unbound
            || self.declared_shared_write
            || (interface.buffers.iter()).any(|&(slot, access)| {
                access.writes()
                    && matches!(
                        self.slots.get(slot as usize),
                        Some(Slot::Buffer { buffer, .. }) if buffer.is_shared()
                    )
            })
            || (interface.textures.iter()).any(|&(slot, access)| {
                access.writes()
                    && matches!(
                        self.textures.get(slot as usize),
                        Some(Some(texture)) if texture.host_reads()
                    )
            })
    }
}

/// The length `set` gives threadgroup memory index `index`, 0 where it gives none.
fn threadgroup_length(set: &[(u32, u32)], index: u32) -> u32 {
    (set.iter())
        .find(|(at, _)| *at == index)
        .map_or(0, |(_, length)| *length)
}

/// The recording an open encoder holds.
fn recording(building: &mut Option<Building>) -> &mut Building {
    building
        .as_mut()
        .expect("an encoder holds its command buffer's recording until it ends")
}

/// [`check_alignment`] of each buffer bound in `slots` that `interface` reaches, and the nil
/// buffer's address where one it reaches is bound to nothing (`Ring::nil`), else 0. A `set_bytes`
/// copy is at a multiple of 16 in the ring, which every mask divides, and so is the nil buffer.
fn check_alignments(core: &Core, slots: &[Slot], interface: &Interface) -> Result<u64, Error> {
    let mut nil = 0;
    for field in &interface.fields {
        let FieldValue::BufferAddress { slot, mask } = field.value else {
            continue;
        };
        match slots.get(slot as usize) {
            Some(Slot::Buffer { buffer, offset }) => check_alignment(slot, mask, buffer, *offset)?,
            Some(Slot::Bytes(_)) => {}
            Some(Slot::Empty) | None => nil = core.ring.nil(core)?,
        }
    }
    Ok(nil)
}

/// A compute encoder's argument tables: buffer slots, textures and samplers.
type Bound<'a> = (
    &'a [Slot],
    &'a [Option<Lent<Texture>>],
    &'a [Option<Lent<SamplerState>>],
);

impl Parameters {
    /// Fills the push window and the ring block of one launch of `kernel` with `bound`, the
    /// encoder's argument tables, whose regions are `region`, the `DispatchRegion` push field's
    /// bytes, and `stage_in`, the `StageInRegion` field's where the encoder set a region (a
    /// kernel that reads it has one: the dispatch checked first), and whose state gave its
    /// numbered samplers `sampler_ids`. Only a kernel that reads the stage-in region copies it.
    /// The block's address is written by [`Building::record_dispatch`] once the block is in the
    /// ring. A buffer argument bound to nothing takes `nil`, the nil buffer's address.
    ///
    /// Every buffer address is a multiple of the alignment the module takes it to have, which the
    /// dispatch checked first ([`check_alignments`], which gave `nil`).
    fn fill(
        &mut self,
        kernel: &Kernel,
        (bound, nil): (Bound<'_>, u64),
        sampler_ids: &[u32],
        (region, stage_in): (&[u8; 32], Option<&[u8; 32]>),
    ) {
        let (slots, textures, samplers) = bound;
        let interface: &Interface = &kernel.interface;
        self.push.clear();
        self.push
            .resize(interface.push_len.next_multiple_of(4) as usize, 0);
        self.block.clear();
        self.block.resize(interface.block_len as usize, 0);
        for field in &interface.fields {
            let (bytes, at) = match field.place {
                Place::Push(offset) => (&mut self.push, offset as usize),
                Place::Block(offset) => (&mut self.block, offset as usize),
            };
            match field.value {
                FieldValue::BufferAddress { slot, .. } => {
                    let address = match slots.get(slot as usize) {
                        Some(Slot::Buffer { buffer, offset }) => {
                            buffer.gpu_address().wrapping_add(*offset)
                        }
                        Some(Slot::Bytes(address)) => *address,
                        Some(Slot::Empty) | None => nil,
                    };
                    let word = address_word(address);
                    bytes[at..at + 8].copy_from_slice(&word.to_le_bytes());
                }
                FieldValue::TextureId(slot) => {
                    let id = match textures.get(slot as usize) {
                        Some(Some(texture)) => texture.gpu_resource_id().to_bits(),
                        _ => 0,
                    };
                    bytes[at..at + 8].copy_from_slice(&id.to_le_bytes());
                }
                FieldValue::SamplerId(slot) => {
                    let id = match samplers.get(slot as usize) {
                        Some(Some(sampler)) => sampler.gpu_resource_id().to_bits(),
                        _ => 0,
                    };
                    bytes[at..at + 8].copy_from_slice(&id.to_le_bytes());
                }
                FieldValue::IntegerChannels(slot) => {
                    let word = match textures.get(slot as usize) {
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
                FieldValue::StageInRegion => {
                    if let Some(stage_in) = stage_in {
                        bytes[at..at + 32].copy_from_slice(stage_in);
                    }
                }
                // A compute content has none of these (`pipeline::interface`).
                FieldValue::Word(_)
                | FieldValue::TessellationFactors
                | FieldValue::AlphaTestReference => {}
            }
        }
        self.block_address = (interface.block_len > 0).then_some(interface.block_address);
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
    textures: &'a [Option<Lent<Texture>>],
    declared: &'a [([u32; 2], Access)],
    indirect: Option<[u32; 2]>,
}

impl Accesses<'_> {
    /// Each resource the dispatch of a kernel of `interface` touches, with its access: by its own
    /// id ([`Buffer::id`], [`Texture::hazard_id`]), which the copies over buffers name it by, or,
    /// where `tracked`, by its id in the hazard sets ([`Buffer::tracked_id`],
    /// [`Texture::tracked_id`]: a heap's resources take the heap's).
    pub(crate) fn each<'b>(
        &'b self,
        interface: &'b Interface,
        tracked: bool,
    ) -> impl Iterator<Item = (u32, Access)> + 'b {
        let at = usize::from(tracked);
        let slots = interface.buffers.iter().filter_map(move |&(slot, access)| {
            match self.slots.get(slot as usize) {
                Some(Slot::Buffer { buffer, .. }) => {
                    Some(([buffer.id(), buffer.tracked_id()][at], access))
                }
                _ => None,
            }
        });
        let textures = interface
            .textures
            .iter()
            .filter_map(move |&(slot, access)| {
                let texture = self.textures.get(slot as usize)?.as_ref()?;
                Some(([texture.hazard_id(), texture.tracked_id()][at], access))
            });
        slots
            .chain(textures)
            .chain(
                self.declared
                    .iter()
                    .map(move |&(ids, access)| (ids[at], access)),
            )
            .chain(self.indirect.map(|ids| (ids[at], Access::Read)))
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
                .each(interface, true)
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
        for (id, access) in accesses.each(interface, true) {
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
    /// pushes the window, and records the dispatch, on `core`, the recording's device.
    fn record_dispatch(
        &mut self,
        core: &Core,
        pipeline: vk::Pipeline,
        parameters: &mut Parameters,
        launch: &Launch<'_>,
    ) -> Result<(), Error> {
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
        // window lies in the layout's one push range, of `PUSH_STAGES`.
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
            if !push.is_empty() && self.push_needed(push) {
                device.cmd_push_constants(command, core.layout, PUSH_STAGES, 0, push);
            }
            match *launch {
                Launch::Direct(groups) => {
                    device.cmd_dispatch(command, groups[0], groups[1], groups[2]);
                }
                Launch::Indirect(buffer, offset) => {
                    let (reached, at) = buffer.device_range();
                    device.cmd_dispatch_indirect(command, reached, at + offset);
                }
            }
        }
        Ok(())
    }
}
