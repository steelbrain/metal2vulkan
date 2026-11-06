//! Compute planning: [`ComputePipeline::plan`].
//!
//! The plan is a pure function of the pipeline, the device it was compiled for, and the shape:
//!
//! - **Descriptors.** Every binding of the pipeline's layout gets one write: a bound buffer slot
//!   its `Buffer` (the range comes per dispatch) or `InlineBytes`, an unbound one the `ZeroBuffer`
//!   neutral, the stage's `ArgumentTable` UBO, and the pipeline's `ConstantBlob`. Textures,
//!   samplers, combined image samplers, and texel buffers go through the shared resource planner
//!   (`plan::resources`); the handle table and every element of the set-1 heap arrays up to the
//!   heap's members through the heap planner (`bind::heap`).
//! - **Device addresses.** Every `BufferForm::Psb` slot is reached through its root-table entry, so
//!   a buffer bound there without a device address, or `set*Bytes` data, is `MissingDeviceAddress`;
//!   an unbound one gets a zero entry (every access contained).
//! - **Specialization.** Threadgroup lengths (`TgLength`, Metal's 16-byte rounding applied) and
//!   slot presence are fixed here; the specialized threadgroup total
//!   `static_threadgroup_memory_length + Σ round_up(len, 16)` must fit
//!   `maxComputeSharedMemorySize`. With emulated simdgroups the static part is the arena's own
//!   (`RuntimeFact::SimdgroupEmulation`), and every region also fits the scratch of its simdgroups
//!   beside the declared arena. A predicated walk's vote word (`RuntimeFact::PredicationVote`) is
//!   counted beside the arena too. Local size and `MultiRegion` are per region.
//! - **Variant.** The shape's storage-image choices and device-address slots, and the emulated
//!   simdgroups of a pipeline compiled for them (`RuntimeFact::EmulatedSimdgroups`); a dispatch
//!   adds those where its threadgroups need them (`BindingPlan::dispatch_variant`). The flat grid
//!   of a pipeline compiled for one (`RuntimeFact::FlatGrid`), and the folded groups of one
//!   compiled for them (`RuntimeFact::FoldedGroups`).
//! - **Imageblocks**. A kernel with an explicit imageblock (`RuntimeFact::ImageblockSize`)
//!   specializes its extent from `BindState::imageblock_size`, which must be 1 to 65535 in each
//!   axis (`BindError::InvalidImageblockSize`); its `width · height · cell_bytes` bytes are one
//!   more 16-byte-aligned region of the threadgroup total, after the static part rounded to 16. A
//!   slot receiving imageblock slices takes any pixel format: a narrow element's texel is its lanes
//!   and zeros in every format, as Metal writes it.
//! - **Tile functions** (`RuntimeFact::TileFunction`). The render area (`BindState::render_area`,
//!   at least one pixel in each axis, `BindError::InvalidRenderArea` otherwise) is specialized, a
//!   layered pass is `Unsupported{TileShading, NotImplemented}`, and each attachment the function
//!   stages is a `WriteSource::TileAttachment` in the view format its staging reads
//!   (`lower::access::tile::staging_format`).
//! - **Tile threadgroup memory** (`RuntimeFact::TileThreadgroupMemory`). A tile function that reads
//!   its `[[threadgroup(n)]]` arguments specializes each argument's offset in its tile's memory
//!   (`SpecKey::TgOffset`, from `BindState::tile_threadgroup`), the tile's bytes
//!   (`SpecKey::TileThreadgroupLength`, the pass's length rounded up to 16) and the tiles across
//!   the render area (`SpecKey::TileGridWidth`), and binds the pass's tile memory
//!   (`WriteSource::TileMemory`, every tile's bytes). An argument the encoder did not set is
//!   `BindError::MissingThreadgroupMemory`. Every offset is placed, as plain Metal takes every
//!   offset, at the offset floored to a multiple of 4, which is where plain Metal begins the
//!   argument ([`tile_argument_offset`] carries the measurement). An argument's length bounds
//!   nothing, as plain Metal's does not. The tile's bytes start the threadgroup total.
//! - **Kept imageblock bytes** (`RuntimeFact::TileImageblock`). A tile function whose explicit
//!   imageblock stands alone, in a pass that keeps fewer bytes of its cells than its data
//!   (`BindShape::kept_imageblock_bytes`), asks for `VariantKey::kept_imageblock_bytes`: plain
//!   Metal reads the other words as zero and drops their writes.
//! - **Stage-in strides**. A layout of a fixed stride reads its own, whatever stride the shape
//!   binds its buffer with; one of `MTLBufferLayoutStrideDynamic` (`RuntimeFact::StageInStride`)
//!   reads the bound one, which instantiation pushes (`PushField::StageInStride`, 0 for a buffer
//!   bound without one).
//! - **Push fields.** Every placed field is one the binder fills: the root-table and status
//!   addresses (holes the consumer fills), `DispatchRegion` (always written), `StageInRegion` (the
//!   dispatch's `BindState::stage_in_region`), `StageInStride` (the bound buffer's attribute
//!   stride), `SlotBindOffset`, and `SlotGuestBase` (the bound buffer's Metal address; a slot bound
//!   with `set*Bytes` has none and is refused as `MissingGuestAddress`).

use super::resources::{separate_samplers, ResourcePlanner};
use super::slots::{
    address_map_variant, anisotropic_variant, bindings, bound_value, buffer_use, check_slots,
    check_sparse_textures, cross_variant, cubic_variant, mark_disjoint, plan_buffer, psb_variant,
    record_noted, record_specialized, srgb_variant, storage_view_variant, used_uses,
    ycbcr_in_place_variant, ycbcr_planes_sampled_variant,
};
use super::{
    BindingPlan, BoundKind, ComputeTemplate, DescriptorWrite, FieldLocation, LocalRules,
    PushFieldPlacement, PushTemplate, ResourceUse, ScratchRule, WriteSource,
};
use crate::bind::heap::{planner, HeapPlanner};
use crate::bind::neutral::{zero_buffer_bytes, NeutralResource};
use crate::bind::state::{BindShape, StageShape};
use crate::cache::{SpecializationKey, VariantKey};
use crate::device::requirements::VkLimit;
use crate::device::Device;
use crate::error::{
    BindError, Capability, Feature, InternalError, InternalKind, LowerFamily, Unsupported,
};
use crate::lower::simd::divergent::VOTE_BYTES;
use crate::lower::simd::workgroup::arena_bytes;
use crate::pipeline::artifact::{ComputePipeline, RuntimeDependency, SpecConstant};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::graphics::StorageAttachmentView;
use crate::pipeline::interface::{BufferForm, PushLoc, SlotInterface};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{PushField, SlotClass, SlotRef, SpecKey, Stage, SyntheticBinding};
use crate::vk;
use std::collections::BTreeMap;

/// Metal rounds every threadgroup-memory length to 16 bytes.
const TG_ALIGN: u64 = 16;

/// What plain Metal floors a tile threadgroup argument's offset to, in bytes
/// ([`tile_argument_offset`] carries the measurement).
const TG_FLOOR: u32 = 4;

pub(crate) fn internal(detail: impl Into<String>) -> BindError {
    BindError::Internal(InternalError::new(InternalKind::Invariant, detail))
}

pub(crate) fn not_implemented(feature: Feature) -> BindError {
    BindError::Unsupported(Unsupported::not_implemented(feature))
}

/// The largest imageblock extent in either axis: `get_imageblock_width` is a `ushort`.
const MAX_IMAGEBLOCK_EXTENT: u32 = u16::MAX as u32;

/// The bytes of the kernel's explicit imageblock for `shape`'s extent, rounded up to 16; `None`
/// without one. An extent of 0 or above 65535 in either axis is `BindError::InvalidImageblockSize`.
fn imageblock_bytes(p: &ComputePipeline, shape: &BindShape) -> Result<Option<u64>, BindError> {
    let Some(cell_bytes) = p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::ImageblockSize { cell_bytes } => Some(cell_bytes),
        _ => None,
    }) else {
        return Ok(None);
    };
    let s = shape.imageblock_size;
    let valid = |n: u32| (1..=MAX_IMAGEBLOCK_EXTENT).contains(&n);
    if !valid(s.width) || !valid(s.height) {
        return Err(BindError::InvalidImageblockSize {
            width: s.width,
            height: s.height,
        });
    }
    let bytes = u64::from(s.width) * u64::from(s.height) * u64::from(cell_bytes);
    Ok(Some(bytes.next_multiple_of(TG_ALIGN)))
}

/// A tile function's pass has a render area (`BindState::render_area` of at least one pixel in each
/// axis; `BindError::InvalidRenderArea` otherwise: the consumer never set it, and the function
/// would stage no pixel), and is not layered: what a tile dispatch of a layered pass runs, and what
/// `[[render_target_array_index]]` reads in it, is not read yet.
fn check_tile_pass(p: &ComputePipeline, shape: &BindShape) -> Result<(), BindError> {
    if !is_tile_function(p) {
        return Ok(());
    }
    let [width, height] = shape.render_area;
    if width == 0 || height == 0 {
        return Err(BindError::InvalidRenderArea { width, height });
    }
    if shape.render_target_array_length > 1 {
        return Err(not_implemented(Feature::TileShading));
    }
    Ok(())
}

/// A tile or fragment function's tile threadgroup memory as a bind shape places it
/// (`tile_memory`): each tile's bytes (the pass's `threadgroupMemoryLength` rounded up to 16),
/// the pass's tile, the tiles across and down the render area, and each argument's offset by
/// index.
pub(crate) struct TileMemoryPlan {
    pub(crate) tile_bytes: u64,
    pub(crate) tile: [u32; 2],
    pub(crate) grid: [u32; 2],
    pub(crate) offsets: BTreeMap<u32, u32>,
}

impl TileMemoryPlan {
    /// The buffer's bytes: every tile's, and 16 at least, as a buffer of no bytes cannot be bound
    /// (a pass of no threadgroup memory, whose arguments are all empty, reads none of them).
    pub(crate) fn bytes(&self) -> u64 {
        (u64::from(self.grid[0]) * u64::from(self.grid[1]) * self.tile_bytes).max(TG_ALIGN)
    }

    /// The value of tile memory specialization constant `key` (`SpecKey::TgOffset`,
    /// `TileThreadgroupLength` and `TileGridWidth`; a fragment function's tile, `ImageblockWidth`
    /// and `ImageblockHeight`), or `None` for another key or an argument without its fact.
    pub(crate) fn spec(&self, key: SpecKey) -> Option<u64> {
        match key {
            SpecKey::TgOffset(index) => self.offsets.get(&index).map(|&o| u64::from(o)),
            SpecKey::TileThreadgroupLength => Some(self.tile_bytes),
            SpecKey::TileGridWidth => Some(u64::from(self.grid[0])),
            SpecKey::ImageblockWidth => Some(u64::from(self.tile[0])),
            SpecKey::ImageblockHeight => Some(u64::from(self.tile[1])),
            _ => None,
        }
    }
}

/// Where an argument the encoder set at byte `offset` of its tile's threadgroup memory begins in
/// the threadgroup arena: `offset` floored to a multiple of 4, which is where plain Metal begins
/// it.
///
/// Plain Metal takes every offset and places it by this one rule, measured on an Apple M3 Max
/// under macOS 27.0.1 over eighteen offsets, each in its own process with Metal's API validation
/// off: with a tile memory of 128 bytes whose word `t` held a value only a seeding dispatch could
/// have written, an argument at byte 0, 2, 4, 8, 16, 17, 20 and 32 began at word 0, 0, 1, 2, 4, 4,
/// 5 and 8. So the floor is to 4 and not to 16, which byte 17 alone cannot tell apart (both place
/// it at byte 16) but bytes 4, 8 and 20 do.
///
/// Nothing is clamped. An offset at or past the tile's memory needs no clamp: every access from it
/// fails the arena's containment, which reads zero and drops the store (`memory::contain`), and
/// bytes 128 and 144 of that 128-byte memory read zero throughout. Nor may it be clamped, because
/// plain Metal's own address arithmetic wraps at 2^32 rather than reading zero from the memory's
/// end on for ever: an argument at `u64::MAX`, which is byte `0xfffffffc` after the truncation the
/// encoder applies, read zero and then the memory's words 0, 1 and 2 as its reads carried past
/// 2^32. The arena's offsets are `u32` and add with wrapping, so such accesses wrap where Metal's
/// do.
///
/// With containment off (`Containment::UnsafeOff`, never a default) a base past the arena is
/// outside its array, as the end of an argument that ends past the tile's memory already is: that
/// switch gives up Metal's answer for every out-of-range access, not this one alone.
///
/// A 4-byte-aligned base is exact here because the threadgroup arena has no 16-byte unit
/// (`memory::arena::ArenaMem` declares no `Units`), so a 16-byte access takes `Path::Vec4`'s
/// fallback to four words at `off / 4` rather than a `uvec4` index at `off / 16`
/// (`memory::leaf`): no access to the arena claims an alignment the base does not hold.
fn tile_argument_offset(offset: u32) -> u32 {
    offset - offset % TG_FLOOR
}

/// The tile threadgroup memory of a pipeline of `dependencies` as `shape` places it on `device`;
/// `None` for a pipeline that reads no threadgroup argument (`RuntimeFact::TileThreadgroupMemory`:
/// a tile function's, or a fragment function's). An argument the encoder did not set is
/// `BindError::MissingThreadgroupMemory`. An argument's length takes no part in the layout and
/// bounds no access, and every offset is placed where plain Metal begins the argument
/// ([`tile_argument_offset`]), as plain Metal takes every length and every offset. A shape without
/// a render area or a tile is `InvalidRenderArea` or `InvalidImageblockSize`. A buffer of every
/// tile's memory past the device's `maxStorageBufferRange`, which one descriptor binds at most, is
/// `Unsupported`.
pub(crate) fn tile_memory(
    dependencies: &[RuntimeDependency],
    device: &Device,
    shape: &BindShape,
) -> Result<Option<TileMemoryPlan>, BindError> {
    let mut offsets = BTreeMap::new();
    let memory = &shape.tile_threadgroup;
    for d in dependencies {
        let RuntimeFact::TileThreadgroupMemory { index } = d.fact else {
            continue;
        };
        let argument = memory
            .arguments
            .get(&index)
            .ok_or(BindError::MissingThreadgroupMemory { index })?;
        offsets.insert(index, tile_argument_offset(argument.offset));
    }
    if offsets.is_empty() {
        return Ok(None);
    }
    let [width, height] = shape.render_area;
    if width == 0 || height == 0 {
        return Err(BindError::InvalidRenderArea { width, height });
    }
    let tile = shape.imageblock_size;
    if tile.width == 0 || tile.height == 0 {
        return Err(BindError::InvalidImageblockSize {
            width: tile.width,
            height: tile.height,
        });
    }
    let plan = TileMemoryPlan {
        tile_bytes: u64::from(memory.length).next_multiple_of(TG_ALIGN),
        tile: [tile.width, tile.height],
        grid: [
            shape.render_area[0].div_ceil(tile.width),
            shape.render_area[1].div_ceil(tile.height),
        ],
        offsets,
    };
    let available = u64::from(device.profile().fields().limits.max_storage_buffer_range);
    if plan.bytes() > available {
        return Err(BindError::Unsupported(Unsupported::needs(
            Feature::TileShading,
            Capability::Limit {
                limit: VkLimit::MaxStorageBufferRange,
                required: plan.bytes(),
                available,
            },
        )));
    }
    Ok(Some(plan))
}

/// The bytes of each cell of `p`'s explicit imageblock standing alone
/// (`RuntimeFact::TileImageblock`) that `shape`'s pass keeps (`BindShape::kept_imageblock_bytes`),
/// where they are fewer than its data: `None` for every other pipeline and pass.
fn kept_imageblock_bytes(p: &ComputePipeline, shape: &BindShape) -> Option<u32> {
    let data_size = p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::TileImageblock { data_size, .. } => Some(data_size),
        _ => None,
    })?;
    shape.kept_imageblock_bytes.filter(|&kept| kept < data_size)
}

/// The threadgroup Metal answers `p`'s tile function in `shape`'s pass where its threadgroup is
/// the tile (`RuntimeFact::TileThreadgroup` with `tile_sized`): the tile, `shape`'s imageblock
/// size, at least 1 in each axis (`BindError::InvalidImageblockSize`). `None` for every other
/// pipeline.
fn tile_threadgroup(p: &ComputePipeline, shape: &BindShape) -> Result<Option<[u32; 3]>, BindError> {
    let tile_sized = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::TileThreadgroup { tile_sized: true });
    if !tile_sized {
        return Ok(None);
    }
    let tile = shape.imageblock_size;
    if tile.width == 0 || tile.height == 0 {
        return Err(BindError::InvalidImageblockSize {
            width: tile.width,
            height: tile.height,
        });
    }
    Ok(Some([tile.width, tile.height, 1]))
}

/// Whether `p` is a tile function's pipeline (`RuntimeFact::TileFunction`).
fn is_tile_function(p: &ComputePipeline) -> bool {
    p.dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::TileFunction)
}

/// The view format of tile attachment `color` of `p` (`WriteSource::TileAttachment`): the staging
/// format (`lower::access::tile::staging_format`) of the pixel format the tile function stages it
/// in (`RuntimeFact::TileAttachmentFormat`).
fn tile_view_format(p: &ComputePipeline, color: u32) -> Result<crate::vk::Format, BindError> {
    let pixel_format = p
        .dependencies
        .iter()
        .find_map(|d| match d.fact {
            RuntimeFact::TileAttachmentFormat { color: c, format } if c == color => Some(format),
            _ => None,
        })
        .ok_or_else(|| internal(format!("tile attachment {color} without its format")))?;
    let view = crate::lower::access::tile::staging_format(pixel_format);
    crate::metal::format::direct_format(view).ok_or_else(|| {
        internal(format!(
            "tile attachment {color}: no Vulkan format for {view:?}"
        ))
    })
}

impl ComputePipeline {
    /// The shape-level binding plan of this pipeline for `shape`: a pure function of the pipeline,
    /// `device` (the device that compiled it), and the shape, so the consumer memoizes it on
    /// `(PipelineKey, BindShape)`. Per-dispatch bytes come from
    /// [`BindingPlan::instantiate_dispatch`].
    pub fn plan(&self, device: &Device, shape: &BindShape) -> Result<BindingPlan, BindError> {
        plan(self, device, shape)
    }
}

/// The specialization data of the constants in `layout` that `value` gives a value, in `SpecId`
/// order: each value's low `size` bytes, little-endian, without the bits no stage reads
/// (`SpecConstant::unread_bits`), so values that differ only there give one key, one `VkPipeline`,
/// and one specialization the driver sees. Its two lists are made at once, at the first constant
/// given a value, at the size it and every later constant take; a key of no value makes none.
pub(crate) fn spec_key(
    layout: &[SpecConstant],
    value: impl Fn(SpecKey) -> Option<u64>,
) -> SpecializationKey {
    let mut key = SpecializationKey::new();
    for (at, c) in layout.iter().enumerate() {
        let Some(v) = value(c.key) else { continue };
        // Room for the rest of the layout at once, rather than growing a push at a time: a
        // dispatch's regions each make one.
        if key.entries.is_empty() {
            let rest = &layout[at..];
            key.entries.reserve_exact(rest.len());
            (key.data).reserve_exact(rest.iter().map(|c| c.size.min(8) as usize).sum());
        }
        let v = v & !c.unread_bits;
        let size = c.size.min(8);
        key.entries.push(vk::SpecializationMapEntry {
            constant_id: c.id,
            offset: key.data.len() as u32,
            size,
        });
        key.data
            .extend_from_slice(&v.to_le_bytes()[..size as usize]);
    }
    key
}

fn plan(p: &ComputePipeline, device: &Device, shape: &BindShape) -> Result<BindingPlan, BindError> {
    check_slots(shape)?;
    if p.stage.stage != vk::ShaderStage::COMPUTE {
        return Err(internal("a compute pipeline whose stage is not compute"));
    }
    let f = device.profile().fields();
    let limits = &f.limits;
    let empty = StageShape::default();
    let ks = shape.stages.get(&Stage::Kernel).unwrap_or(&empty);

    let bindings = bindings(&p.layout);

    // Buffer slots.
    let mut buffers = Vec::new();
    for (&slot, iface) in &p.interface.slots {
        let b = match iface {
            SlotInterface::Buffer(b) if slot.stage == Stage::Kernel => b,
            SlotInterface::Texture(_) | SlotInterface::Sampler(_)
                if slot.stage == Stage::Kernel =>
            {
                continue
            }
            SlotInterface::Buffer(_) | SlotInterface::Texture(_) | SlotInterface::Sampler(_) => {
                return Err(internal(format!("a compute pipeline uses {slot:?}")))
            }
        };
        buffers.push(plan_buffer(
            slot,
            b,
            ks.buffers.get(&slot.index),
            limits,
            &bindings,
        )?);
    }
    mark_disjoint(&mut buffers, &p.dependencies);
    let planned = |slot: SlotRef| buffers.iter().find(|b| b.slot == slot);

    // Push placement.
    let placement = p.interface.push.iter().find(|pl| pl.stage == Stage::Kernel);
    let fields = placement.map(|pl| pl.fields.clone()).unwrap_or_default();
    let mut push = PushTemplate::default();
    let mut reads_root_table = false;
    for field in &fields {
        let in_push = matches!(field.loc, PushLoc::Push { .. });
        match field.field {
            PushField::RootTableAddress => reads_root_table = true,
            PushField::ExecutionStatusAddress
            | PushField::DispatchRegion
            | PushField::ReplayRegionAddress
            | PushField::StageInRegion => {}
            PushField::SlotBindOffset(slot)
                if planned(slot).is_some_and(|b| b.binding.is_some()) => {}
            // The bound buffer's Metal address; a `set*Bytes` binding has none.
            PushField::SlotGuestBase(slot)
                if planned(slot).is_some_and(|b| b.bound != BoundKind::Bytes) => {}
            PushField::SlotGuestBase(slot) if planned(slot).is_some() => {
                return Err(BindError::MissingGuestAddress { slot })
            }
            PushField::SlotBindOffset(SlotRef {
                class: SlotClass::Texture,
                ..
            }) => return Err(not_implemented(Feature::TexelBuffer)),
            // A storage texture's level count and integer channel widths, which the resource plan
            // records; and a value the bound resources decide, which the plan records too
            // (`slots::record_specialized`).
            PushField::TextureLevels(_)
            | PushField::TextureIntegerChannels(_)
            | PushField::Specialized(_) => {}
            // A structure's instance count, which instantiation reads from the state.
            PushField::InstanceCount(_) => {}
            // An in-place YCbCr texture's words, which instantiation reads from the state.
            PushField::YCbCrInPlace(_) => {}
            // A stage-in buffer's bound stride, which instantiation reads from the state.
            PushField::StageInStride(_) => {}
            other => return Err(internal(format!("a compute push field {other:?}"))),
        }
        let per_region = matches!(
            field.field,
            PushField::RootTableAddress
                | PushField::ExecutionStatusAddress
                | PushField::DispatchRegion
                | PushField::ReplayRegionAddress
        );
        if per_region && !in_push {
            // The placement order puts these first, so a compute window (128 bytes) always holds
            // them.
            return Err(not_implemented(Feature::Lowering(LowerFamily::Dispatch)));
        }
        push.fields.push(PushFieldPlacement {
            stage: Stage::Kernel,
            field: field.field,
            location: match field.loc {
                PushLoc::Push { offset } => FieldLocation::PushWindow { offset },
                PushLoc::ArgumentTable { offset } => FieldLocation::ArgumentTable { offset },
            },
            size: field.size,
        });
    }
    let root_first = reads_root_table.then(|| {
        let top = buffers
            .iter()
            .filter(|b| b.form == BufferForm::Psb)
            .map(|b| b.slot.index + 1)
            .max()
            .unwrap_or(0);
        2 + top
    });

    // A tile function's tile threadgroup memory.
    let tile_memory = tile_memory(&p.dependencies, device, shape)?;
    // Descriptors.
    let mut resources = ResourcePlanner {
        profile: device.profile(),
        shape: &shape.stages,
        slots: &p.interface.slots,
        dependencies: &p.dependencies,
        stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
        variant: VariantKey::new(),
        diagnostics: Vec::new(),
        uses: Vec::new(),
        template: Default::default(),
        alpha_red_texels: Default::default(),
    };
    let mut heap = HeapPlanner::new(device.profile(), &p.layout, &p.dependencies, shape, |_| {
        vec![Stage::Kernel]
    })?;
    let mut descriptors = Vec::new();
    let separate = separate_samplers(&p.layout);
    let mut table = None;
    for set in &p.layout.sets {
        for b in &set.bindings {
            if let BindingSource::Heap(_) = b.source {
                descriptors.extend(planner(&mut heap)?.array(set.set, b)?);
                continue;
            }
            // A separate sampler: a constexpr one is the layout's immutable sampler, no write; a
            // runtime one the sampler bound at its slot, made for the use.
            if let BindingSource::SeparateSampler { stage, sampler } = b.source {
                let source =
                    resources.separate_sampler(stage, sampler, b.immutable_sampler.as_ref())?;
                if let Some(source) = source {
                    descriptors.push(DescriptorWrite {
                        set: set.set,
                        binding: b.binding,
                        array_element: 0,
                        ty: b.ty,
                        source,
                    });
                }
                continue;
            }
            if let BindingSource::HeapSampler { stage, sampler } = b.source {
                // A constexpr heap sampler is the layout's immutable sampler: no write. A runtime
                // one is the sampler bound at its slot, made for the use.
                let runtime =
                    planner(&mut heap)?.sampler(stage, sampler, b.immutable_sampler.as_ref())?;
                if let Some(slot) = runtime {
                    descriptors.push(DescriptorWrite {
                        set: set.set,
                        binding: b.binding,
                        array_element: 0,
                        ty: b.ty,
                        source: resources.heap_sampler(slot, sampler)?,
                    });
                }
                continue;
            }
            // A storage slot's level views.
            if let (0, BindingSource::Texture(slot)) = (set.set, b.source) {
                if b.count > 1 {
                    let views = resources.level_views(set.set, b.binding, b.ty, slot, b.count)?;
                    descriptors.extend(views.into_iter().zip(0..).map(|(source, i)| {
                        DescriptorWrite {
                            set: set.set,
                            binding: b.binding,
                            array_element: i,
                            ty: b.ty,
                            source,
                        }
                    }));
                    continue;
                }
            }
            if set.set != 0 || b.count != 1 {
                return Err(internal(format!(
                    "a {:?} binding of {} descriptors in set {}",
                    b.source, b.count, set.set
                )));
            }
            let source = match b.source {
                BindingSource::Buffer(slot) => {
                    let pb = planned(slot).ok_or_else(|| {
                        internal(format!("binding of {slot:?} without an interface slot"))
                    })?;
                    match pb.bound {
                        BoundKind::Buffer { .. } => WriteSource::Buffer { slot },
                        BoundKind::Bytes => WriteSource::InlineBytes { slot },
                        BoundKind::Unbound => WriteSource::Neutral(NeutralResource::ZeroBuffer {
                            min_bytes: zero_buffer_bytes(device.profile()),
                        }),
                    }
                }
                BindingSource::Synthetic {
                    stage: Stage::Kernel,
                    binding,
                } => match binding {
                    SyntheticBinding::ArgumentTable => {
                        let size = placement.map_or(0, |pl| pl.argument_table_size);
                        if size == 0 {
                            return Err(internal("an ArgumentTable binding without fields"));
                        }
                        table = Some((set.set, b.binding, size));
                        WriteSource::ArgumentTable {
                            stage: Stage::Kernel,
                            bytes_len: size,
                        }
                    }
                    SyntheticBinding::ConstantBlob if p.stage.constant_blob.is_some() => {
                        WriteSource::ConstantBlob {
                            stage: Stage::Kernel,
                        }
                    }
                    SyntheticBinding::HandleTable => {
                        planner(&mut heap)?.table(Stage::Kernel, set.set, b.binding)?
                    }
                    SyntheticBinding::SamplerTable => {
                        planner(&mut heap)?.sampler_table(Stage::Kernel, set.set, b.binding)?
                    }
                    SyntheticBinding::FunctionTableDirectory => {
                        planner(&mut heap)?.directory(Stage::Kernel, set.set, b.binding)?
                    }
                    // The tables' words on a device without the array's features.
                    SyntheticBinding::FunctionTableSnapshot => {
                        planner(&mut heap)?.snapshot(Stage::Kernel, set.set, b.binding)?
                    }
                    // The pass's tile threadgroup memory.
                    SyntheticBinding::TileMemory => match &tile_memory {
                        Some(memory) => WriteSource::TileMemory {
                            bytes: memory.bytes(),
                        },
                        None => return Err(internal("a tile memory binding without its facts")),
                    },
                    // The pass's imageblock storage, a tile function's explicit imageblock.
                    SyntheticBinding::Imageblock => WriteSource::Imageblock,
                    // The command buffer's status, on a device that does not address buffers.
                    SyntheticBinding::ExecutionStatus => WriteSource::ExecutionStatus,
                    other => return Err(internal(format!("a compute binding {other:?}"))),
                },
                BindingSource::Texture(slot) => {
                    resources.texture(set.set, b.binding, b.ty, slot)?
                }
                BindingSource::Sampler(slot) => resources.sampler(slot)?,
                BindingSource::CombinedTextureSampler { texture, sampler } => {
                    resources.combined(texture, sampler, b.immutable_sampler.as_ref())?
                }
                BindingSource::SeparateTexture { texture, sampler } => {
                    let immutable = separate.get(&(texture.stage, sampler)).ok_or_else(|| {
                        internal(format!(
                            "separate texture {texture:?} without its sampler {sampler:?}"
                        ))
                    })?;
                    resources.separate_texture(texture, sampler, *immutable)?
                }
                // The consumer builds the structures (`bind::acceleration`).
                BindingSource::AccelerationStructure { slot, level } => {
                    WriteSource::AccelerationStructure { slot, level }
                }
                // A sparse texture's tile map.
                BindingSource::TileMap(slot) => resources.tile_map(slot),
                // A two-plane YCbCr texture's planes image, through the device's conversion.
                BindingSource::YCbCrPlanes(slot) => resources.planes_image_write(slot)?,
                // A tile function's attachment.
                BindingSource::TileAttachment(color_index) => WriteSource::TileAttachment {
                    color_index,
                    format: tile_view_format(p, color_index)?,
                    view: StorageAttachmentView::Attachment,
                    view_alias: None,
                },
                other => return Err(internal(format!("a compute binding {other:?}"))),
            };
            descriptors.push(DescriptorWrite {
                set: set.set,
                binding: b.binding,
                array_element: 0,
                ty: b.ty,
                source,
            });
        }
    }

    // Specialization the shape decides.
    let layout = &p.stage.specialization.constants;
    let imageblock = imageblock_bytes(p, shape)?;
    check_tile_pass(p, shape)?;
    check_sparse_textures(&p.dependencies, device.profile(), shape)?;
    let mut fixed = Vec::new();
    // A tile function's tile threadgroup memory starts its arena.
    let mut regions_total: Option<u64> = match (&tile_memory, imageblock) {
        (Some(memory), imageblock) => Some(memory.tile_bytes + imageblock.unwrap_or(0)),
        (None, imageblock) => imageblock,
    };
    for c in layout {
        if let Some(v) = bound_value(c.key, shape, &resources)? {
            fixed.push((c.key, v));
            continue;
        }
        let v = match c.key {
            SpecKey::LocalSize(_) | SpecKey::FlatLocalSize | SpecKey::MultiRegion => continue,
            SpecKey::TgLength(n) => {
                let len = u64::from(ks.threadgroup_lengths.get(&n).copied().unwrap_or(0));
                let len = len.next_multiple_of(TG_ALIGN);
                regions_total = Some(regions_total.unwrap_or(0) + len);
                len
            }
            SpecKey::ImageblockWidth if imageblock.is_some() => {
                u64::from(shape.imageblock_size.width)
            }
            SpecKey::ImageblockHeight if imageblock.is_some() => {
                u64::from(shape.imageblock_size.height)
            }
            // A tile dispatch's render area.
            SpecKey::RenderAreaWidth => u64::from(shape.render_area[0]),
            SpecKey::RenderAreaHeight => u64::from(shape.render_area[1]),
            // A tile function's tile threadgroup memory.
            SpecKey::TgOffset(_) | SpecKey::TileThreadgroupLength | SpecKey::TileGridWidth => {
                tile_memory
                    .as_ref()
                    .and_then(|m| m.spec(c.key))
                    .ok_or_else(|| internal(format!("{:?} without its facts", c.key)))?
            }
            other => return Err(internal(format!("a compute specialization {other:?}"))),
        };
        fixed.push((c.key, v));
    }
    record_specialized(&push, shape, &mut resources)?;
    record_noted([&p.stage], shape, &mut resources, &|_| vec![Stage::Kernel])?;
    // Emulated simdgroups: the reported static length also counts the scratch of the most threads,
    // so the arena's own static part comes from the fact.
    let emulation = p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::SimdgroupEmulation {
            bytes_per_simdgroup,
            arena,
        } => Some((bytes_per_simdgroup, arena)),
        _ => None,
    });
    // A predicated walk's vote word sits beside the arena; the reported static length counts it, so
    // the arena's own static part is the rest.
    let vote = if p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::PredicationVote)
    {
        u64::from(VOTE_BYTES)
    } else {
        0
    };
    let static_bytes = match emulation {
        Some((_, arena)) => arena.map_or(0, u64::from),
        None => u64::from(p.metal_info.static_threadgroup_memory_length).saturating_sub(vote),
    };
    // The imageblock, the arena's last region, starts 16-byte aligned.
    let static_bytes = if imageblock.is_some() {
        static_bytes.next_multiple_of(TG_ALIGN)
    } else {
        static_bytes
    };
    if let Some(regions) = regions_total {
        let required = static_bytes + regions + vote;
        let available = u64::from(limits.max_compute_shared_memory_size);
        if required > available {
            return Err(BindError::Unsupported(Unsupported::needs(
                Feature::ThreadgroupMemorySize,
                Capability::Limit {
                    limit: VkLimit::MaxComputeSharedMemorySize,
                    required,
                    available,
                },
            )));
        }
    }
    let specialization = spec_key(layout, |k| {
        fixed.iter().find(|(key, _)| *key == k).map(|(_, v)| *v)
    });

    // Local-size rules.
    let multiples = p
        .dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::LocalSizeConstraint { multiple } => Some(multiple),
            _ => None,
        })
        .collect();
    let local = LocalRules {
        flat: layout.iter().any(|c| c.key == SpecKey::FlatLocalSize),
        multiples,
        max_size: limits.max_compute_work_group_size,
        max_invocations: limits.max_compute_work_group_invocations,
        max_subgroups: f.subgroup.max_compute_workgroup_subgroups,
        max_count: limits.max_compute_work_group_count,
        required_subgroup_size: p.stage.required_subgroup_size,
        stage_flags: p.stage.stage_flags,
        scratch: emulation.map(|(per_simdgroup, arena)| ScratchRule {
            per_simdgroup,
            fixed: arena_bytes(arena, regions_total.unwrap_or(0)),
            available: limits.max_compute_shared_memory_size,
        }),
        tile: is_tile_function(p),
        tile_threadgroup: tile_threadgroup(p, shape)?,
    };

    // Uses: every bound buffer slot, with its proven access and window, and the used resources
    // the function can reach.
    let compute_stage = vk::PipelineStageFlags2::COMPUTE_SHADER;
    let mut uses: Vec<ResourceUse> = buffers
        .iter()
        .filter(|b| matches!(b.bound, BoundKind::Buffer { .. }))
        .map(|b| buffer_use(b, compute_stage))
        .collect();
    uses.extend(used_uses(
        &shape.used,
        p.interface.needs_pointer_translation,
        p.interface.needs_handle_translation,
        |_| compute_stage,
        compute_stage,
    ));
    uses.append(&mut resources.uses);
    let mut diagnostics = resources.diagnostics;
    let mut template = resources.template;
    if let Some(h) = heap {
        h.finish(
            &mut descriptors,
            &mut uses,
            |_| compute_stage,
            &mut template,
            &mut diagnostics,
        );
    }
    let mut variant = resources.variant;
    variant.psb_slots = psb_variant(&p.dependencies, &buffers, device.profile());
    variant.storage_view_slots = storage_view_variant(&p.dependencies, &buffers, device.profile());
    variant.cross_class = cross_variant(&p.dependencies, &p.interface.slots, shape);
    variant.srgb_textures = srgb_variant(
        device.profile(),
        &p.dependencies,
        shape,
        &variant.cross_class,
    );
    variant.ycbcr_in_place = ycbcr_in_place_variant(&p.dependencies, shape);
    variant.ycbcr_planes_sampled =
        ycbcr_planes_sampled_variant(&p.dependencies, shape, &variant.ycbcr_in_place);
    variant.address_maps = address_map_variant(&p.dependencies, shape);
    variant.cubic_samplers = cubic_variant(&p.dependencies, shape)?;
    variant.anisotropic_samplers = anisotropic_variant(&p.dependencies, shape);
    // A pipeline compiled for emulated simdgroups states its variant; a dispatch asks for it where
    // the flattened tiers cannot run its threadgroups (`dispatch_variant`).
    variant.emulated_simdgroups = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::EmulatedSimdgroups);
    // A pipeline compiled for a flat grid states its variant.
    variant.flat_grid = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::FlatGrid);
    // And one compiled to fold its group counts.
    variant.folded_groups = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::FoldedGroups);
    // A tile function's explicit imageblock standing alone in a pass that keeps fewer of its
    // cells' bytes: the variant that reads the others as zero and drops their writes.
    variant.kept_imageblock_bytes = kept_imageblock_bytes(p, shape);
    // Tables that may pass `maxStorageBufferRange`: the variant that reads them at their
    // addresses (and a pipeline compiled so states it).
    variant.addressed_function_tables = crate::bind::heap::addressed_function_tables(
        device.profile(),
        &p.dependencies,
        &shape.function_tables,
    );

    Ok(BindingPlan {
        variant,
        specialization,
        descriptors,
        vertex_buffers: Vec::new(),
        uses,
        push,
        diagnostics,
        graphics: None,
        compute: Some(ComputeTemplate {
            buffers,
            spec: layout.clone(),
            fixed,
            local,
            fields,
            push_range: placement.and_then(|pl| pl.push_range),
            table,
            root_first,
            storage_alignment: limits.min_storage_buffer_offset_alignment,
            resources: template,
            stage_in_region: p
                .dependencies
                .iter()
                .any(|d| d.fact == RuntimeFact::StageInRegion),
        }),
        views: Default::default(),
    })
}

#[cfg(test)]
mod tests;
