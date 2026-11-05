//! From a dispatch's or draw's plan and commands to what the encoder records: the buffer descriptor
//! of every plan write, the command's own data in the command buffer's upload arena, the push holes
//! filled, and the uses the tracker orders. A compute encoder binds one stage's [`ArgumentTable`],
//! a render encoder its vertex and fragment stages'.
//!
//! - **Buffers** a slot binds take exactly their `DescriptorRange`.
//! - **Tables and blobs** (an argument table, a handle table, the constant blob) are uploaded
//!   per command, padded to 16 bytes; a handle table is bound over exactly its planned length
//!   (the shader counts its entries from the range).
//! - **Neutral buffers** are zero regions of the arena, fresh for each command, so a contained
//!   store into a stand-in reaches nothing another command reads.
//! - **Root tables**, one per stage that reads one, are uploaded with a fresh sink's device address
//!   in the sink entry; a stage's push holes take its root table's address and that of the command
//!   buffer's execution status words, each hole in the stage's own push write. A flat variant's
//!   dispatch takes the device address of the region record its preprocess writes
//!   (`Tables::region_record`) in its `ReplayRegionAddress` hole.
//! - **Textures** a slot binds take the `VkImageView` the plan's view asks for (the texture's,
//!   made at its first bind, `binding_view.rs`), in the `GENERAL` layout every runtime image
//!   rests in; a draw's sampled binding of a texture its render pass instance renders takes the
//!   texture's copy (`texture/attached_reads.rs`). **Samplers** take the `VkSampler` of their
//!   state for the write's use and the bind's LOD clamps (`bind::sampler_desc`, `sampler.rs`), its
//!   address modes mapped where the slot samples a texture of an address-mode map
//!   (`address_map.rs`); a combined write checks the view format's filtering against it. A slot
//!   with nothing bound takes the device's neutral image or sampler (`neutral.rs`).
//! - **Texture buffers** a slot binds take a texel buffer view of their buffer's bytes in the
//!   format the plan names, over the range it gives (`buffer_texture.rs`); an unbound slot takes
//!   the device's neutral texel buffer.
//! - **Tessellation factors** (a patch draw's control stage, `tessellation.rs`) take the
//!   encoder's factor buffer over the range the plan gives, as a slot's buffer does.
//! - **Input attachments** (a draw's framebuffer fetch, `framebuffer_fetch.rs`) take the view of
//!   the pass's color attachment of their index, the framebuffer's. **Storage attachments** (its
//!   fetch in the pixel interlock) take a storage view of the attachment's level and layer, or of
//!   its stand-in's (`attachment_stand_in.rs`).
//! - **The heap** (set 1): the device's heap set of the command's layout, which holds every live
//!   texture at its resource ID and each array's neutral at element 0 (`heap_sets.rs`), so a heap
//!   entry writes nothing: the texture it names (made resident, or bound at a slot) is checked as a
//!   slot's view is, and nor does any other image element (an array's neutral where a member's
//!   texture does not fit its class). It holds every live sampler state at its element too, so an
//!   argument sampler writes nothing either. A texture handle table is the device's texture table
//!   (`heap_sets::texture_table`), and a sampler table the device's sampler table of the layout's
//!   kind (`heap_sets::sampler_table`), so a handle reaches every live texture and sampler state
//!   by its ID as the command runs, as Metal's does. A draw whose caller keeps its heap part
//!   (`render/heap_part.rs`) writes, after one draw checked each entry, only the plan's writes
//!   past the heap's entries that draw listed ([`PastHeap`]).
//! - **Function tables** an argument buffer names: a set-1 element takes a listed table's words
//!   buffer whole. On a device without the function-table array's features the stage's snapshot
//!   takes the listed tables' words copied into the arena (`function_table.rs`), which the command
//!   buffer copies again as it commits.
//! - Every other write source names a later step's resource and is refused, typed, before
//!   anything is recorded.

use std::mem::MaybeUninit;

use super::address_map::SamplerMaps;
use super::argument_table::{ArgumentTable, InlineBytes};
use super::buffer::Buffer;
use super::command_buffer::{NotedTable, State};
use super::degradation::EachWatch;
use super::device::lock;
use super::error::{internal, Error, RuntimeGap};
use super::function_table::{self, TableSnapshot, VisibleFunctionTable};
use super::heap_sets::{self, Keeper, SamplerTable, TextureTable, HEAP_SET};
use super::key_hash::KeyMap;
use super::render_pass::{PassLayout, Target};
use super::render_target;
use super::residency::{Residency, Used};
use super::resource::ResourceKey;
use super::sampler::{OwnedSampler, SamplerState};
use super::texture::{Texture, TextureShared};
use super::tracker::{Access, Use, UseRange};
use super::upload::Upload;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{
    AccessClass, BindingPlan, BufferRange, DescriptorBytes, DescriptorRange, DescriptorWrite,
    FunctionTableWords, HeapElements, HeapSource, ImageSource, ImageViewDesc, IndexBufferRef,
    NeutralResource, PushConstantWrite, PushHole, ResourceUse, RootTable, SamplerDescriptor,
    SamplerSource, UseTarget, ViewSetup, WriteSource,
};
use metal2vulkan::metal::sampler::{OpaqueBlackBorder, SamplerUse};
use metal2vulkan::pipeline::{HandleClass, SlotClass, SlotRef, Stage, StorageAttachmentView};
use std::collections::BTreeSet;
use std::ops::{Deref, Range};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};

/// What an encoder binds: each stage's argument table.
pub(super) struct Tables<'a> {
    stages: &'a [(Stage, &'a ArgumentTable)],
    /// The resources the encoder made resident, which the bind state lists as used.
    residency: Option<&'a Residency>,
    /// A render encoder's pass, whose attachments a draw's input and storage attachments read,
    /// and the draw's views of its layers (`render_target::viewed`).
    pass: Option<(&'a PassLayout, ViewSetup)>,
    /// A draw's render pass instance's attachments: a sampled slot binding of a texture that
    /// presents one of its color attachments takes the texture's copy
    /// (`texture/attached_reads.rs`).
    attached: Option<&'a PassLayout>,
    /// The sampler states the bind state lists as argument samplers, in its order.
    argument_samplers: &'a [SamplerState],
    /// Where the plan's heap writes go, when the command writes a heap.
    heap: Option<&'a HeapElements>,
    /// The device's texture table, which the layout's texture handle tables bind.
    texture_table: Option<&'a TextureTable<'a>>,
    /// The device's sampler table, which the layout's sampler tables bind.
    sampler_table: Option<&'a SamplerTable<'a>>,
    /// The tables the bind state lists as function tables, in its order.
    function_tables: &'a [VisibleFunctionTable],
    /// A tile dispatch's scratch images, by color index, which its storage attachments read and
    /// write in place of the pass's attachments (`render/tile_dispatch.rs`).
    tile_scratch: Option<&'a [(u32, Target)]>,
    /// The storage view of a render encoder's shader-side stencil's shadow, when the draw tests
    /// the stencil in the shader (`shadow_stencil.rs`): its returned `[[stencil]]`, or, as the
    /// stencil's stand-in, the draw's stencil test.
    shadow: Option<vk::ImageView>,
    /// The storage view of a render encoder's depth stand-in, when the draw tests the depth in
    /// the shader (`depth_stand_in.rs`).
    depth_stand_in: Option<vk::ImageView>,
    /// The storage views of the `half` shadows a render encoder's current instance holds of
    /// its extended-range colors, by color index (`render/extended_shadow.rs`).
    extended_shadows: Vec<(u32, vk::ImageView)>,
    /// A render encoder's tessellation factor buffer and its offset, when it set one
    /// (`tessellation.rs`).
    factors: Option<(&'a Buffer, u64)>,
    /// The storage view of a render encoder's fragment imageblock storage, when the draw's
    /// fragment function declares an explicit imageblock (`fragment_imageblock.rs`).
    imageblock: Option<vk::ImageView>,
    /// The device address of the 32-byte region record a flat variant's dispatch reads, when
    /// the command is one (`BindingPlan::instantiate_dispatch` of a flat plan): a region of the
    /// command buffer's upload arena (`upload.rs`), with the dispatch's
    /// `VkDispatchIndirectCommand` beside it, which a single-dispatch preprocess writes on the
    /// device before the dispatch. Its `PushHole::ReplayRegionAddress` takes it.
    pub(super) region_record: Option<u64>,
    /// A patch draw's patch index buffer and its offset, when the draw has one.
    patch_indices: Option<(&'a Buffer, u64)>,
    /// An indexed patch draw's control-point index buffer and its offset.
    control_point_indices: Option<(&'a Buffer, u64)>,
    /// A render pass's tile threadgroup memory, when a tile dispatch's function reads it
    /// (`render/tile_memory.rs`).
    tile_memory: Option<&'a Buffer>,
    /// The address-mode maps the command's runtime samplers are bound with (`address_map.rs`).
    sampler_maps: SamplerMaps,
    /// A de-indexed draw's first index's address (`VariantKey::primitive_id_indices`): its
    /// `PushHole::CaptureIndexAddress` takes it.
    indices: Option<u64>,
    /// A draw's slots read through snapshots (`render/snapshot.rs`): each binds its own texture,
    /// not a copy, in the feedback loop layout, and its snapshot at `WriteSource::SnapshotTexels`.
    snapshots: Option<&'a SnapshotViews>,
    /// A merged draw replay's header's address (`render/executions/gpu.rs`): its vertex
    /// function's `PushHole::MergedDrawAddress` takes it.
    merged_header: Option<u64>,
}

/// The fragment texture slots a draw reads through snapshots, and the storage view of each one's
/// snapshot (`render/snapshot.rs`, `Texture::snapshot_texels`).
pub(super) struct SnapshotViews {
    pub(super) slots: BTreeSet<u32>,
    pub(super) views: Vec<(u32, vk::ImageView)>,
}

impl<'a> Tables<'a> {
    /// The tables of `stages`.
    pub(super) fn new(stages: &'a [(Stage, &'a ArgumentTable)]) -> Tables<'a> {
        Tables {
            stages,
            residency: None,
            pass: None,
            attached: None,
            argument_samplers: &[],
            heap: None,
            texture_table: None,
            sampler_table: None,
            function_tables: &[],
            tile_scratch: None,
            shadow: None,
            depth_stand_in: None,
            extended_shadows: Vec::new(),
            factors: None,
            imageblock: None,
            region_record: None,
            patch_indices: None,
            control_point_indices: None,
            tile_memory: None,
            sampler_maps: SamplerMaps::default(),
            indices: None,
            snapshots: None,
            merged_header: None,
        }
    }

    /// The tables, with a draw's slots read through snapshots and their snapshots' views
    /// (`render/snapshot.rs`).
    pub(super) fn with_snapshots(self, snapshots: &'a SnapshotViews) -> Tables<'a> {
        Tables {
            snapshots: Some(snapshots),
            ..self
        }
    }

    /// Whether `slot` is one a draw reads through a snapshot: its binding takes the texture
    /// itself, in the feedback loop layout, never its copy.
    fn snapshot_slot(&self, slot: SlotRef) -> bool {
        slot.stage == Stage::Fragment
            && slot.class == SlotClass::Texture
            && self
                .snapshots
                .is_some_and(|s| s.slots.contains(&slot.index))
    }

    /// The tables, with a merged draw replay's header's address (`Tables::merged_header`).
    pub(super) fn with_merged_header(self, address: u64) -> Tables<'a> {
        Tables {
            merged_header: Some(address),
            ..self
        }
    }

    /// The tables, with a de-indexed draw's first index's address (`Tables::indices`).
    pub(super) fn with_indices(self, address: u64) -> Tables<'a> {
        Tables {
            indices: Some(address),
            ..self
        }
    }

    /// The tables, with the address-mode maps the command's runtime samplers are bound with,
    /// the bind state's (`address_map::sampler_maps`).
    pub(super) fn with_sampler_maps(self, maps: SamplerMaps) -> Tables<'a> {
        Tables {
            sampler_maps: maps,
            ..self
        }
    }

    /// The tables, with a tile dispatch's `scratch` images by color index.
    pub(super) fn with_tile_scratch(self, scratch: &'a [(u32, Target)]) -> Tables<'a> {
        Tables {
            tile_scratch: Some(scratch),
            ..self
        }
    }

    /// The tables, with the render encoder's `pass` and the draw's `views` of its layers.
    pub(super) fn with_pass(self, pass: &'a PassLayout, views: ViewSetup) -> Tables<'a> {
        Tables {
            pass: Some((pass, views)),
            ..self
        }
    }

    /// The tables, with a draw's render pass instance of `instance`'s attachments: a sampled slot
    /// binding of a texture that presents one of its color attachments takes the texture's copy
    /// (`texture/attached_reads.rs`).
    pub(super) fn with_attachments(self, instance: &'a PassLayout) -> Tables<'a> {
        Tables {
            attached: Some(instance),
            ..self
        }
    }

    /// Whether a slot binding of `texture` of the kind `storage` reads it where the draw's
    /// instance renders it: a sampled one of a texture that presents one of its color
    /// attachments (`texture/attached_reads.rs`).
    fn attached(&self, texture: &Texture, storage: bool) -> bool {
        !storage
            && self
                .attached
                .is_some_and(|instance| texture.attached_in(instance))
    }

    /// The tables, with the storage view of the shader-side stencil's `shadow`.
    pub(super) fn with_shadow(self, shadow: vk::ImageView) -> Tables<'a> {
        Tables {
            shadow: Some(shadow),
            ..self
        }
    }

    /// The tables, with the storage view of the depth stand-in `stand_in`.
    pub(super) fn with_depth_stand_in(self, stand_in: vk::ImageView) -> Tables<'a> {
        Tables {
            depth_stand_in: Some(stand_in),
            ..self
        }
    }

    /// The tables, with the storage views of the extended-range colors' `half` `shadows`, by
    /// color index.
    pub(super) fn with_extended_shadows(self, shadows: Vec<(u32, vk::ImageView)>) -> Tables<'a> {
        Tables {
            extended_shadows: shadows,
            ..self
        }
    }

    /// The tables, with the render encoder's tessellation factor buffer and its offset.
    pub(super) fn with_factors(self, buffer: &'a Buffer, offset: u64) -> Tables<'a> {
        Tables {
            factors: Some((buffer, offset)),
            ..self
        }
    }

    /// The tables, with a patch draw's patch index buffer and its offset.
    pub(super) fn with_patch_indices(self, buffer: &'a Buffer, offset: u64) -> Tables<'a> {
        Tables {
            patch_indices: Some((buffer, offset)),
            ..self
        }
    }

    /// The tables, with an indexed patch draw's control-point index buffer and its offset.
    pub(super) fn with_control_point_indices(self, buffer: &'a Buffer, offset: u64) -> Tables<'a> {
        Tables {
            control_point_indices: Some((buffer, offset)),
            ..self
        }
    }

    /// The tables, with the render pass's tile threadgroup memory.
    pub(super) fn with_tile_memory(self, buffer: &'a Buffer) -> Tables<'a> {
        Tables {
            tile_memory: Some(buffer),
            ..self
        }
    }

    /// The tables, with the storage view of the fragment imageblock's storage.
    pub(super) fn with_imageblock(self, imageblock: vk::ImageView) -> Tables<'a> {
        Tables {
            imageblock: Some(imageblock),
            ..self
        }
    }

    /// The tables, with the sampler states the bind state lists as argument samplers and where
    /// the plan's heap writes go.
    pub(super) fn with_heap(
        self,
        samplers: &'a [SamplerState],
        heap: &'a HeapElements,
    ) -> Tables<'a> {
        Tables {
            argument_samplers: samplers,
            heap: Some(heap),
            ..self
        }
    }

    /// The tables, with the device's texture table at the layout's texture handle tables.
    pub(super) fn with_texture_table(self, table: Option<&'a TextureTable<'a>>) -> Tables<'a> {
        Tables {
            texture_table: table,
            ..self
        }
    }

    /// The tables, with the device's sampler table at the layout's sampler tables.
    pub(super) fn with_sampler_table(self, table: Option<&'a SamplerTable<'a>>) -> Tables<'a> {
        Tables {
            sampler_table: table,
            ..self
        }
    }

    /// The tables, with the visible-function tables the bind state lists as function tables.
    pub(super) fn with_function_tables(self, tables: &'a [VisibleFunctionTable]) -> Tables<'a> {
        Tables {
            function_tables: tables,
            ..self
        }
    }

    /// The tables, with the resources the encoder made resident (`use_resource(s)`).
    pub(super) fn with_residency(self, residency: &'a Residency) -> Tables<'a> {
        Tables {
            residency: Some(residency),
            ..self
        }
    }

    /// The resource the encoder made resident under `token`.
    fn used(&self, token: u32) -> Option<&'a Used> {
        self.residency?.get(token)
    }

    /// The pixel format of the texture bound at texture slot `slot`, if one is: what a specialized
    /// notice names (`Device::degrade_specialized`).
    pub(super) fn bound_format(&self, slot: SlotRef) -> Option<metal2vulkan::metal::PixelFormat> {
        slot_texture(self, slot).map(Texture::pixel_format)
    }

    /// The table of `stage`, if the encoder has one.
    fn of(&self, stage: Stage) -> Option<&'a ArgumentTable> {
        self.stages
            .iter()
            .find(|(of, _)| *of == stage)
            .map(|&(_, table)| table)
    }
}

/// What one command's instantiation gives the binder, a dispatch's or a draw's alike.
pub(super) struct Command<'a> {
    pub(super) descriptor_bytes: &'a [DescriptorBytes],
    /// The plan's heap tables, where its caller keeps them apart from the command's other
    /// descriptor bytes (`BindingPlan::heap_tables_into`, `render/heap_part.rs`): empty where
    /// they are among those.
    pub(super) heap_tables: &'a [DescriptorBytes],
    pub(super) descriptor_ranges: &'a [DescriptorRange],
    /// Each stage's root table, by the stages its push hole names.
    pub(super) root_tables: Vec<(metal2vulkan::vk::ShaderStageFlags, &'a RootTable)>,
    pub(super) push_holes: &'a [PushHole],
    /// The command's own resources beyond the plan's uses (an index buffer, indirect arguments).
    pub(super) uses: &'a [ResourceUse],
    /// Each Metal stage's constant blob, for its `ConstantBlob` write.
    pub(super) blobs: Vec<(Stage, &'a Arc<[u8]>)>,
    /// An emulated mesh draw's work buffer and records, for its `MeshWork` and `MeshRecords`
    /// writes.
    pub(super) mesh: Option<MeshRegions>,
    /// An emulated tessellation draw's work buffer and records, for its `TessellationWork` and
    /// `TessellationRecords` writes.
    pub(super) tessellation: Option<MeshRegions>,
}

/// Where an emulated mesh or tessellation draw's work buffer and records lie (`render/mesh.rs`,
/// `render/tessellation_scan.rs`), each bound whole.
#[derive(Clone, Copy)]
pub(super) struct MeshRegions {
    pub(super) work: vk::DescriptorBufferInfo,
    pub(super) records: vk::DescriptorBufferInfo,
}

/// The descriptor of a sampler `state` bound with `clamps`: the state's, its LOD clamps replaced
/// by the bind's when it set them, as plain Metal samples with them
/// ([`metal_lod_clamps`](super::sampler::metal_lod_clamps): inverted ones read the upper clamp's
/// level).
pub(super) fn bound_sampler(state: &SamplerState, clamps: Option<[f32; 2]>) -> SamplerDescriptor {
    let mut descriptor = *state.descriptor();
    if let Some(clamps) = clamps {
        [descriptor.lod_min_clamp, descriptor.lod_max_clamp] =
            super::sampler::metal_lod_clamps(clamps);
    }
    descriptor
}

/// The type a set-0 binding of `ty` takes where its variant makes it dynamic
/// (`objects::dynamic_bindings`): a uniform or storage buffer's dynamic type.
pub(crate) fn dynamic_type(ty: vk::DescriptorType) -> vk::DescriptorType {
    match ty {
        vk::DescriptorType::UNIFORM_BUFFER => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        vk::DescriptorType::STORAGE_BUFFER => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        ty => ty,
    }
}

/// Make the descriptors `writes` holds for the set-0 bindings numbered in `dynamic` (a variant's,
/// ascending: `VariantObjects::dynamic_bindings`) dynamic, and their offsets `offsets`, one per
/// binding in that order, as `vkCmdBindDescriptorSets` takes them ("ordered by the binding
/// numbers in the descriptor set layouts"): each such write of its dynamic type at offset 0, its
/// offset the binding's dynamic offset. The effective offset is "the sum of the dynamic offset
/// and the offset" of the descriptor (`VkDescriptorBufferInfo`), its range unchanged: each
/// command reaches the bytes it reached through the static descriptor, and the set holds the same
/// descriptors whatever the offsets. Valid as the static one: a dynamic offset must be a multiple
/// of the binding's `min*BufferOffsetAlignment`, as a static descriptor's offset must, and offset
/// and range within the buffer likewise. An offset past 32 bits stays in its descriptor, its
/// dynamic offset 0; a binding no write names takes 0.
pub(super) fn dynamic_writes(writes: &mut [Write], dynamic: &[u32], offsets: &mut Vec<u32>) {
    offsets.clear();
    offsets.resize(dynamic.len(), 0);
    if dynamic.is_empty() {
        return;
    }
    for write in writes.iter_mut() {
        dynamic_write(write, dynamic, offsets);
    }
}

/// [`dynamic_writes`]' work on one write, where it is set 0's at a binding `dynamic` numbers:
/// made dynamic, its offset its binding's in `offsets`.
fn dynamic_write(write: &mut Write, dynamic: &[u32], offsets: &mut [u32]) {
    if write.set != 0 {
        return;
    }
    let (Ok(at), Info::Buffer(info)) = (dynamic.binary_search(&write.binding), &mut write.info)
    else {
        return;
    };
    write.ty = dynamic_type(write.ty);
    offsets[at] = 0;
    if let Ok(offset) = u32::try_from(info.offset) {
        offsets[at] = offset;
        info.offset = 0;
    }
}

/// Write again, over `writes` and `offsets` (a command's set-0 writes and dynamic offsets as
/// [`bind_into`] and [`dynamic_writes`] made them, for a variant of the dynamic bindings
/// `dynamic`), the descriptors of the plan's writes from the buffer, texture and sampler slots
/// `slots`, which the command's tables now bind otherwise (another buffer, offset or bytes, another
/// texture or sampler state, the shape the same), with `command`'s ranges for the buffers: the
/// writes and offsets [`bind_into`] and [`dynamic_writes`] make for the tables now, each image
/// descriptor's sampler kept in `samplers` (emptied first) as [`bind_into`] keeps it, through
/// `memo` likewise. Its buffers' descriptors take nothing of the command buffer's state
/// ([`slot_descriptor`]), so it locks none. `Ok(false)` where a write of theirs is not at its
/// place among `writes` (a heap's, which skips writes), holds a keeper, or reads a changed slot
/// otherwise (a texel buffer, a YCbCr texture's planes, an argument sampler): the caller binds
/// whole.
pub(super) fn rebind_into(
    call: &'static str,
    device: &Device,
    (plan, command, tables): (&BindingPlan, &Command<'_>, &Tables<'_>),
    slots: &[SlotRef],
    (writes, dynamic, offsets): (&mut [Write], &[u32], &mut [u32]),
    (mut memo, samplers): (Option<&mut BindMemo>, &mut Vec<Arc<OwnedSampler>>),
) -> Result<bool, Error> {
    samplers.clear();
    if tables.heap.is_some() || writes.len() != plan.descriptors.len() {
        return Ok(false);
    }
    let changed = |slot: &SlotRef| slots.contains(slot);
    for (write, planned) in writes.iter_mut().zip(&plan.descriptors) {
        // Whether its descriptor is a buffer's or an image's of a changed slot.
        let buffer = match &planned.source {
            WriteSource::Buffer { slot } | WriteSource::InlineBytes { slot } => {
                if !changed(slot) {
                    continue;
                }
                true
            }
            WriteSource::Texture { slot, .. } | WriteSource::Sampler { slot, .. } => {
                if !changed(slot) {
                    continue;
                }
                false
            }
            WriteSource::CombinedImageSampler { image, sampler } => {
                let image = match image {
                    ImageSource::Texture { slot, .. } => changed(slot),
                    ImageSource::Neutral(_) => false,
                };
                let sampler = match sampler {
                    SamplerSource::Slot { slot, .. } => changed(slot),
                    SamplerSource::Immutable | SamplerSource::Neutral(_) => false,
                };
                if !image && !sampler {
                    continue;
                }
                false
            }
            // A slot's other reads: a changed one's are its caller's to bind whole.
            WriteSource::TexelBuffer { slot, .. }
            | WriteSource::YCbCrInPlace { slot }
            | WriteSource::YCbCrPlanesImage { slot }
                if changed(slot) =>
            {
                return Ok(false)
            }
            WriteSource::ArgumentSampler { .. }
                if slots.iter().any(|slot| slot.class == SlotClass::Sampler) =>
            {
                return Ok(false)
            }
            _ => continue,
        };
        let ty: vk::DescriptorType = planned.ty.into();
        let place = (write.set, write.binding, write.element);
        if place != (planned.set, planned.binding, planned.array_element) || write.held.is_some() {
            return Ok(false);
        }
        let info = match buffer {
            true => match slot_descriptor(command, tables, planned)? {
                Some(info) => Info::Buffer(info),
                None => return Err(internal("a slot's buffer write of another source")),
            },
            false => {
                let kept = (&mut *samplers, memo.as_deref_mut());
                Info::Image(image_descriptor(call, device, tables, planned, ty, kept)?)
            }
        };
        *write = Write {
            set: planned.set,
            binding: planned.binding,
            element: planned.array_element,
            ty,
            info,
            held: None,
        };
        dynamic_write(write, dynamic, offsets);
    }
    Ok(true)
}

/// One descriptor to write: a buffer's or an image's and sampler's.
pub(super) struct Write {
    pub(super) set: u32,
    pub(super) binding: u32,
    pub(super) element: u32,
    pub(super) ty: vk::DescriptorType,
    pub(super) info: Info,
    /// What keeps what a heap element names while the device's heap set holds it: an argument
    /// sampler's `VkSampler`, a function table's words (`heap_sets.rs`).
    pub(super) held: Option<Keeper>,
}

impl Write {
    /// A copy of it, where it holds no keeper.
    pub(super) fn unheld(&self) -> Option<Write> {
        self.held.is_none().then_some(Write {
            set: self.set,
            binding: self.binding,
            element: self.element,
            ty: self.ty,
            info: self.info,
            held: None,
        })
    }
}

/// What a descriptor write holds.
#[derive(Clone, Copy)]
pub(super) enum Info {
    Buffer(vk::DescriptorBufferInfo),
    Image(vk::DescriptorImageInfo),
    Texel(vk::BufferView),
}

impl Info {
    /// `write` with this as its one descriptor: pointing at it where it lies, with no copy.
    fn written<'a>(&'a self, write: vk::WriteDescriptorSet<'a>) -> vk::WriteDescriptorSet<'a> {
        match self {
            Info::Buffer(info) => write.buffer_info(std::slice::from_ref(info)),
            Info::Image(info) => write.image_info(std::slice::from_ref(info)),
            Info::Texel(view) => write.texel_buffer_view(std::slice::from_ref(view)),
        }
    }
}

/// What a write holds, as [`SetWrite::same`] compares it: its handles, offset, range and layout,
/// the layout by its raw value (ash's info structures print nothing).
impl std::fmt::Debug for Info {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Info::Buffer(info) => (f.debug_tuple("Buffer"))
                .field(&info.buffer)
                .field(&info.offset)
                .field(&info.range)
                .finish(),
            Info::Image(info) => (f.debug_tuple("Image"))
                .field(&info.sampler)
                .field(&info.image_view)
                .field(&info.image_layout.as_raw())
                .finish(),
            Info::Texel(view) => f.debug_tuple("Texel").field(view).finish(),
        }
    }
}

/// One push-constant write: stages, offset, bytes.
pub(super) type PushWrite = (vk::ShaderStageFlags, u32, Vec<u8>);

/// A push hole's address: the stages of the push write it lies in, its offset, the address.
#[derive(Clone, Copy)]
pub(super) struct Hole {
    stages: vk::ShaderStageFlags,
    offset: u32,
    address: u64,
}

/// What a command records besides its pipeline, its draw or groups, and its descriptor writes
/// ([`bind`]'s).
pub(super) struct Bound {
    /// The push holes' addresses, for [`push_writes`].
    pub(super) holes: Vec<Hole>,
    /// The Vulkan samplers the writes name, which the command buffer holds until it completes.
    pub(super) samplers: Vec<Arc<OwnedSampler>>,
}

/// The buffer bound at buffer slot `slot`, if any.
fn slot_buffer<'a>(tables: &Tables<'a>, slot: SlotRef) -> Option<&'a (Buffer, u64)> {
    (slot.class == SlotClass::Buffer)
        .then(|| tables.of(slot.stage)?.buffers.get(&slot.index))
        .flatten()
}

/// The inline bytes set at buffer slot `slot`, if any.
fn slot_bytes<'a>(tables: &Tables<'a>, slot: SlotRef) -> Option<&'a InlineBytes> {
    (slot.class == SlotClass::Buffer)
        .then(|| tables.of(slot.stage)?.bytes.get(&slot.index))
        .flatten()
}

/// The texture bound at texture slot `slot`, if any.
fn slot_texture<'a>(tables: &Tables<'a>, slot: SlotRef) -> Option<&'a Texture> {
    (slot.class == SlotClass::Texture)
        .then(|| tables.of(slot.stage)?.textures.get(&slot.index))
        .flatten()
}

/// The tile map of the sparse texture bound at texture slot `slot`, which the plan reads where
/// the slot's shape is sparse: every sparse texture whose type has a tile map keeps one. One
/// without (a view of another type of a cube texture's slices) is refused
/// `NotImplemented(SparseTileMaps)`: the format lays out no cube map.
fn slot_tile_map<'a>(tables: &Tables<'a>, slot: SlotRef) -> Result<&'a Buffer, Error> {
    let image = slot_texture(tables, slot)
        .and_then(Texture::sparse_image)
        .ok_or_else(|| internal("a tile map read of a slot without a sparse texture"))?;
    image
        .tile_map()
        .map(|tile_map| &tile_map.buffer)
        .ok_or(Error::NotImplemented(RuntimeGap::SparseTileMaps))
}

/// The sampler bound at sampler slot `slot`, if any.
fn slot_sampler<'a>(
    tables: &Tables<'a>,
    slot: SlotRef,
) -> Option<&'a (SamplerState, Option<[f32; 2]>)> {
    (slot.class == SlotClass::Sampler)
        .then(|| tables.of(slot.stage)?.samplers.get(&slot.index))
        .flatten()
}

/// `n` rounded up to 16 bytes, at least 16 (the padding contract).
fn padded(n: usize) -> u64 {
    (n as u64).next_multiple_of(16).max(16)
}

/// The descriptors, uploads, and push hole addresses of `command`, a command (by `call`) of
/// `plan` over `tables`, uploading into the command buffer's `state`: the descriptor writes in
/// `writes`, emptied first, so a caller that keeps the vector writes each command's over the last
/// one's storage (a draw's, `LiveBind::writes`).
pub(super) fn bind(
    call: &'static str,
    device: &Device,
    state: &mut State,
    plan: &BindingPlan,
    command: &Command<'_>,
    tables: &Tables<'_>,
    (memo, writes): (Option<&mut BindMemo>, &mut Vec<Write>),
) -> Result<Bound, Error> {
    let mut samplers = Vec::new();
    let mut bound = bind_into(
        call,
        device,
        state,
        (plan, command, tables),
        (memo, writes, &mut samplers),
        None,
    )?;
    bound.samplers = samplers;
    Ok(bound)
}

/// [`bind`]'s, the samplers the writes name in `samplers`, emptied first (its [`Bound`]'s
/// `samplers` left empty), so a caller that keeps the vector fills each command's into the last
/// one's storage (a draw's, `LiveBind::samplers`). With `past`, where the command has a heap and
/// its caller keeps its heap part (`render/heap_part.rs`): the writes it lists, where it lists
/// them, alone, else every write, listed there (the module docs).
pub(super) fn bind_into(
    call: &'static str,
    device: &Device,
    state: &mut State,
    (plan, command, tables): (&BindingPlan, &Command<'_>, &Tables<'_>),
    (mut memo, writes, samplers): (
        Option<&mut BindMemo>,
        &mut Vec<Write>,
        &mut Vec<Arc<OwnedSampler>>,
    ),
    past: Option<&mut PastHeap>,
) -> Result<Bound, Error> {
    writes.clear();
    writes.reserve(plan.descriptors.len());
    samplers.clear();
    match past.filter(|_| tables.heap.is_some()) {
        Some(past) if past.listed => {
            // Checked against the writes listed again, in place: a draw's allocations are
            // bounded in builds with debug assertions too (`runtime_lane::frame_costs`).
            #[cfg(debug_assertions)]
            {
                let mut kept = past.at.iter();
                for (at, write) in plan.descriptors.iter().enumerate() {
                    if written_at(call, tables, write, false)?.is_some() {
                        assert_eq!(
                            kept.next(),
                            Some(&(at as u32)),
                            "a command's writes past its kept heap that are not its own"
                        );
                    }
                }
                assert_eq!(
                    kept.next(),
                    None,
                    "a command's writes past its kept heap that are not its own"
                );
            }
            for &at in past.at.iter() {
                let write = (plan.descriptors.get(at as usize))
                    .ok_or_else(|| internal("a listed write past the plan's"))?;
                let element = written_at(call, tables, write, false)?
                    .ok_or_else(|| internal("a listed write the command does not write"))?;
                let kept = (&mut *samplers, memo.as_deref_mut());
                let write = written(
                    call,
                    device,
                    state,
                    (command, tables),
                    (write, element),
                    kept,
                )?;
                writes.push(write);
            }
        }
        mut past => {
            // A check that announced a notice its device announces at every occurrence lists
            // nothing: the next command meets it again.
            let watch = EachWatch::start();
            if let Some(past) = &mut past {
                past.forget();
            }
            for (at, write) in plan.descriptors.iter().enumerate() {
                let Some(element) = written_at(call, tables, write, true)? else {
                    continue;
                };
                if let Some(past) = &mut past {
                    past.at.push(at as u32);
                }
                let kept = (&mut *samplers, memo.as_deref_mut());
                let write = written(
                    call,
                    device,
                    state,
                    (command, tables),
                    (write, element),
                    kept,
                )?;
                writes.push(write);
            }
            if let Some(past) = past {
                past.listed = !watch.met();
            }
        }
    }
    let mut roots = Vec::with_capacity(command.root_tables.len());
    for &(stages, table) in &command.root_tables {
        roots.push((stages, root_table(device, state, table)?));
    }
    let mut holes = Vec::with_capacity(command.push_holes.len());
    for hole in command.push_holes {
        holes.push(match *hole {
            PushHole::RootTableAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: roots
                    .iter()
                    .find(|(of, _)| *of == stages)
                    .map(|&(_, address)| address)
                    .ok_or_else(|| internal("a root-table hole without its stage's root table"))?,
            },
            PushHole::ExecutionStatusAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: state
                    .status_address(device)?
                    .ok_or_else(|| internal("a status hole without device addresses"))?,
            },
            // A flat variant's dispatch reads its region record; a device replay's records the
            // runtime does not run.
            PushHole::ReplayRegionAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: tables
                    .region_record
                    .ok_or(Error::NotImplemented(RuntimeGap::CommandReplay))?,
            },
            PushHole::CommandRecordsAddress { .. } => {
                return Err(Error::NotImplemented(RuntimeGap::CommandReplay));
            }
            // A de-indexed draw's first index, or a private list's (`Tables::with_indices`).
            PushHole::CaptureIndexAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: tables
                    .indices
                    .ok_or_else(|| internal("an index hole without its indices"))?,
            },
            // A merged draw replay's header, which its vertex function finds its command in.
            PushHole::MergedDrawAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: tables
                    .merged_header
                    .ok_or_else(|| internal("a merged draw's hole outside a merged draw"))?,
            },
            // Which of a snapshot variant's draws runs: 0 in the plan's bytes, the drawing draw's,
            // and set before each draw as it records (`render/snapshot.rs`).
            PushHole::SnapshotPhase { .. } if tables.snapshots.is_some() => continue,
            PushHole::SnapshotPhase { .. } => {
                return Err(internal(
                    "a snapshot phase hole the runtime did not ask for",
                ));
            }
        });
    }
    Ok(Bound {
        holes,
        samplers: Vec::new(),
    })
}

/// The indices of a plan's descriptor writes that a command whose caller keeps its heap part
/// writes past its heap's entries (`render/heap_part.rs`): listed by [`bind_into`] at a command
/// that checked each entry and met no notice its device announces at every occurrence, so a
/// later command of the same plan under the same heap part writes those alone. They are a
/// function of the plan and of where its heap writes go (`metal2vulkan::bind::HeapElements`),
/// and an entry's check of its texture and the plan's view, which no call changes.
#[derive(Default)]
pub(super) struct PastHeap {
    /// Whether `at` lists them.
    listed: bool,
    /// Their positions among the plan's writes, ascending.
    at: Vec<u32>,
}

impl PastHeap {
    /// List none, keeping the list's storage: a heap part worked out again, or let go of.
    pub(super) fn forget(&mut self) {
        self.listed = false;
        self.at.clear();
    }
}

/// Where `write`, one of its plan's descriptor writes, is written for a command over `tables`:
/// its element, or `None` where the command writes nothing there. A heap's write no handle
/// reaches is none, and in the device's heap set, which holds every texture, every sampler state
/// and the neutrals (the module docs), so are a heap entry, its texture checked as a slot's view
/// is where `check` says so, every other image element (an array's neutral where a member's
/// texture does not fit its class), an argument sampler, and a neutral element
/// (`heap_sets::holds_element`).
fn written_at(
    call: &'static str,
    tables: &Tables<'_>,
    write: &DescriptorWrite,
    check: bool,
) -> Result<Option<u32>, Error> {
    let Some(heap) = tables.heap else {
        return Ok(Some(write.array_element));
    };
    // No handle reaches it.
    let Some(element) = heap.element(write) else {
        return Ok(None);
    };
    if write.set != HEAP_SET {
        return Ok(Some(element));
    }
    match &write.source {
        WriteSource::HeapEntry { source, view, .. } => {
            if check {
                let ty: vk::DescriptorType = write.ty.into();
                let storage = ty == vk::DescriptorType::STORAGE_IMAGE;
                heap_texture(tables, *source)?.bound_view(call, view, storage)?;
            }
            Ok(None)
        }
        WriteSource::ArgumentSampler { .. } => Ok(None),
        _ if heap_sets::holds_element(write.ty.into(), element) => Ok(None),
        _ => Ok(Some(element)),
    }
}

/// The write of `write`, one of its plan's descriptor writes, at `element`, for `command` over
/// `tables` ([`bind_into`]): its image descriptor's sampler kept in `samplers`, through `memo`.
fn written(
    call: &'static str,
    device: &Device,
    state: &mut State,
    (command, tables): (&Command<'_>, &Tables<'_>),
    (write, element): (&DescriptorWrite, u32),
    (samplers, memo): (&mut Vec<Arc<OwnedSampler>>, Option<&mut BindMemo>),
) -> Result<Write, Error> {
    let ty: vk::DescriptorType = write.ty.into();
    let info = match (&write.source, ty) {
        (WriteSource::HeapEntry { .. }, _) => {
            return Err(internal("a heap entry outside the heap set"))
        }
        (_, ty) => match ty {
            vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER => {
                Info::Buffer(descriptor(device, state, command, tables, write)?)
            }
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER | vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                let storage = ty == vk::DescriptorType::STORAGE_TEXEL_BUFFER;
                Info::Texel(texel_descriptor(device, command, tables, write, storage)?)
            }
            _ => Info::Image(image_descriptor(
                call,
                device,
                tables,
                write,
                ty,
                (samplers, memo),
            )?),
        },
    };
    // What keeps an argument sampler's `VkSampler` or a function table's words while the
    // heap set holds them.
    let held: Option<Keeper> = match (&write.source, &info) {
        (WriteSource::ArgumentSampler { .. }, Info::Image(image)) => samplers
            .iter()
            .find(|kept| kept.raw() == image.sampler)
            .map(|kept| Box::new(Arc::clone(kept)) as Keeper),
        (WriteSource::FunctionTable { index }, _) => tables
            .function_tables
            .get(*index as usize)
            .map(|table| Box::new(table.words().clone()) as Keeper),
        _ => None,
    };
    Ok(Write {
        set: write.set,
        binding: write.binding,
        element,
        ty,
        info,
        held,
    })
}

/// Make the device's neutral images, texel buffers and samplers that [`bind`] may write or check
/// for `plan`: at its unbound texture, sampler and texel slots, its heap elements whose texture it
/// cannot present (which the device's heap sets hold, so [`bind`] writes none), and its argument
/// sampler elements past the states listed. So binding a command of the plan later makes none: a
/// host step runs no submission of its own and makes no neutral, which it refuses (`neutral.rs`).
pub(super) fn make_neutrals(device: &Device, plan: &BindingPlan) -> Result<(), Error> {
    for write in &plan.descriptors {
        let ty: vk::DescriptorType = write.ty.into();
        match &write.source {
            WriteSource::Neutral(NeutralResource::TexelBuffer { format }) => {
                let storage = ty == vk::DescriptorType::STORAGE_TEXEL_BUFFER;
                device.neutral_texel_view(*format, storage)?;
            }
            // The zero buffer a read-only binding shares, so a host step binds one already made
            // (the module docs' "no wait"), as it finds the images and samplers made.
            WriteSource::Neutral(
                NeutralResource::ZeroBuffer { min_bytes } | NeutralResource::Sink { min_bytes },
            ) if read_only_buffer(ty) => {
                device.neutral_zero_buffer(padded(*min_bytes as usize))?;
            }
            WriteSource::Neutral(kind @ NeutralResource::Image { .. }) => {
                device.neutral_view(*kind)?;
            }
            WriteSource::Neutral(NeutralResource::Sampler { compare })
            | WriteSource::ArgumentSampler { compare, .. } => {
                device.neutral_sampler(*compare)?;
            }
            WriteSource::CombinedImageSampler { image, sampler } => {
                if let ImageSource::Neutral(kind) = image {
                    device.neutral_view(*kind)?;
                }
                if let SamplerSource::Neutral(NeutralResource::Sampler { compare }) = sampler {
                    device.neutral_sampler(*compare)?;
                }
            }
            WriteSource::HeapEntry {
                class: HandleClass::Image(image),
                ..
            } => {
                if let Some(kind) = NeutralResource::for_image(image) {
                    device.neutral_view(kind)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The texture heap entry `source` names: made resident by the encoder, or bound at a slot.
fn heap_texture<'a>(tables: &Tables<'a>, source: HeapSource) -> Result<&'a Texture, Error> {
    match source {
        HeapSource::Used(token) => match tables.used(token) {
            Some(Used::Texture(texture)) => Ok(texture),
            _ => Err(internal(
                "a heap entry of a texture the encoder did not use",
            )),
        },
        HeapSource::Bound(slot) => slot_texture(tables, slot)
            .ok_or_else(|| internal("a heap entry of an unbound texture slot")),
    }
}

/// The push writes of `writes` (one region's, or one draw's) with `holes` filled: each hole in
/// the write of its own stages that covers its eight bytes. Its writes' bytes become theirs, in
/// their storage.
pub(super) fn push_writes(
    writes: Vec<PushConstantWrite>,
    holes: &[Hole],
) -> Result<Vec<PushWrite>, Error> {
    let mut filled: Vec<PushWrite> = writes
        .into_iter()
        .map(|write| (write.stages.into(), write.offset, write.bytes))
        .collect();
    fill_holes(&mut filled, holes)?;
    Ok(filled)
}

/// [`push_writes`] of `writes`, written over `filled`: each write's bytes copied into the storage
/// of the write a draw before left at its place, where there is one.
pub(super) fn push_writes_into(
    writes: &[PushConstantWrite],
    holes: &[Hole],
    filled: &mut Vec<PushWrite>,
) -> Result<(), Error> {
    filled.truncate(writes.len());
    for (at, write) in writes.iter().enumerate() {
        let stages = write.stages.into();
        match filled.get_mut(at) {
            Some(kept) => {
                (kept.0, kept.1) = (stages, write.offset);
                kept.2.clear();
                kept.2.extend_from_slice(&write.bytes);
            }
            None => filled.push((stages, write.offset, write.bytes.clone())),
        }
    }
    fill_holes(filled, holes)
}

/// Fill each of `holes` in the write of `filled` of its own stages that covers its eight bytes.
fn fill_holes(filled: &mut [PushWrite], holes: &[Hole]) -> Result<(), Error> {
    for hole in holes {
        let write = filled
            .iter_mut()
            .find(|(stages, start, bytes)| {
                *stages == hole.stages
                    && *start <= hole.offset
                    && hole.offset as usize + 8 <= *start as usize + bytes.len()
            })
            .ok_or_else(|| internal("no push write of its stages covers a push hole"))?;
        let at = (hole.offset - write.1) as usize;
        write.2[at..at + 8].copy_from_slice(&hole.address.to_le_bytes());
    }
    Ok(())
}

/// One descriptor of a pushed set 0: [`Write`]'s, without its set number and keeper.
#[derive(Clone, Copy)]
pub(super) struct SetWrite {
    pub(super) binding: u32,
    pub(super) element: u32,
    pub(super) ty: vk::DescriptorType,
    pub(super) info: Info,
}

impl SetWrite {
    /// Set 0's `writes`, then a variant's `immutable` pushes (`VariantObjects::immutable_pushes`):
    /// what a draw pushes.
    pub(super) fn pushed(writes: &[Write], immutable: &[SetWrite]) -> Vec<SetWrite> {
        let mut pushed = Vec::new();
        Self::push_into(writes, immutable, &mut pushed);
        pushed
    }

    /// [`pushed`](Self::pushed) in `pushed`, emptied first: over the last draw's storage, where
    /// the caller keeps it (`LiveBind::pushed`).
    pub(super) fn push_into(writes: &[Write], immutable: &[SetWrite], pushed: &mut Vec<SetWrite>) {
        pushed.clear();
        pushed.extend(
            writes
                .iter()
                .map(SetWrite::of)
                .chain(immutable.iter().copied()),
        );
    }

    /// `write`'s descriptor, without its set number and keeper.
    pub(super) fn of(write: &Write) -> SetWrite {
        SetWrite {
            binding: write.binding,
            element: write.element,
            ty: write.ty,
            info: write.info,
        }
    }

    /// Its raw write, of no set (a push's), pointing at its info.
    pub(super) fn raw(&self) -> vk::WriteDescriptorSet<'_> {
        self.info.written(
            vk::WriteDescriptorSet::default()
                .dst_binding(self.binding)
                .dst_array_element(self.element)
                .descriptor_type(self.ty),
        )
    }

    /// Whether it writes what `other` writes.
    pub(super) fn same(&self, other: &SetWrite) -> bool {
        (self.binding, self.element, self.ty) == (other.binding, other.element, other.ty)
            && match (self.info, other.info) {
                (Info::Buffer(a), Info::Buffer(b)) => {
                    (a.buffer, a.offset, a.range) == (b.buffer, b.offset, b.range)
                }
                (Info::Image(a), Info::Image(b)) => {
                    (a.sampler, a.image_view, a.image_layout)
                        == (b.sampler, b.image_view, b.image_layout)
                }
                (Info::Texel(a), Info::Texel(b)) => a == b,
                _ => false,
            }
    }
}

/// `vkCmdPushDescriptorSet` by its core name from Vulkan 1.4 (`api`), else by
/// `VK_KHR_push_descriptor`'s where the device enabled it; none otherwise.
///
/// # Safety
///
/// `device` is live, of `instance`, made with `enabled` enabled.
pub(crate) unsafe fn push_descriptor_command(
    instance: &ash::Instance,
    device: &ash::Device,
    api: metal2vulkan::device::ApiVersion,
    enabled: &metal2vulkan::device::EnableSet,
) -> Option<vk::PFN_vkCmdPushDescriptorSetKHR> {
    let name: &std::ffi::CStr = if api >= (metal2vulkan::device::ApiVersion { major: 1, minor: 4 })
    {
        c"vkCmdPushDescriptorSet"
    } else if enabled
        .extensions
        .contains(&metal2vulkan::device::VkExtension::KhrPushDescriptor)
    {
        c"vkCmdPushDescriptorSetKHR"
    } else {
        return None;
    };
    // SAFETY: the caller's; a command the device does not have is null.
    let command = unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) }?;
    // SAFETY: the device gave the command of this name, and both names share its signature.
    Some(unsafe {
        std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkCmdPushDescriptorSetKHR>(
            command,
        )
    })
}

/// Push `writes` as set 0 of `layout` at `point` into `command_buffer`, in one
/// `vkCmdPushDescriptorSet` (`push`): each raw write pointing at its write's own info, the raw
/// writes on the stack up to [`WRITES_AT_ONCE`], else on the heap.
///
/// # Safety
///
/// `command_buffer` is recording, `layout`'s set 0 is a push descriptor set layout, and every
/// buffer, view and sampler written stays live while the command buffer can run.
pub(super) unsafe fn push_set(
    push: vk::PFN_vkCmdPushDescriptorSetKHR,
    command_buffer: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    writes: &[SetWrite],
) {
    if writes.len() <= WRITES_AT_ONCE {
        let mut places = [const { MaybeUninit::uninit() }; WRITES_AT_ONCE];
        let raw = fill_raw(writes.iter().map(SetWrite::raw), &mut places);
        // SAFETY: the caller's contract; every info `raw` points at is `writes`', live for the
        // call.
        unsafe {
            push(
                command_buffer,
                point,
                layout,
                0,
                raw.len() as u32,
                raw.as_ptr(),
            )
        };
    } else {
        let raw: Vec<_> = writes.iter().map(SetWrite::raw).collect();
        // SAFETY: as above.
        unsafe {
            push(
                command_buffer,
                point,
                layout,
                0,
                raw.len() as u32,
                raw.as_ptr(),
            )
        };
    }
}

/// `raw` written into `places`' first places, as many as fit: those places, which nothing else
/// writes or zeroes.
pub(super) fn fill_raw<'a, 'p>(
    raw: impl Iterator<Item = vk::WriteDescriptorSet<'a>>,
    places: &'p mut [MaybeUninit<vk::WriteDescriptorSet<'a>>; WRITES_AT_ONCE],
) -> &'p [vk::WriteDescriptorSet<'a>] {
    let mut count = 0;
    for (place, write) in places.iter_mut().zip(raw) {
        place.write(write);
        count += 1;
    }
    // SAFETY: the first `count` places were written above, and a `MaybeUninit<T>` has `T`'s
    // layout.
    unsafe { std::slice::from_raw_parts(places.as_ptr().cast(), count) }
}

/// The descriptor writes [`push_set`] and [`write_sets`] hand the driver from the stack in one call.
pub(super) const WRITES_AT_ONCE: usize = 16;

/// Write `writes` into `sets`, the descriptor sets of their layout by set number: in order,
/// [`WRITES_AT_ONCE`] at a time.
///
/// # Safety
///
/// No pending or recording command uses `sets`, and every buffer, view and sampler written stays
/// live while a command that binds the sets can run.
pub(super) unsafe fn write_sets(
    device: &Device,
    sets: &[vk::DescriptorSet],
    writes: &[Write],
) -> Result<(), Error> {
    if writes.iter().any(|write| write.set as usize >= sets.len()) {
        return Err(internal("a descriptor write to a set the layout lacks"));
    }
    for writes in writes.chunks(WRITES_AT_ONCE) {
        let raw = writes.iter().map(|write| {
            write.info.written(
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[write.set as usize])
                    .dst_binding(write.binding)
                    .dst_array_element(write.element)
                    .descriptor_type(write.ty),
            )
        });
        let mut places = [const { MaybeUninit::uninit() }; WRITES_AT_ONCE];
        let raw = fill_raw(raw, &mut places);
        // SAFETY: the caller's contract; every info `raw` points at is `writes`', live for the
        // call.
        unsafe { device.shared().vk().update_descriptor_sets(raw, &[]) };
    }
    Ok(())
}

/// The set 0 an encoder's last command bound from its command buffer's pools
/// (`EncoderCore::set_zero`): the set, its layout and the descriptors written into it, kept once
/// the command records (`EncoderCore::keep_set_zero`) or as it finds the set its command buffer
/// keeps ([`SetCache`]). A later command of the encoder that writes the same descriptors into a
/// set of the same layout binds this set again, and allocates and writes none
/// (`WorkCounts::set_zero_sets_reused`). Exact:
///
/// - **The set holds what was written.** A set is written once, as it is allocated, and nothing
///   writes it again; set 0's writes hold no keeper (only the heap set's do, `Write::held`).
/// - **No handle it holds names another object.** The set lies in its command buffer's pools,
///   which last until the command buffer completes; so do the objects its writes name, which a
///   command that recorded with them holds (its buffers, textures, views and samplers), the
///   command buffer's arena (uploads) or the device (neutrals). The memo lives with its encoder,
///   inside its command buffer's life.
/// - **The writes compare by value** ([`SetWrite::same`], in order: each binding, element, type
///   and descriptor, a buffer's offset and range, an image's view, sampler and layout), so the set
///   is bound again only where every descriptor written is the one the command writes.
#[derive(Default)]
pub(super) struct LastSetZero {
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    writes: Vec<SetWrite>,
}

impl LastSetZero {
    /// The kept set, where it is of `layout` and holds `writes` (set 0's).
    pub(super) fn holds(
        &self,
        layout: vk::DescriptorSetLayout,
        writes: &[Write],
    ) -> Option<vk::DescriptorSet> {
        let same = self.set != vk::DescriptorSet::null()
            && self.layout == layout
            && self.writes.len() == writes.len()
            && (self.writes.iter().zip(writes))
                .all(|(kept, write)| kept.same(&SetWrite::of(write)));
        same.then_some(self.set)
    }

    /// Keep `set`, of `layout`, as written with `writes` (set 0's), over the last one's storage.
    pub(super) fn keep(
        &mut self,
        layout: vk::DescriptorSetLayout,
        set: vk::DescriptorSet,
        writes: &[Write],
    ) {
        self.layout = layout;
        self.set = set;
        self.writes.clear();
        self.writes.extend(writes.iter().map(SetWrite::of));
    }
}

/// The most sets 0 one generation of a [`SetCache`] keeps.
const SETS_KEPT: usize = 128;

/// The sets 0 a [`Generation`] makes room for at its first set; it grows past them to
/// [`SETS_KEPT`]. A command buffer's commands write a few sets as a rule (one per pipeline state
/// and bound resources), so the room for all of them at once (about 27 KB) would be made and let
/// go of at every command buffer.
const SETS_ROOM: usize = 8;

/// The sets 0 a command buffer's recorded commands allocated and wrote, by what they hold
/// (`EncoderCore::set_zero`): a later command of the command buffer, in any of its encoders, that
/// writes the same descriptors into a set of the same layout binds the set again, and allocates
/// and writes none (`WorkCounts::set_zero_sets_found`). Kept in the command buffer's arena,
/// whose pools hold the sets (`descriptor_arena.rs`), and gone as they go. Exact, as
/// [`LastSetZero`] is:
///
/// - **A set holds what was written.** Written once, as it is allocated; nothing writes it again.
/// - **No handle it holds names another object.** A set is kept only once its command records,
///   and only while the command buffer holds what its commands took strongly, which is until it
///   is committed (`command_buffer/mod.rs`): every object a kept set's writes name is held by the
///   command (its buffers, textures, views and samplers), the command buffer's arena (uploads) or
///   the device (neutrals), so no such object goes, nor can another take its handle, while its
///   sets are kept. A refused command's set is never kept. At the commit the holds go weak, so a
///   rebind may let the pages go, and the cache is shut there and keeps and serves nothing again
///   (`DescriptorArena::shut`): every command recorded from the commit on -- an encoder encoded
///   again at a rebind (`command_buffer/reencode.rs`), a host step's -- allocates and writes its
///   own set.
/// - **The writes compare by value** ([`SetWrite::same`], in order), behind a hash of them
///   ([`set_key`]), so a set is bound again only where every descriptor written is the one the
///   command writes.
///
/// Bounded: two generations of at most [`SETS_KEPT`] sets each. A set found in the older
/// generation moves to the newer one; a newer generation that fills becomes the older one, the
/// older one's sets forgotten (they stay in their pools, unbound by later commands). So a command
/// buffer whose commands write fewer distinct sets than a generation holds allocates each once.
/// A generation keeps its storage as it empties, and makes room for [`SETS_ROOM`] sets at its
/// first: a few allocations a command buffer, none a command but where the room grows.
#[derive(Default)]
pub(crate) struct SetCache {
    generations: [Generation; 2],
    /// The newer generation's place in `generations`.
    newer: usize,
}

/// One generation of a [`SetCache`]: its sets, each's place by its key (a set of other writes
/// under a key a set holds is not kept), and their writes one after another.
#[derive(Default)]
struct Generation {
    by_key: KeyMap<u64, usize>,
    sets: Vec<CachedSet>,
    writes: Vec<SetWrite>,
}

/// One set 0 a [`Generation`] keeps: its layout, the set, and where the descriptors written into
/// it lie in the generation's writes.
struct CachedSet {
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    writes: Range<usize>,
}

impl Generation {
    /// The set of `layout` that holds `writes` it keeps under `key`, by its place.
    fn find(&self, key: u64, layout: vk::DescriptorSetLayout, writes: &[Write]) -> Option<usize> {
        let &at = self.by_key.get(&key)?;
        let kept = &self.sets[at];
        let held = &self.writes[kept.writes.clone()];
        let same = kept.layout == layout
            && held.len() == writes.len()
            && (held.iter().zip(writes)).all(|(held, write)| held.same(&SetWrite::of(write)));
        same.then_some(at)
    }

    /// Keep `set`, of `layout`, holding `writes`, under `key`.
    fn keep(
        &mut self,
        key: u64,
        (layout, set): (vk::DescriptorSetLayout, vk::DescriptorSet),
        writes: impl Iterator<Item = SetWrite>,
    ) {
        if self.sets.capacity() == 0 {
            self.by_key.reserve(SETS_ROOM);
            self.sets.reserve(SETS_ROOM);
            self.writes.reserve(4 * SETS_ROOM);
        }
        let start = self.writes.len();
        self.writes.extend(writes);
        self.by_key.insert(key, self.sets.len());
        self.sets.push(CachedSet {
            layout,
            set,
            writes: start..self.writes.len(),
        });
    }

    /// Forget its sets, keeping its storage.
    fn clear(&mut self) {
        self.by_key.clear();
        self.sets.clear();
        self.writes.clear();
    }
}

/// The hash a [`SetCache`] keeps a set of `layout` holding `writes` under: of what
/// [`SetWrite::same`] compares.
fn set_key(layout: vk::DescriptorSetLayout, writes: &[Write]) -> u64 {
    use ash::vk::Handle;
    use std::hash::Hasher;
    let mut hasher = crate::key_hash::KeyHasher::default();
    hasher.write_u64(layout.as_raw());
    for write in writes {
        hasher.write_u64(u64::from(write.binding) << 32 | u64::from(write.element));
        hasher.write_u64(write.ty.as_raw() as u64);
        match write.info {
            Info::Buffer(info) => {
                hasher.write_u64(info.buffer.as_raw());
                hasher.write_u64(info.offset);
                hasher.write_u64(info.range);
            }
            Info::Image(info) => {
                hasher.write_u64(info.image_view.as_raw());
                hasher.write_u64(info.sampler.as_raw());
                hasher.write_u64(info.image_layout.as_raw() as u64);
            }
            Info::Texel(view) => hasher.write_u64(view.as_raw()),
        }
    }
    hasher.finish()
}

impl SetCache {
    /// The kept set of `layout` that holds `writes` (set 0's), moved to the newer generation
    /// where it was in the older one.
    pub(super) fn find(
        &mut self,
        layout: vk::DescriptorSetLayout,
        writes: &[Write],
    ) -> Option<vk::DescriptorSet> {
        let key = set_key(layout, writes);
        let newer = &self.generations[self.newer];
        if let Some(at) = newer.find(key, layout, writes) {
            return Some(newer.sets[at].set);
        }
        let older = &self.generations[1 - self.newer];
        let at = older.find(key, layout, writes)?;
        let set = older.sets[at].set;
        self.keep_under(key, (layout, set), writes);
        Some(set)
    }

    /// Keep `set`, of `layout`, as written with `writes` (set 0's), once its command recorded.
    pub(super) fn keep(
        &mut self,
        layout: vk::DescriptorSetLayout,
        set: vk::DescriptorSet,
        writes: &[Write],
    ) {
        self.keep_under(set_key(layout, writes), (layout, set), writes);
    }

    /// Keep the set in the newer generation under `key`: where it is full, it becomes the older
    /// one first, the older one's sets forgotten.
    fn keep_under(
        &mut self,
        key: u64,
        set: (vk::DescriptorSetLayout, vk::DescriptorSet),
        writes: &[Write],
    ) {
        if self.generations[self.newer].sets.len() >= SETS_KEPT {
            self.newer = 1 - self.newer;
            self.generations[self.newer].clear();
        }
        self.generations[self.newer].keep(key, set, writes.iter().map(SetWrite::of));
    }
}

/// A set 0 a command allocated and wrote (`EncoderCore::set_zero`), which its encoder keeps for
/// later commands once the command records (`EncoderCore::keep_set_zero`): a refused command's
/// set is never kept.
#[must_use]
pub(super) struct WrittenSet {
    pub(super) layout: vk::DescriptorSetLayout,
    pub(super) set: vk::DescriptorSet,
}

/// The texel buffer view of `write`, a storage texel buffer's (`storage`) or a uniform one's.
fn texel_descriptor(
    device: &Device,
    command: &Command<'_>,
    tables: &Tables<'_>,
    write: &DescriptorWrite,
    storage: bool,
) -> Result<vk::BufferView, Error> {
    match &write.source {
        WriteSource::TexelBuffer { slot, format, .. } => {
            let texture = slot_texture(tables, *slot)
                .ok_or_else(|| internal("the plan binds an unbound texture slot"))?;
            // A swizzled texture buffer's reads take its swizzle in the function, under the
            // slot's `TextureSwizzle` specialization; the plan refuses one at a slot whose stage
            // does not apply it.
            let texels = texture
                .texels()
                .ok_or_else(|| internal("a texel buffer write of a texture with an image"))?;
            let range = command
                .descriptor_ranges
                .iter()
                .find(|r| (r.set, r.binding) == (write.set, write.binding))
                .ok_or_else(|| internal("a texel buffer write without its range"))?
                .range;
            texels.view(*format, storage, range)
        }
        WriteSource::Neutral(NeutralResource::TexelBuffer { format }) => {
            device.neutral_texel_view(*format, storage)
        }
        // The planes of a two-plane YCbCr texture read in place (`texture/ycbcr_image.rs`).
        WriteSource::YCbCrInPlace { slot } => {
            let texture = slot_texture(tables, *slot)
                .ok_or_else(|| internal("the plan binds an unbound texture slot's planes"))?;
            device.shared().counts.ycbcr_planes_bound();
            texture.planes_view()
        }
        other => Err(unexpected(other)),
    }
}

/// The most sampler descriptions a [`BindMemo`] keeps.
const KEPT: usize = 8;

/// The most views a [`BindMemo`] keeps: once full, it lets them all go and starts again. Each
/// is looked up by its key, so an encoder that binds more textures in turn than a short list
/// holds (a page's tiles) finds every one it bound before.
const KEPT_VIEWS: usize = 256;

/// The image views and samplers render encoders' last binds made or found
/// ([`image_descriptor`]), kept by everything each is a function of, so that a bind of the same
/// takes them again without working them out. An encoder that ends gives its memo to its device
/// ([`SpareBindMemos`]), and the next render encoder opened takes it, so a pass that binds the
/// textures and samplers the one before it bound works none out. A memo holds its textures
/// weakly, so it keeps none alive:
///
/// - **A view** of a texture slot is a function of the texture (its storage, image, decoded and
///   twin and record are fixed as it is made; its usage shadow is made once and kept), the
///   view's description, whether the binding is a storage one, and its images' views, which an
///   image keeps once made until it lets them all go under its views lock (a rebind's or a
///   storage form's new `VkImage`, a detach's stand-in, a reattach). Each letting go counts in
///   `DeviceShared::views_epoch`: the views are kept while it holds, and all dropped once it
///   moves. A view is kept by its texture's address ([`KeptView`]), which stays the texture's
///   while the memo keeps the view: a texture found there is the one the view is of.
/// - **A sampler's description** (`sampler_desc`'s answer) for a sampler slot is a function of
///   the descriptor the bind maps the slot's state to (its LOD clamps and the address modes its
///   textures map), the facts `sampler_desc` takes beside the device (the use: a comparison, an
///   integer texture, a cube's explicit gradients; the opaque black border; the image's format)
///   and the device. The bind still asks the
///   state for its `VkSampler` of the description (`SamplerState::vk_sampler`), so the state keeps
///   and forgets its samplers as before.
///
/// A sampler's descriptor is compared exactly, a float field by its bits.
#[derive(Default)]
pub(super) struct BindMemo {
    /// The views epoch the views were kept under.
    epoch: u64,
    views: KeyMap<ViewKey, KeptView>,
    samplers: Vec<KeptSampler>,
}

/// What a view a bind made or found is a function of ([`BindMemo`]): a binding of the texture at
/// the address as the view, a storage one where the first flag is set, of a texture the draw's
/// instance renders where the second is (`Tables::attached`).
type ViewKey = (usize, ImageViewDesc, bool, bool);

/// A view a [`BindMemo`] keeps, and its texture, held weakly: a texture that goes leaves its
/// allocation, and so its address, to its weak handles, so no other texture is at the address
/// while the view is kept.
struct KeptView {
    texture: Weak<TextureShared>,
    raw: vk::ImageView,
}

/// The most bind memos a device keeps for its render encoders to take: one per encoder open at
/// once beyond it starts empty.
const SPARE_BIND_MEMOS: usize = 4;

/// A device's bind memos that ended render encoders left, which the render encoders it opens
/// next take ([`BindMemo`]): at most [`SPARE_BIND_MEMOS`], the latest given taken first.
#[derive(Default)]
pub(super) struct SpareBindMemos(Mutex<Vec<BindMemo>>);

impl SpareBindMemos {
    /// The memo a render encoder opening takes: the latest an encoder gave back, else an empty
    /// one.
    pub(super) fn take(&self) -> BindMemo {
        lock(&self.0).pop().unwrap_or_default()
    }

    /// Keep `memo`, an ended render encoder's, for the next encoder to take, where fewer than
    /// [`SPARE_BIND_MEMOS`] are kept; else it goes.
    pub(super) fn give(&self, memo: BindMemo) {
        let mut spare = lock(&self.0);
        if spare.len() < SPARE_BIND_MEMOS {
            spare.push(memo);
        }
    }
}

/// A sampler description a bind worked out ([`BindMemo`]): `descriptor`'s under `facts`.
struct KeptSampler {
    descriptor: SamplerDescriptor,
    facts: SamplerFacts,
    desc: metal2vulkan::vk::SamplerDesc,
}

/// What `sampler_desc` takes of a binding beside its descriptor and the device.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SamplerFacts {
    use_: SamplerUse,
    opaque_black: OpaqueBlackBorder,
    format: Option<metal2vulkan::vk::Format>,
}

/// Whether `a` and `b` are the same descriptor: every field equal, the floats by their bits.
fn same_descriptor(a: &SamplerDescriptor, b: &SamplerDescriptor) -> bool {
    a == b
        && a.lod_min_clamp.to_bits() == b.lod_min_clamp.to_bits()
        && a.lod_max_clamp.to_bits() == b.lod_max_clamp.to_bits()
}

/// Keep `entry` at the front of `kept`, at most [`KEPT`] entries.
fn keep<T>(kept: &mut Vec<T>, entry: T) {
    if kept.len() == KEPT {
        kept.pop();
    }
    kept.insert(0, entry);
}

/// The image and sampler descriptor of `write`, a descriptor of type `ty` that is no buffer, for
/// a command (by `call`) over `tables`; the Vulkan samplers it names go to `samplers`. Views and
/// sampler descriptions are taken from `memo` where it keeps them.
fn image_descriptor(
    call: &'static str,
    device: &Device,
    tables: &Tables<'_>,
    write: &DescriptorWrite,
    ty: vk::DescriptorType,
    (samplers, memo): (&mut Vec<Arc<OwnedSampler>>, Option<&mut BindMemo>),
) -> Result<vk::DescriptorImageInfo, Error> {
    let image = |view: vk::ImageView| {
        vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)
    };
    // A slot a draw reads through a snapshot samples its texture's subresource that the
    // instance's color attachment holds in the feedback loop layout, the layout its accesses
    // there name (`render_target::moved_barriers`; "This layout is valid only for image
    // subresources of images created with the VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT
    // usage flag set", which `Texture::snapshot_color` checked).
    let in_feedback =
        |slot: SlotRef, info: vk::DescriptorImageInfo| match tables.snapshot_slot(slot) {
            true => info.image_layout(vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT),
            false => info,
        };
    // The kept views are dropped once an image of the device let its views go.
    let (mut kept_views, mut kept_samplers) = match memo {
        Some(memo) => {
            let epoch = device.shared().views_epoch.load(Ordering::Acquire);
            if memo.epoch != epoch {
                memo.views.clear();
                memo.epoch = epoch;
            }
            (Some(&mut memo.views), Some(&mut memo.samplers))
        }
        None => (None, None),
    };
    // A texture whose image and usage lack the binding's usage binds its shadow
    // (`texture/usage_shadow.rs`).
    let mut texture_view = |slot: SlotRef, view: &ImageViewDesc, storage: bool| {
        let texture = slot_texture(tables, slot)
            .ok_or_else(|| internal("the plan binds an unbound texture slot"))?;
        let attached = tables.attached(texture, storage) && !tables.snapshot_slot(slot);
        let Some(kept_views) = kept_views.as_deref_mut() else {
            return texture.slot_view(call, view, storage, attached);
        };
        let key = (texture.address(), *view, storage, attached);
        if let Some(kept) = kept_views.get(&key) {
            debug_assert_eq!(kept.texture.as_ptr() as usize, key.0);
            return Ok(kept.raw);
        }
        device.shared().counts.bind_view_worked_out();
        let raw = texture.slot_view(call, view, storage, attached)?;
        if kept_views.len() == KEPT_VIEWS {
            kept_views.clear();
        }
        let texture = texture.downgrade();
        kept_views.insert(key, KeptView { texture, raw });
        Ok::<_, Error>(raw)
    };
    let mut sampler = |slot: SlotRef, use_: SamplerUse, opaque_black, format| {
        let (state, clamps) = slot_sampler(tables, slot)
            .ok_or_else(|| internal("the plan binds an unbound sampler slot"))?;
        // Its address modes as the textures it samples map them.
        let descriptor = tables
            .sampler_maps
            .sampler(slot, bound_sampler(state, *clamps));
        let facts = SamplerFacts {
            use_,
            opaque_black,
            format,
        };
        let kept = kept_samplers.as_deref().and_then(|kept| {
            kept.iter()
                .find(|k| same_descriptor(&k.descriptor, &descriptor) && k.facts == facts)
        });
        let desc = match kept {
            Some(kept) => kept.desc,
            None => {
                let desc = metal2vulkan::bind::sampler_desc(
                    device.compile_device(),
                    &descriptor,
                    use_,
                    opaque_black,
                    format,
                )
                .map_err(Error::Bind)?;
                if let Some(kept) = kept_samplers.as_deref_mut() {
                    let entry = KeptSampler {
                        descriptor,
                        facts,
                        desc,
                    };
                    keep(kept, entry);
                }
                desc
            }
        };
        let owned = state.vk_sampler(&desc)?;
        let raw = owned.raw();
        samplers.push(owned);
        Ok::<_, Error>(raw)
    };
    let neutral_sampler = |kind: &NeutralResource| match *kind {
        NeutralResource::Sampler { compare } => device.neutral_sampler(compare),
        _ => Err(internal("a neutral sampler of a kind that is no sampler")),
    };
    Ok(match &write.source {
        WriteSource::Texture { slot, view } => in_feedback(
            *slot,
            image(texture_view(
                *slot,
                view,
                ty == vk::DescriptorType::STORAGE_IMAGE,
            )?),
        ),
        WriteSource::CombinedImageSampler {
            image: source,
            sampler: sampler_source,
        } => {
            let (view, format, slot) = match source {
                ImageSource::Texture { slot, view } => (
                    texture_view(*slot, view, false)?,
                    Some(view.format),
                    Some(*slot),
                ),
                ImageSource::Neutral(kind) => (device.neutral_view(*kind)?, kind.format(), None),
            };
            let raw = match sampler_source {
                SamplerSource::Slot {
                    slot,
                    compare,
                    integer,
                    opaque_black,
                } => sampler(
                    *slot,
                    SamplerUse::new(*compare, *integer),
                    *opaque_black,
                    format,
                )?,
                SamplerSource::Immutable => vk::Sampler::null(),
                SamplerSource::Neutral(kind) => neutral_sampler(kind)?,
            };
            match slot {
                Some(slot) => in_feedback(slot, image(view).sampler(raw)),
                None => image(view).sampler(raw),
            }
        }
        WriteSource::Sampler {
            slot,
            compare,
            integer,
            opaque_black,
        } => vk::DescriptorImageInfo::default().sampler(sampler(
            *slot,
            SamplerUse::new(*compare, *integer),
            *opaque_black,
            None,
        )?),
        WriteSource::Neutral(kind @ NeutralResource::Image { .. }) => {
            image(device.neutral_view(*kind)?)
        }
        WriteSource::Neutral(kind @ NeutralResource::Sampler { .. }) => {
            vk::DescriptorImageInfo::default().sampler(neutral_sampler(kind)?)
        }
        WriteSource::Attachment { color_index } if ty == vk::DescriptorType::INPUT_ATTACHMENT => {
            let (pass, views) = tables
                .pass
                .filter(|(pass, _)| pass.fetch_inputs)
                .ok_or_else(|| internal("an input attachment outside a pass that reads them"))?;
            let color = pass
                .colors
                .get(*color_index as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    internal("an input attachment of a color index without a texture")
                })?;
            // The framebuffer's view of the layers the draw renders.
            let target = render_target::viewed(&color.target, views);
            image(render_target::color_view(&target, true)?)
        }
        // A tile dispatch's scratch, viewed in the staging format.
        WriteSource::TileAttachment {
            color_index,
            format,
            ..
        } if ty == vk::DescriptorType::STORAGE_IMAGE && tables.tile_scratch.is_some() => {
            let scratch = tables
                .tile_scratch
                .and_then(|scratch| scratch.iter().find(|(color, _)| color == color_index))
                .map(|(_, target)| target)
                .ok_or_else(|| internal("a tile attachment without its scratch"))?;
            image(render_target::storage_view_in(scratch, *format)?)
        }
        // The attachment's own image, or its sample planes (`sample_planes.rs`), which the
        // encoder made before the draw.
        // Viewed in `view_alias` where the pipeline reads it so (an 8-bit sRGB attachment's
        // codes, a BGRA-order one's RGBA-order twin), which the encoder made the image or planes
        // mutable to (it held the view to `framebuffer_fetch::storage_view_format`, `render.rs`).
        WriteSource::TileAttachment {
            color_index,
            format,
            view,
            view_alias,
        } if ty == vk::DescriptorType::STORAGE_IMAGE && tables.pass.is_some() => {
            let color = tables
                .pass
                .and_then(|(pass, _)| pass.colors.get(*color_index as usize))
                .and_then(Option::as_ref)
                .filter(|color| color.target.format == *format)
                .ok_or_else(|| internal("a storage attachment the pass does not have"))?;
            let in_format = view_alias.unwrap_or(*format);
            match view {
                StorageAttachmentView::Attachment => {
                    image(render_target::storage_view_in(&color.target, in_format)?)
                }
                StorageAttachmentView::SamplePlanes { layers } => {
                    let planes = color
                        .planes
                        .as_ref()
                        .filter(|planes| planes.layers == *layers && planes.view == in_format)
                        .ok_or_else(|| internal("sample planes the pass does not have"))?;
                    image(planes.draw_view()?)
                }
            }
        }
        // The shader-side stencil's shadow, which the encoder made before the draw.
        WriteSource::ShadowStencil if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables.shadow {
                Some(shadow) => image(shadow),
                None => return Err(internal("a shader-side stencil without its shadow")),
            }
        }
        // The depth stand-in, which the encoder made before the draw.
        WriteSource::DepthStandIn if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables.depth_stand_in {
                Some(stand_in) => image(stand_in),
                None => return Err(internal("a shader-side depth test without its stand-in")),
            }
        }
        // The stencil's stand-in: the pass's shadow of the stencil, which the encoder made before
        // the draw (`shadow_stencil.rs`).
        WriteSource::StencilStandIn if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables.shadow {
                Some(shadow) => image(shadow),
                None => return Err(internal("a shader-side stencil test without its stand-in")),
            }
        }
        // An extended-range color's `half` shadow, which the encoder made before the draw.
        WriteSource::ExtendedShadow { color_index } if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables
                .extended_shadows
                .iter()
                .find(|(color, _)| color == color_index)
            {
                Some(&(_, shadow)) => image(shadow),
                None => return Err(internal("an extended-range color's shadow the pass lacks")),
            }
        }
        // A slot's snapshot, which the encoder made before the draw, in `GENERAL`.
        WriteSource::SnapshotTexels { slot } if ty == vk::DescriptorType::STORAGE_IMAGE => {
            let view = tables
                .snapshots
                .and_then(|s| s.views.iter().find(|(at, _)| at == slot))
                .map(|&(_, view)| view);
            match view {
                Some(view) => image(view),
                None => return Err(internal("a snapshot the draw did not take")),
            }
        }
        // The pass's fragment imageblock storage, which the encoder made before the draw.
        WriteSource::Imageblock if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables.imageblock {
                Some(imageblock) => image(imageblock),
                None => return Err(internal("a fragment imageblock without its storage")),
            }
        }
        WriteSource::ArgumentSampler {
            index,
            compare,
            integer,
        } => match tables.argument_samplers.get(*index as usize) {
            Some(state) => {
                let desc = metal2vulkan::bind::sampler_desc(
                    device.compile_device(),
                    state.descriptor(),
                    SamplerUse::new(*compare, *integer),
                    OpaqueBlackBorder::Stated,
                    None,
                )
                .map_err(Error::Bind)?;
                let owned = state.vk_sampler(&desc)?;
                let raw = owned.raw();
                samplers.push(owned);
                vk::DescriptorImageInfo::default().sampler(raw)
            }
            // An element past the listed states: no handle names it.
            None => vk::DescriptorImageInfo::default().sampler(device.neutral_sampler(*compare)?),
        },
        // The planes of a two-plane YCbCr texture as the image the device's conversion samples
        // (`texture/ycbcr_planes.rs`), its sampler the layout's immutable one of that conversion.
        WriteSource::YCbCrPlanesImage { slot }
            if ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER =>
        {
            let texture = slot_texture(tables, *slot)
                .ok_or_else(|| internal("the plan binds an unbound texture slot's planes image"))?;
            let view = texture.planes_image_view()?;
            device.shared().counts.ycbcr_planes_image_bound();
            image(view)
        }
        _ => return Err(unexpected(&write.source)),
    })
}

/// The refusal of a write source the runtime does not fill.
fn unexpected(source: &WriteSource) -> Error {
    match source {
        WriteSource::Imageblock
        | WriteSource::TileAttachment { .. }
        | WriteSource::ShadowStencil
        | WriteSource::DepthStandIn
        | WriteSource::StencilStandIn
        | WriteSource::ExtendedShadow { .. } => {
            Error::NotImplemented(RuntimeGap::RenderAttachments)
        }
        WriteSource::AccelerationStructure { .. } => {
            Error::NotImplemented(RuntimeGap::AccelerationStructures)
        }
        _ => internal("a descriptor of a type that does not hold its write source"),
    }
}

/// The buffer descriptor of `write`.
/// Whether a buffer descriptor of `ty` is read-only to every shader that binds it: a uniform
/// buffer, whose block Vulkan and SPIR-V make read-only, so no store can reach it. A storage
/// buffer is not, whatever a particular shader does with it.
///
/// It decides which bindings may share the device's one neutral zero buffer (`neutral.rs`): a
/// writable binding keeps a region of its own, so nothing a shader stores is ever seen by another
/// command. Structural: the descriptor type alone, never a function or shader name.
fn read_only_buffer(ty: vk::DescriptorType) -> bool {
    matches!(
        ty,
        vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
    )
}

/// The range `command` plans for its buffer write at `key` (set and binding).
fn planned_range(command: &Command<'_>, key: (u32, u32)) -> Result<BufferRange, Error> {
    (command.descriptor_ranges.iter())
        .find(|r| (r.set, r.binding) == key)
        .map(|r| r.range)
        .ok_or_else(|| internal("a buffer write without its range"))
}

/// The descriptor of `range` of `buffer`. A buffer's range is in the buffer's space, which the
/// bind state states it in (`ArgumentTable::stage_table`, `tessellation.rs`): a heap buffer's
/// starts at its offset in the heap's `VkBuffer`.
fn in_space(buffer: &Buffer, range: BufferRange) -> vk::DescriptorBufferInfo {
    let (raw, start) = buffer.space();
    vk::DescriptorBufferInfo::default()
        .buffer(raw)
        .offset(start + range.offset)
        .range(range.range)
}

/// The buffer descriptor of `write` where its source is a slot's: the buffer it binds, or the
/// call's copy of the bytes it sets (whose planned range counts from the region's start), at
/// the range `command` plans; `None` for any other source. It takes nothing of the command
/// buffer's state, so a draw rebound at buffer slots locks none (`binding::rebind_into`).
fn slot_descriptor(
    command: &Command<'_>,
    tables: &Tables<'_>,
    write: &DescriptorWrite,
) -> Result<Option<vk::DescriptorBufferInfo>, Error> {
    let range = || planned_range(command, (write.set, write.binding));
    Ok(Some(match &write.source {
        WriteSource::Buffer { slot } => {
            let (buffer, _) = slot_buffer(tables, *slot)
                .ok_or_else(|| internal("the plan binds an unbound buffer slot"))?;
            in_space(buffer, range()?)
        }
        WriteSource::InlineBytes { slot } => {
            let bytes = slot_bytes(tables, *slot)
                .ok_or_else(|| internal("the plan binds inline bytes the slot does not hold"))?;
            let range = range()?;
            if range.offset + range.range > bytes.region.len() {
                return Err(internal("an inline-bytes range past the call's region"));
            }
            vk::DescriptorBufferInfo::default()
                .buffer(bytes.region.raw())
                .offset(bytes.region.offset() + range.offset)
                .range(range.range)
        }
        _ => return Ok(None),
    }))
}

fn descriptor(
    device: &Device,
    state: &mut State,
    command: &Command<'_>,
    tables: &Tables<'_>,
    write: &DescriptorWrite,
) -> Result<vk::DescriptorBufferInfo, Error> {
    let key = (write.set, write.binding);
    let ty: vk::DescriptorType = write.ty.into();
    let bytes = || {
        (command.descriptor_bytes.iter())
            .chain(command.heap_tables)
            .find(|d| (d.set, d.binding) == key)
            .map(|d| d.bytes.as_slice())
            .ok_or_else(|| internal("a table write without its bytes"))
    };
    let whole = |upload: Upload| {
        vk::DescriptorBufferInfo::default()
            .buffer(upload.raw())
            .offset(upload.offset())
            .range(upload.len())
    };
    if let Some(info) = slot_descriptor(command, tables, write)? {
        return Ok(info);
    }
    let ranged = |buffer: &Buffer| Ok::<_, Error>(in_space(buffer, planned_range(command, key)?));
    let mut upload = |bytes: &[u8], len: u64| state.uploads.upload(device, bytes, len);
    Ok(match &write.source {
        WriteSource::TessellationFactors => {
            let (buffer, _) = tables
                .factors
                .ok_or_else(|| internal("a tessellation factor write without a factor buffer"))?;
            ranged(buffer)?
        }
        // The draw's patch index buffer, or zeros the control stage discards.
        WriteSource::PatchIndices { min_bytes } => match tables.patch_indices {
            Some((buffer, _)) => ranged(buffer)?,
            None => whole(upload(&[], padded(*min_bytes as usize))?),
        },
        // An emulated mesh draw's records and work buffer, whole: the draw's own regions, or the
        // command buffer's records pool (`render/mesh.rs`).
        WriteSource::MeshRecords => {
            command
                .mesh
                .ok_or_else(|| internal("a mesh records write without the draw's dispatch"))?
                .records
        }
        WriteSource::MeshWork => {
            command
                .mesh
                .ok_or_else(|| internal("a mesh work write without the draw's dispatch"))?
                .work
        }
        // An emulated tessellation draw's records and work buffer, whole: the draw's own
        // regions, or the command buffer's records pool (`render/tessellation_scan.rs`).
        WriteSource::TessellationRecords => {
            command
                .tessellation
                .ok_or_else(|| internal("a tessellation records write without the draw's scan"))?
                .records
        }
        WriteSource::TessellationWork => {
            command
                .tessellation
                .ok_or_else(|| internal("a tessellation work write without the draw's scan"))?
                .work
        }
        // An indexed patch draw's control-point index buffer.
        WriteSource::ControlPointIndices => {
            let (buffer, _) = tables.control_point_indices.ok_or_else(|| {
                internal("a control-point index write without a control-point index buffer")
            })?;
            ranged(buffer)?
        }
        // The pass's tile threadgroup memory, every tile's bytes.
        WriteSource::TileMemory { bytes } => {
            let buffer = tables
                .tile_memory
                .ok_or_else(|| internal("a tile memory write without the pass's tile memory"))?;
            let (raw, start) = buffer.space();
            vk::DescriptorBufferInfo::default()
                .buffer(raw)
                .offset(start)
                .range(*bytes)
        }
        WriteSource::ArgumentTable { .. } => {
            let bytes = bytes()?;
            whole(upload(bytes, padded(bytes.len()))?)
        }
        // A texture handle table: the device's, every live texture at its ID (`heap_sets.rs`).
        WriteSource::HandleTable { .. }
            if tables
                .texture_table
                .is_some_and(|table| table.at.contains(&key)) =>
        {
            tables
                .texture_table
                .map(|table| table.info)
                .ok_or_else(|| internal("a texture table gone"))?
        }
        // A sampler table: the device's, every live sampler state at its ID (`heap_sets.rs`).
        WriteSource::HandleTable { .. }
            if tables
                .sampler_table
                .is_some_and(|table| table.at.contains(&key)) =>
        {
            tables
                .sampler_table
                .map(|table| table.info)
                .ok_or_else(|| internal("a sampler table gone"))?
        }
        WriteSource::HandleTable { bytes_len, .. } => {
            let bytes = bytes()?;
            if bytes.len() != *bytes_len as usize {
                return Err(internal("a handle table of another length than planned"));
            }
            let region = upload(bytes, padded(bytes.len()))?;
            vk::DescriptorBufferInfo::default()
                .buffer(region.raw())
                .offset(region.offset())
                .range(u64::from(*bytes_len))
        }
        WriteSource::ConstantBlob { stage } => {
            let blob = command
                .blobs
                .iter()
                .find(|(of, _)| of == stage)
                .map(|&(_, blob)| blob)
                .ok_or_else(|| internal("a constant-blob write without its stage's blob"))?;
            whole(upload(blob, padded(blob.len()))?)
        }
        // The command buffer's status words, the ones every command of it writes, on a device
        // that does not address buffers (`status_address` elsewhere).
        WriteSource::ExecutionStatus => whole(state.status_words(device)?),
        // A read-only binding takes the device's kept zero buffer, made once for its lifetime
        // (`neutral.rs`): zeros of a length the device's own limits fix cannot have changed since
        // it was made, and before this every draw and dispatch that bound one took a fresh
        // zero-filled region, each past half the upload arena's chunk and so a dedicated 64 KiB
        // `VkBuffer` allocated and written to zero. A WRITABLE binding keeps its own region, as
        // every binding did: the one buffer is shared by every command of the device, so a
        // shader that could store through the binding would write zeros another command reads.
        WriteSource::Neutral(
            NeutralResource::ZeroBuffer { min_bytes } | NeutralResource::Sink { min_bytes },
        ) => {
            let bytes = padded(*min_bytes as usize);
            match read_only_buffer(ty) {
                true => vk::DescriptorBufferInfo::default()
                    .buffer(device.neutral_zero_buffer(bytes)?)
                    .offset(0)
                    .range(bytes),
                false => whole(upload(&[], bytes)?),
            }
        }
        WriteSource::HeapEntry { .. } => {
            return Err(Error::NotImplemented(RuntimeGap::ResourceHeap))
        }
        // The bound sparse texture's tile map, whole.
        WriteSource::TileMap { slot } => {
            let buffer = slot_tile_map(tables, *slot)?;
            let (raw, start) = buffer.space();
            vk::DescriptorBufferInfo::default()
                .buffer(raw)
                .offset(start)
                .range(buffer.length())
        }
        // The listed tables' words as they stand now, whole, written again as the command buffer
        // commits.
        WriteSource::FunctionTableSnapshot => {
            let layout = function_table::snapshot_layout(device, tables.function_tables)?;
            let bytes = function_table::snapshot_bytes(&layout, tables.function_tables)?;
            let region = state.uploads.upload(device, &bytes, padded(bytes.len()))?;
            let info = vk::DescriptorBufferInfo::default()
                .buffer(region.raw())
                .offset(region.offset())
                .range(layout.byte_len());
            state.snapshot_tables(TableSnapshot::new(
                region,
                layout,
                tables.function_tables.to_vec(),
            ));
            info
        }
        // A listed table's words, whole; an element past the list is the null table. The plan of
        // tables that may pass `maxStorageBufferRange` reads them at their addresses instead where
        // the device has device addresses.
        WriteSource::FunctionTable { index } => match tables.function_tables.get(*index as usize) {
            Some(table) => {
                let len = table.words_len();
                let limit = device
                    .compile_device()
                    .profile()
                    .limits()
                    .max_storage_buffer_range;
                if len > u64::from(limit) {
                    return Err(function_table::past_storage_buffer_range(device));
                }
                let (raw, start) = table.words().space();
                vk::DescriptorBufferInfo::default()
                    .buffer(raw)
                    .offset(start)
                    .range(len)
            }
            None => whole(upload(
                &[],
                padded(FunctionTableWords::byte_len(0) as usize),
            )?),
        },
        other => return Err(unexpected(other)),
    })
}

/// Upload `table` with a fresh sink's device address in its sink entry, at a 32-byte-aligned
/// device address, noted for its command buffer's resident set (`live_buffers.rs`); its device
/// address.
fn root_table(device: &Device, state: &mut State, table: &RootTable) -> Result<u64, Error> {
    let no_address = || internal("a root table on a device without device addresses");
    let sink = state
        .uploads
        .upload(device, &[], u64::from(NeutralResource::SINK_BYTES))?;
    let sink = sink.device_address().ok_or_else(no_address)?;
    let mut bytes = table.bytes.clone();
    let at = RootTable::SINK_HOST_OFFSET as usize;
    bytes
        .get_mut(at..at + 8)
        .ok_or_else(|| internal("a root table without a sink entry"))?
        .copy_from_slice(&sink.to_le_bytes());
    let len = bytes.len() as u64;
    let uploaded = state
        .uploads
        .upload_aligned(device, &bytes, len, RootTable::ENTRY_ALIGNMENT)?;
    let address = uploaded.device_address().ok_or_else(no_address)?;
    let noted = NotedTable {
        upload: uploaded,
        address,
    };
    state.root_table(device, noted, table.resident_ranges())?;
    Ok(address)
}

/// What `command`, a command of `plan` over `tables`, uses, for the tracker: each bound buffer
/// the plan reads or writes, over the bytes the use reaches, each bound texture, over the levels
/// and slices it presents, and a draw's index buffer `index` and argument buffer `indirect`
/// (each with its offset), over the indices and arguments the draw reads (its index buffer's up
/// to where Metal's fetch stops reading it at most, `IndexBufferRef::fetch_end`: positions past
/// there read a zero-padded copy), in the stages the plan names and with the access types of its
/// class. The command's own uploads and the device's neutrals are fresh or never read back, so
/// nothing orders against them; a sampler is no resource. The uses of what the encoder made
/// resident (`UseTarget::Used`) come first.
pub(super) fn command_uses(
    call: &'static str,
    plan: &BindingPlan,
    command: &Command<'_>,
    tables: &Tables<'_>,
    index: Option<(&Buffer, u64)>,
    indirect: Option<(&Buffer, u64)>,
) -> Result<CommandUses, Error> {
    let mut uses = CommandUses::default();
    uses_into(
        call,
        (plan, None, command, tables),
        (index, indirect),
        &mut uses,
    )?;
    Ok(uses)
}

/// [`command_uses`] in `into`, over the last command's storage, where the caller keeps it
/// (`LiveBind::uses`): its uses of what the encoder made resident, and the keys they reach, kept
/// from the last command where that command's plan was `plan`, or a plan that lists the same uses
/// (as a pipeline state's plans for draws of different primitive types may), and its residency
/// of the generation `tables`' is. They are a function of those alone: each use's resource is the
/// residency's under its token, whose contents a generation names (`residency.rs`), its access
/// and stages the plan's, and its key and range the resource's own, which no call changes (a
/// resource's identity and length, a texture's levels and slices; `WorkCounts::resident_uses_made`
/// counts those made). Builds with debug assertions check those kept against the uses
/// [`command_uses`] makes whole.
pub(super) fn command_uses_into(
    call: &'static str,
    (plan, command, tables): (&Arc<BindingPlan>, &Command<'_>, &Tables<'_>),
    (index, indirect): (Option<(&Buffer, u64)>, Option<(&Buffer, u64)>),
    into: &mut CommandUses,
) -> Result<(), Error> {
    let kept = (&**plan, Some(plan), command, tables);
    let keeps = uses_into(call, kept, (index, indirect), into)?;
    if cfg!(debug_assertions) && keeps {
        let whole = command_uses(call, plan, command, tables, index, indirect)?;
        let same = (into.uses == whole.uses && into.sources == whole.sources)
            && (into.used == whole.used && into.resident == whole.resident);
        assert!(
            same,
            "a command kept uses of what the encoder made resident not its own"
        );
        KEPT_RESIDENT_CHECKS.with(|checks| checks.set(checks.get() + 1));
    }
    Ok(())
}

thread_local! {
    /// The commands whose uses of what the encoder made resident this thread kept (none, where
    /// their plan lists none) and checked against those made whole ([`command_uses_into`]), in builds with debug assertions.
    static KEPT_RESIDENT_CHECKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many commands' kept uses of what the encoder made resident this thread checked so far
/// ([`command_uses_into`]).
#[cfg(all(test, debug_assertions, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn kept_resident_checks() -> u64 {
    KEPT_RESIDENT_CHECKS.with(std::cell::Cell::get)
}

/// What the uses of what an encoder made resident at the start of a [`CommandUses`] were made
/// for (`command_uses_into`): the plan (held weakly, so its allocation names no other plan while
/// this lives) and the residency's generation; and the other plans found to list the plan's
/// uses, or not to, held weakly likewise.
#[derive(Clone)]
struct ResidentOf {
    plan: Weak<BindingPlan>,
    generation: Option<u64>,
    /// Held in place, so a command that switches plan allocates nothing for them: one whose
    /// plan lists other uses makes its uses again, and its own `ResidentOf`.
    compared: [Option<(Weak<BindingPlan>, bool)>; COMPARED_PLANS],
}

/// The most plans a [`ResidentOf`] keeps compared with its own: draws of one pipeline state whose
/// primitive types take plans of their own (Metal's five, `MTLPrimitiveType`) find each compared
/// once.
const COMPARED_PLANS: usize = 5;

impl ResidentOf {
    /// Whether its uses are those `plan` makes under `generation`: made for it, or for a plan
    /// that lists the same uses (`BindingPlan::uses`, which with the residency decide them alone,
    /// [`resident_uses_into`]), found once by comparing them. The plans are compared by address,
    /// which takes no count of their references.
    fn holds(&mut self, plan: &Arc<BindingPlan>, generation: Option<u64>) -> bool {
        let at = Arc::as_ptr(plan);
        if self.generation != generation {
            return false;
        }
        if self.plan.as_ptr() == at {
            return true;
        }
        let found = (self.compared.iter().flatten()).find(|(of, _)| of.as_ptr() == at);
        if let Some(&(_, alike)) = found {
            return alike;
        }
        let alike = (self.plan.upgrade()).is_some_and(|made| made.uses == plan.uses);
        if let Some(free) = self.compared.iter_mut().find(|slot| slot.is_none()) {
            *free = Some((Arc::downgrade(plan), alike));
        }
        alike
    }
}

/// [`command_uses`]' work in `into`: `kept` the plan's `Arc`, where `into` keeps the uses of
/// what the encoder made resident from one command to the next (`command_uses_into`). Whether it
/// kept them.
fn uses_into(
    call: &'static str,
    (plan, kept, command, tables): (
        &BindingPlan,
        Option<&Arc<BindingPlan>>,
        &Command<'_>,
        &Tables<'_>,
    ),
    (index, indirect): (Option<(&Buffer, u64)>, Option<(&Buffer, u64)>),
    into: &mut CommandUses,
) -> Result<bool, Error> {
    // One use at most a resource use of the plan's or the command's.
    let resources = plan.uses.len() + command.uses.len();
    if u32::try_from(resources).is_err() {
        return Err(internal("a command of 2^32 uses"));
    }
    let generation = tables.residency.map(Residency::generation);
    let holds = |of: &mut ResidentOf| kept.is_some_and(|plan| of.holds(plan, generation));
    into.made = 0;
    let keeps = into.of.as_mut().is_some_and(holds);
    match keeps {
        true => {
            into.uses.truncate(into.used);
            into.sources.truncate(into.used);
        }
        false => {
            into.of = None;
            resident_uses_into(plan, tables, into)?;
            into.of = kept.map(|plan| ResidentOf {
                plan: Arc::downgrade(plan),
                generation,
                compared: Default::default(),
            });
            // Kept from one command to the next alone, so the identity of those kept alone
            // (none where there are none: `prefix` reads it only where there are some).
            if into.used > 0 {
                into.prefix = match kept {
                    Some(_) => identity(),
                    None => 0,
                };
            }
        }
    }
    let CommandUses {
        uses,
        sources,
        slots,
        others,
        reaching,
        named_now,
        ..
    } = into;
    uses.reserve(others.len() + command.uses.len());
    // Listed in uses a caller keeps alone, which alone [`patch_buffer_uses`] patches.
    slots.clear();
    let listed = kept.is_some();
    // Every use but those names its resource (a descriptor, a root table's static, a view):
    // the keys of `reaching` named are gathered.
    named_now.clear();
    let plan_others = others.iter().map(|&at| (at, &plan.uses[at as usize]));
    let commands = (command.uses.iter().enumerate())
        .map(|(at, resource)| ((plan.uses.len() + at) as u32, resource));
    for (source, resource) in plan_others.chain(commands) {
        let Some(used) = other_use(call, resource, (index, indirect), tables)? else {
            continue;
        };
        if !reaching.is_empty() && reaching.contains(&used.key) {
            named_now.push(used.key);
        }
        if let UseTarget::Slot(slot) = resource.resource {
            if listed && slot.class == SlotClass::Buffer {
                slots.push((slot, uses.len() as u32));
            }
        }
        uses.push(used);
        sources.push(source);
    }
    named_now.sort_unstable();
    named_now.dedup();
    into.settle_resident(!keeps);
    Ok(keeps)
}

/// Make the uses of what the encoder made resident `plan` lists, as `tables` hold it, first in
/// `into`'s, emptied: their count, `made`, and of the buffers among them (reached by address)
/// the keys no texture use of them names, `reaching`; their stages, `prefix_stages`; and the
/// positions of the plan's other uses, `others`.
fn resident_uses_into(
    plan: &BindingPlan,
    tables: &Tables<'_>,
    into: &mut CommandUses,
) -> Result<(), Error> {
    let CommandUses {
        uses,
        sources,
        used,
        made,
        others,
        reaching,
        prefix_stages,
        ..
    } = into;
    uses.clear();
    sources.clear();
    others.clear();
    reaching.clear();
    *used = 0;
    let mut named = BTreeSet::new();
    for (at, resource) in plan.uses.iter().enumerate() {
        let UseTarget::Used(token) = resource.resource else {
            others.push(at as u32);
            continue;
        };
        if !resource.access.read && !resource.access.write {
            continue;
        }
        let stages: vk::PipelineStageFlags2 = resource.stages.into();
        if stages.is_empty() {
            return Err(internal("a command's resource use in no stage"));
        }
        let access = |flags: vk::AccessFlags2, used: bool| match used {
            true => Access::new(stages, flags),
            false => Access::NONE,
        };
        let buffer = match tables.used(token) {
            Some(Used::Buffer(buffer)) => buffer,
            Some(Used::Heap { bytes, .. }) => bytes,
            Some(Used::IndirectCommandBuffer(icb)) => icb.storage(),
            Some(Used::Texture(texture)) => {
                let (read, write) = image_access(resource.class)?;
                let used = texture.bound_use(
                    access(read, resource.access.read),
                    access(write, resource.access.write),
                );
                named.insert(used.key);
                uses.push(used);
                sources.push(at as u32);
                continue;
            }
            None => return Err(internal("a use of a resource the encoder did not use")),
        };
        let (read, write) = buffer_access(resource.class)?;
        let accesses = (
            access(read, resource.access.read),
            access(write, resource.access.write),
        );
        let used = bytes_use(resource, buffer, 0, accesses);
        reaching.insert(used.key);
        uses.push(used);
        sources.push(at as u32);
    }
    reaching.retain(|key| !named.contains(key));
    *used = uses.len();
    *made += uses.len();
    *prefix_stages = (uses.iter()).fold(vk::PipelineStageFlags2::empty(), |all, used| {
        all | used.read.stages | used.write.stages
    });
    Ok(())
}

/// The use `resource`, a use of the plan's or the command's other than of a resource made
/// resident, makes as `tables` bind it, a draw's index buffer `index` and argument buffer
/// `indirect` (each with its offset): `None` for an unbound slot (a neutral, fresh for the
/// command), a sampler, or a use that neither reads nor writes.
fn other_use(
    call: &'static str,
    resource: &ResourceUse,
    (index, indirect): (Option<(&Buffer, u64)>, Option<(&Buffer, u64)>),
    tables: &Tables<'_>,
) -> Result<Option<Use<ResourceKey>>, Error> {
    if !resource.access.read && !resource.access.write {
        return Ok(None);
    }
    let stages: vk::PipelineStageFlags2 = resource.stages.into();
    if stages.is_empty() {
        return Err(internal("a command's resource use in no stage"));
    }
    let access = |flags: vk::AccessFlags2, used: bool| match used {
        true => Access::new(stages, flags),
        false => Access::NONE,
    };
    let (buffer, offset) = match resource.resource {
        UseTarget::Slot(slot) => match slot.class {
            SlotClass::Buffer => match slot_buffer(tables, slot) {
                Some((buffer, offset)) => (buffer, *offset),
                // An unbound slot reads a neutral, fresh for the command.
                None => return Ok(None),
            },
            SlotClass::Texture => {
                // An unbound slot takes a neutral image; a texture whose image and usage lack
                // the binding's usage, its shadow (`texture/usage_shadow.rs`), as does a draw's
                // sampled one its instance renders (`texture/attached_reads.rs`). A two-plane
                // YCbCr texture's planes read in place: their bytes, not the image, which
                // converts for no such read (`texture/ycbcr_image.rs`).
                let Some(texture) = slot_texture(tables, slot) else {
                    return Ok(None);
                };
                if resource.class == AccessClass::YCbCrPlanes {
                    let read = vk::AccessFlags2::SHADER_SAMPLED_READ;
                    return texture
                        .planes_use(access(read, resource.access.read))
                        .map(Some);
                }
                let (read, write) = image_access(resource.class)?;
                let storage = resource.class == AccessClass::StorageImage;
                let attached = tables.attached(texture, storage) && !tables.snapshot_slot(slot);
                let accesses = (
                    access(read, resource.access.read),
                    access(write, resource.access.write),
                );
                return texture
                    .slot_use(call, storage, attached, accesses)
                    .map(Some);
            }
            SlotClass::Sampler => return Ok(None),
        },
        UseTarget::IndexBuffer => {
            index.ok_or_else(|| internal("an index buffer use of a draw without one"))?
        }
        UseTarget::Indirect => {
            indirect.ok_or_else(|| internal("an indirect use of a command without arguments"))?
        }
        UseTarget::TileMap(slot) => (slot_tile_map(tables, slot)?, 0),
        // Only a plan lists one (`resident_uses_into`).
        UseTarget::Used(_) => {
            return Err(internal("a command's own use of a resource made resident"))
        }
        UseTarget::TessellationFactors => tables
            .factors
            .ok_or_else(|| internal("a tessellation factor use without a factor buffer"))?,
        UseTarget::PatchIndices => tables
            .patch_indices
            .ok_or_else(|| internal("a patch index use of a draw without a buffer"))?,
        UseTarget::ControlPointIndices => tables
            .control_point_indices
            .ok_or_else(|| internal("a control-point index use of a draw without a buffer"))?,
    };
    let (read, write) = buffer_access(resource.class)?;
    let accesses = (
        access(read, resource.access.read),
        access(write, resource.access.write),
    );
    Ok(Some(bytes_use(resource, buffer, offset, accesses)))
}

/// The use `resource` makes of `buffer`'s bytes bound at `offset`, reading and writing as
/// `(read, write)`: its range from the offset (the buffer's rest where it states none; an index
/// buffer's up to its last whole index), as the buffer tracks it (`Buffer::tracked`).
#[inline]
fn bytes_use(
    resource: &ResourceUse,
    buffer: &Buffer,
    offset: u64,
    (read, write): (Access, Access),
) -> Use<ResourceKey> {
    let range = match resource.range {
        Some(range) => {
            let start = offset.saturating_add(range.offset);
            start..start.saturating_add(range.len)
        }
        None => offset..buffer.length().max(offset),
    };
    let range = match resource.resource {
        UseTarget::IndexBuffer => {
            let end = IndexBufferRef::new(0, buffer.length()).fetch_end(range.start);
            range.start..range.end.min(end).max(range.start)
        }
        _ => range,
    };
    let (identity, range) = buffer.tracked(range);
    Use {
        key: identity.key,
        tracked: identity.tracked,
        range: UseRange::Bytes(range),
        read,
        write,
    }
}

/// Patch in `into`, as [`command_uses_into`] made it of `plan`'s and `command`'s uses, the uses
/// of the buffer slots `changed`, as `tables` now bind them, found by `into`'s list of its uses of
/// buffer slots (`CommandUses::slots`), looking at no other use: whether it did, which it does
/// where each such slot binds the buffer it bound (its offset may have moved) and `into` holds the
/// uses that function made alone. Its uses are then those `command_uses_into` makes: every other
/// use is of a slot, buffer or texture no call changed, and the keys, so the resident ones, are
/// the same (`render/draw_again/rebound.rs`). Builds with debug assertions check the list against
/// the uses, and count the uses it looked at ([`patched_slot_uses`]).
pub(super) fn patch_buffer_uses(
    (plan, command, tables): (&BindingPlan, &Command<'_>, &Tables<'_>),
    changed: &[SlotRef],
    into: &mut CommandUses,
) -> Result<bool, Error> {
    if into.sources.len() != into.uses.len() {
        return Ok(false);
    }
    #[cfg(debug_assertions)]
    {
        slot_uses_listed((plan, command), into);
        let looked = into.slots.len() as u64;
        PATCHED_SLOT_USES.with(|patched| {
            let (calls, walked) = patched.get();
            patched.set((calls + 1, walked + looked));
        });
    }
    let CommandUses {
        uses,
        sources,
        slots,
        ..
    } = into;
    for &(slot, at) in slots.iter() {
        if !changed.contains(&slot) {
            continue;
        }
        let at = at as usize;
        let (Some(used), Some(&source)) = (uses.get_mut(at), sources.get(at)) else {
            return Err(internal("a use of a buffer slot past the command's uses"));
        };
        let source = source as usize;
        let resource = match source.checked_sub(plan.uses.len()) {
            None => plan.uses.get(source),
            Some(at) => command.uses.get(at),
        };
        let resource = resource.ok_or_else(|| internal("a use of a resource use since gone"))?;
        let Some((buffer, offset)) = slot_buffer(tables, slot) else {
            return Ok(false);
        };
        // Its reads and writes are its resource use's, as the use made of it holds them.
        let now = bytes_use(resource, buffer, *offset, (used.read, used.write));
        if now.key != used.key {
            return Ok(false);
        }
        *used = now;
    }
    Ok(true)
}

/// Check that `into`, as [`command_uses_into`] made it of `plan`'s and `command`'s uses, lists
/// each of its uses of a buffer slot at its position (`CommandUses::slots`): every use past those
/// of what the encoder made resident (of no slot) of a resource use of a buffer slot. In builds
/// with debug assertions.
#[cfg(debug_assertions)]
fn slot_uses_listed((plan, command): (&BindingPlan, &Command<'_>), into: &CommandUses) {
    let source = |at: usize| {
        let source = into.sources[at] as usize;
        match source.checked_sub(plan.uses.len()) {
            None => plan.uses.get(source),
            Some(at) => command.uses.get(at),
        }
    };
    let listed = (into.used..into.uses.len()).filter_map(|at| match source(at)?.resource {
        UseTarget::Slot(slot) if slot.class == SlotClass::Buffer => Some((slot, at as u32)),
        _ => None,
    });
    assert!(
        listed.eq(into.slots.iter().copied()),
        "a command's uses of buffer slots other than it lists"
    );
}

#[cfg(debug_assertions)]
thread_local! {
    /// The calls of [`patch_buffer_uses`] that reached its uses on this thread, and the uses of
    /// buffer slots they looked at, in builds with debug assertions.
    static PATCHED_SLOT_USES: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// How many calls of [`patch_buffer_uses`] reached their uses on this thread so far, and how
/// many uses of buffer slots they looked at.
#[cfg(all(test, debug_assertions, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn patched_slot_uses() -> (u64, u64) {
    PATCHED_SLOT_USES.with(std::cell::Cell::get)
}

/// A dispatch's or draw's uses (`command_uses`), for the tracker, and the keys among them that
/// the command reaches through its root tables' resident set alone: buffers and heaps an encoder
/// made resident, which no use of the command names. The encoder's core takes both
/// (`EncoderCore::command_reaching`, `note_reaching`); a use added later names its key. The uses
/// of what the encoder made resident come first, kept from one draw to the next while their
/// inputs are the same (`command_uses_into`).
#[derive(Clone, Default)]
pub(super) struct CommandUses {
    uses: Vec<Use<ResourceKey>>,
    resident: BTreeSet<ResourceKey>,
    /// Of each use [`command_uses_into`] made, in order, the index among its plan's and command's
    /// resource uses of the one it was made of: what [`patch_buffer_uses`] patches by.
    sources: Vec<u32>,
    /// Of each of those uses of a buffer slot, in order, the slot and the use's position, in uses
    /// a caller keeps ([`command_uses_into`]): where [`patch_buffer_uses`] finds the uses of the
    /// slots a call changed, looking at no other. Empty in those [`command_uses`] makes, which
    /// nothing patches.
    slots: Vec<(SlotRef, u32)>,
    /// How many of `uses` and `sources`, first, are of what the encoder made resident, made for
    /// `of` (`command_uses_into`); and how many of those the last call made (0 where it kept
    /// them: `WorkCounts::resident_uses_made`).
    used: usize,
    made: usize,
    of: Option<ResidentOf>,
    /// The positions among the plan's uses of those of anything but what the encoder made
    /// resident, ascending.
    others: Vec<u32>,
    /// The keys of the buffers among those uses (reached by address) that none of them names:
    /// `resident` is these but those the command's other uses name, `named`, as the last call
    /// left them (sorted, each once), `named_now` this call's.
    reaching: BTreeSet<ResourceKey>,
    named: Vec<ResourceKey>,
    named_now: Vec<ResourceKey>,
    /// Whether `resident` is other than `reaching` less `named`: a use added since the last call
    /// ([`push`](Self::push)) took a key out of it.
    taken: bool,
    /// The identity of the uses of what the encoder made resident ([`ResidentPrefix`]), 0 where
    /// they have none: given where [`command_uses_into`] made them, from [`IDENTITIES`].
    prefix: u64,
    /// The identity and count of the uses of what the encoder made resident found to share no
    /// memory with an attachment of a render pass of the layout of the version beside them
    /// ([`apart_from`](Self::apart_from)).
    apart: Option<(u64, usize, u64)>,
    /// The identity of `resident` while the uses of what the encoder made resident have one
    /// ([`ReachingPrefix`]): given where it was worked out again, from [`IDENTITIES`].
    resident_at: u64,
    /// The pipeline stages of the uses of what the encoder made resident.
    prefix_stages: vk::PipelineStageFlags2,
}

/// The identities given to [`CommandUses`]' uses of what an encoder made resident, from 1: one of
/// the process's, whose uses no other has.
static IDENTITIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A new one of [`IDENTITIES`].
fn identity() -> u64 {
    IDENTITIES.fetch_add(1, Ordering::Relaxed)
}

/// The uses of what an encoder made resident that come first among a draw's
/// ([`CommandUses::prefix`]), by identity: two draws' uses of the same identity begin with the same
/// `len` uses. What takes a draw's uses may take those of the last one's identity once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ResidentPrefix {
    pub(super) id: u64,
    pub(super) len: usize,
}

/// A draw's [`ResidentPrefix`], with the identity of the keys it reaches through the resident set
/// alone (`resident`) and the stages of those uses ([`CommandUses::reaching_prefix`]): two draws'
/// of the same identities begin with the same uses and reach the same keys so.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct ReachingPrefix {
    pub(super) prefix: ResidentPrefix,
    pub(super) resident: u64,
    pub(super) stages: vk::PipelineStageFlags2,
}

impl CommandUses {
    /// Add `used`, which names its key.
    pub(super) fn push(&mut self, used: Use<ResourceKey>) {
        self.taken |= self.resident.remove(&used.key);
        self.uses.push(used);
    }

    /// The uses of what the encoder made resident at its start, by identity; `None` where there
    /// are none, or they have no identity (made by [`command_uses`]).
    pub(super) fn prefix(&self) -> Option<ResidentPrefix> {
        let prefix = ResidentPrefix {
            id: self.prefix,
            len: self.used,
        };
        (prefix.len > 0 && prefix.id != 0).then_some(prefix)
    }

    /// [`prefix`](Self::prefix), with the identity of the keys the uses reach through the resident
    /// set alone and the prefix's stages (`ReachingPrefix`); `None` also where a use added since
    /// those keys were worked out took one of them ([`push`](Self::push)).
    pub(super) fn reaching_prefix(&self) -> Option<ReachingPrefix> {
        let prefix = self.prefix()?;
        let resident = self.resident_at;
        let stages = self.prefix_stages;
        (resident != 0 && !self.taken).then_some(ReachingPrefix {
            prefix,
            resident,
            stages,
        })
    }

    /// How many uses at its start, of what the encoder made resident, were found to share no
    /// memory with an attachment of a render pass of the layout of `version`
    /// (`Device::degrade_attachment_reads`): its count of them where they are those of the
    /// identity found ([`prefix`](Self::prefix)), else 0.
    pub(super) fn apart(&self, version: u64) -> usize {
        match (self.prefix(), self.apart) {
            (Some(prefix), Some(apart)) if apart == (prefix.id, prefix.len, version) => prefix.len,
            _ => 0,
        }
    }

    /// Note that its uses of what the encoder made resident share no memory with an attachment
    /// of a render pass of the layout of `version` ([`apart`](Self::apart)).
    pub(super) fn apart_from(&mut self, version: u64) {
        self.apart = (self.prefix()).map(|prefix| (prefix.id, prefix.len, version));
    }

    /// How many uses of resources the encoder made resident the last call made: 0 where it kept
    /// the last command's (`command_uses_into`).
    pub(super) fn made(&self) -> usize {
        self.made
    }

    /// Set `resident` to `reaching` less `named_now`, which becomes `named`: worked out again
    /// only where one of those changed (`made_again`: the uses of what the encoder made resident
    /// were), or a use added since took a key out of it.
    fn settle_resident(&mut self, made_again: bool) {
        if !made_again && !self.taken && self.named == self.named_now {
            return;
        }
        self.resident.clone_from(&self.reaching);
        for key in &self.named_now {
            self.resident.remove(key);
        }
        self.named.clone_from(&self.named_now);
        self.taken = false;
        // Their identity, where the uses of what the encoder made resident have one.
        if self.used > 0 {
            self.resident_at = match self.prefix {
                0 => 0,
                _ => identity(),
            };
        }
    }

    /// Add each of `uses`, which name their keys.
    pub(super) fn extend(&mut self, uses: impl IntoIterator<Item = Use<ResourceKey>>) {
        for used in uses {
            self.push(used);
        }
    }

    /// The keys the command reaches through the resident set alone.
    pub(super) fn resident(&self) -> &BTreeSet<ResourceKey> {
        &self.resident
    }
}

impl Deref for CommandUses {
    type Target = [Use<ResourceKey>];

    fn deref(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }
}

/// The read and write access types of a texture used as `class`.
fn image_access(class: AccessClass) -> Result<(vk::AccessFlags2, vk::AccessFlags2), Error> {
    use vk::AccessFlags2 as A;
    match class {
        AccessClass::Sampled => Ok((A::SHADER_SAMPLED_READ, A::NONE)),
        AccessClass::StorageImage => Ok((A::SHADER_STORAGE_READ, A::SHADER_STORAGE_WRITE)),
        _ => Err(internal("a texture use of a class no texture slot has")),
    }
}

/// The read and write access types of a buffer used as `class`.
fn buffer_access(class: AccessClass) -> Result<(vk::AccessFlags2, vk::AccessFlags2), Error> {
    use vk::AccessFlags2 as A;
    match class {
        AccessClass::Storage => Ok((A::SHADER_STORAGE_READ, A::SHADER_STORAGE_WRITE)),
        AccessClass::Uniform => Ok((A::UNIFORM_READ, A::NONE)),
        AccessClass::VertexInput => Ok((A::VERTEX_ATTRIBUTE_READ, A::NONE)),
        AccessClass::Index => Ok((A::INDEX_READ, A::NONE)),
        AccessClass::Indirect => Ok((A::INDIRECT_COMMAND_READ, A::NONE)),
        _ => Err(internal("a buffer use of a class no buffer has")),
    }
}
