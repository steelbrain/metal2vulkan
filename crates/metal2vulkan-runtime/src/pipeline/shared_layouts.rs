//! The device's pipeline layouts, each shared by every variant whose layout is defined alike
//! (`content_keyed.rs`): a variant's set layouts and pipeline layout are a share of the device's,
//! made by the first variant of that definition.
//!
//! Vulkan holds a pipeline layout compatible with itself for every set and for the push constants
//! (Pipeline Layout Compatibility: two layouts are "compatible for set N" where created "with
//! identically defined descriptor set layouts for sets zero through N" and "identical push
//! constant ranges"; one layout is all of that with itself). So where two variants share one, a
//! pipeline bind between their draws keeps what the draws before it bound and pushed with it
//! (`render/bound.rs`): the next draw pushes only the set-0 descriptors that changed, and leaves
//! out push constants of the same bytes, where each variant's own layout made it push them whole.
//!
//! - **Exact.** Keyed by all that `vkCreateDescriptorSetLayout` and `vkCreatePipelineLayout` take
//!   from a variant: each set's flags and whether its draws push it, each binding's number, type
//!   (a set-0 buffer binding's dynamic type where the variant makes it dynamic), count, stages and
//!   flags and the samplers it bakes, and the push-constant ranges. Two variants
//!   share a layout only where they would create identically defined ones. The baked samplers go
//!   by their handles: the device's set of the same descriptions (`objects::sampler_set`), or a
//!   host step's own, which no other key holds.
//! - **The device's, destroyed by its map alone** (`content_keyed.rs`): each layout holds the
//!   sampler set its set layouts bake until the map destroys it.
//! - **Counted**: `WorkCounts::pipeline_layouts_made`, `WorkCounts::pipeline_layouts_shared`.

use super::content_keyed::{program_call, ContentKeyed, Owned, Share};
use super::objects::{create_set_layout, SamplerSet};
use crate::device::{lock, DeviceShared};
use crate::error::{internal, vulkan, Error};
use crate::release;
use crate::Device;
use ash::vk;
use metal2vulkan::pipeline::{PipelineLayout, PushConstantRange};

/// One shared layout: its set layouts, by set number, and its pipeline layout. It keeps no
/// device functions and destroys nothing itself: only the device's map destroys it, once nothing
/// else holds it or at the device's teardown (`content_keyed.rs`).
pub(crate) struct SharedLayout {
    set_layouts: Vec<vk::DescriptorSetLayout>,
    layout: vk::PipelineLayout,
    /// The immutable samplers its set layouts bake, held until the map destroys it.
    _samplers: Share<SamplerSet>,
}

impl Owned<DeviceShared> for SharedLayout {
    fn destroy(&self, shared: &DeviceShared) {
        let raw = shared.vk();
        // SAFETY: the map lets a layout go only when nothing else shares it, so no variant holds
        // it and no command buffer that recorded with it is pending or recording (each retains
        // the variants its commands used until it completes); or at the device's teardown, when
        // no variant and no command buffer is left (each holds the device).
        unsafe {
            release::object(raw, self.layout);
            for &set_layout in &self.set_layouts {
                release::object(raw, set_layout);
            }
        }
    }
}

impl SharedLayout {
    /// The pipeline layout.
    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// The set layouts, by set number.
    pub(crate) fn set_layouts(&self) -> &[vk::DescriptorSetLayout] {
        &self.set_layouts
    }
}

/// What a shared layout is made from (the module docs).
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct LayoutKey {
    sets: Vec<SetKey>,
    push_constants: Vec<PushConstantRange>,
}

/// One set layout of a [`LayoutKey`].
#[derive(Clone, PartialEq, Eq, Hash)]
struct SetKey {
    flags: metal2vulkan::vk::DescriptorSetLayoutCreateFlags,
    /// Its draws push it (`PUSH_DESCRIPTOR`).
    pushed: bool,
    bindings: Vec<BindingKey>,
}

/// One binding of a [`SetKey`].
#[derive(Clone, PartialEq, Eq, Hash)]
struct BindingKey {
    binding: u32,
    /// Its type as the set layout takes it: the dynamic type of a set-0 buffer binding the
    /// variant makes dynamic (`objects::dynamic_bindings`).
    ty: vk::DescriptorType,
    count: u32,
    stages: metal2vulkan::vk::ShaderStageFlags,
    flags: metal2vulkan::vk::DescriptorBindingFlags,
    /// The samplers it bakes, one per element; none where it bakes none.
    samplers: Vec<vk::Sampler>,
}

/// The device's shared layouts by what made them (`DeviceShared::pipeline_layouts`).
pub(crate) type SharedLayouts = ContentKeyed<LayoutKey, SharedLayout>;

/// The device's layout of `layout`, a compiled pipeline's, whose bindings that bake a sampler take
/// `samplers`' in the layout's order of sets and bindings, its set 0 pushed where
/// `push_set_zero`, and its set-0 bindings numbered in `dynamic` of their dynamic types: made now
/// where the device holds none. In a program's call
/// (`content_keyed::program_call`) one made here enters the device's map; elsewhere a miss makes
/// one of the caller's own that no lookup finds, gone at a later sweep once nothing else holds it:
/// each call here sweeps.
pub(crate) fn shared_layout(
    device: &Device,
    layout: &PipelineLayout,
    samplers: &Share<SamplerSet>,
    (push_set_zero, dynamic): (bool, &[u32]),
) -> Result<Share<SharedLayout>, Error> {
    let mut baked = samplers.samplers().iter().copied();
    let mut sets = Vec::with_capacity(layout.sets.len());
    for (index, set) in layout.sets.iter().enumerate() {
        if set.set as usize != index {
            return Err(internal("a layout set is not at its own index"));
        }
        let mut bindings = Vec::with_capacity(set.bindings.len());
        for binding in &set.bindings {
            let samplers = match binding.immutable_sampler {
                Some(_) => {
                    let sampler = baked
                        .next()
                        .ok_or_else(|| internal("a sampler set shorter than its layout"))?;
                    vec![sampler; binding.count as usize]
                }
                // A sampler array's: one sampler per element, each its own.
                None => {
                    let each: Vec<vk::Sampler> = (&mut baked)
                        .take(binding.immutable_samplers.len())
                        .collect();
                    if each.len() != binding.immutable_samplers.len() {
                        return Err(internal("a sampler set shorter than its layout"));
                    }
                    each
                }
            };
            let ty: vk::DescriptorType = binding.ty.into();
            let made_dynamic = index == 0 && dynamic.contains(&binding.binding);
            bindings.push(BindingKey {
                binding: binding.binding,
                ty: match made_dynamic {
                    true => crate::binding::dynamic_type(ty),
                    false => ty,
                },
                count: binding.count,
                stages: binding.stages,
                flags: binding.flags,
                samplers,
            });
        }
        sets.push(SetKey {
            flags: set.flags,
            pushed: push_set_zero && index == 0,
            bindings,
        });
    }
    let key = LayoutKey {
        sets,
        push_constants: layout.push_constants.clone(),
    };
    let shared = device.shared();
    let program = program_call();
    // The lock is held across the creation, so two of the program's calls never make two layouts
    // of the same definition.
    let mut layouts = lock(&shared.pipeline_layouts);
    let made = layouts.get_or_make(key.clone(), program, || {
        made_layout(device, layout, &key, samplers)
    });
    let gone = layouts.sweep();
    drop(layouts);
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
    let (found, made) = made?;
    match made {
        true => shared.counts.pipeline_layout_made(),
        false => shared.counts.pipeline_layout_shared(),
    }
    Ok(found)
}

/// A new layout of `layout` as `key` defines it, holding `samplers`; what it made goes again where
/// a creation fails.
fn made_layout(
    device: &Device,
    layout: &PipelineLayout,
    key: &LayoutKey,
    samplers: &Share<SamplerSet>,
) -> Result<SharedLayout, Error> {
    let shared = device.shared();
    let mut made = SharedLayout {
        set_layouts: Vec::with_capacity(key.sets.len()),
        layout: vk::PipelineLayout::null(),
        _samplers: samplers.clone(),
    };
    for (set, set_key) in layout.sets.iter().zip(&key.sets) {
        let immutable: Vec<&[vk::Sampler]> =
            (set_key.bindings.iter()).map(|b| &b.samplers[..]).collect();
        let types: Vec<vk::DescriptorType> = set_key.bindings.iter().map(|b| b.ty).collect();
        match create_set_layout(device, set, (&immutable, &types), set_key.pushed) {
            Ok(set_layout) => made.set_layouts.push(set_layout),
            Err(error) => {
                made.destroy(shared);
                return Err(error);
            }
        }
    }
    let push: Vec<vk::PushConstantRange> = (layout.push_constants.iter())
        .map(|range| {
            vk::PushConstantRange::default()
                .stage_flags(range.stages.into())
                .offset(range.offset)
                .size(range.size)
        })
        .collect();
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&made.set_layouts)
        .push_constant_ranges(&push);
    // SAFETY: `info` and its arrays are live for the call.
    match unsafe { shared.vk().create_pipeline_layout(&info, None) } {
        Ok(raw) => made.layout = raw,
        Err(error) => {
            made.destroy(shared);
            return Err(vulkan(error));
        }
    }
    Ok(made)
}

/// Let go of the device's layouts no variant holds any more (`content_keyed.rs`): in a program's
/// call only, where the kept variants applied their queue, before the sampler sets they hold.
pub(super) fn settle_layouts(device: &Device) {
    debug_assert!(
        !super::pool::in_piece(),
        "the device's pipeline layouts settled in a pipeline pool's piece"
    );
    if !program_call() {
        return;
    }
    let shared = device.shared();
    let gone = lock(&shared.pipeline_layouts).settle();
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
}
