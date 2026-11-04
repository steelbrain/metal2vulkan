//! A texture's address-mode map (a runtime extension): how a texture that presents another format's
//! sampling rules is sampled, and how a command's samplers take it.
//!
//! - **The map** ([`AddressModeMap`], `TextureDescriptor::address_mode_map`): for each
//!   `MTLSamplerAddressMode` a sampler states, the mode the texture is sampled with. A texture and
//!   its views carry it. The identity map, every texture's default, changes nothing: a command
//!   whose stages bind no mapped texture computes nothing here and binds, plans and compiles as
//!   it did before the map existed.
//! - **Applied at bind** ([`sampler_maps`]): a runtime sampler at a slot of a stage that samples
//!   a mapped texture is bound as the sampler of its mapped modes, its `VkSampler` and the word
//!   its slot's specialization reads (`SpecKey::SamplerState`) alike, so every emulation that reads
//!   the mode (the linear and bicubic filters by fetches, the gather corners, the border texels)
//!   follows with no code of its own. A mapped sampler is one more state of the slot, as binding
//!   another sampler there is.
//! - **Constexpr samplers**: a constexpr sampler's state is the pipeline's (its immutable sampler,
//!   and the modes its emulations read), so a slot a constexpr sampler samples takes a texture's
//!   map through the pipeline variant of the slot (`VariantKey::address_maps`), whose compile maps
//!   the sampler's modes; Metal samples a constexpr sampler as a runtime one of its modes. A
//!   texture of the identity map keeps every key.
//! - **Refused, typed** (`RuntimeGap::MappedTextureSampling`, naming the map): what cannot follow
//!   the map ([`MappedSampling`]): a constexpr sampler whose modes the map sends where a
//!   constexpr sampler cannot address, or whose modes are not stated; one sampler slot sampling
//!   textures whose maps give it different modes in one stage, whose one state cannot be both;
//!   and a mapped texture a function that samples handles (an argument buffer's or a heap's) can
//!   reach, whose sampler is read from memory.

use super::argument_table::ArgumentTable;
use super::error::RuntimeGap;
use super::residency::Residency;
use super::texture::Texture;
use metal2vulkan::bind::{BindState, SamplerDescriptor};
use metal2vulkan::metal::SamplerAddressMode;
use metal2vulkan::pipeline::{
    BindingSource, HandleClass, PipelineLayout, SamplerRef, SlotClass, SlotRef, Stage,
};
use std::collections::BTreeMap;

/// The modes a texture is sampled with, `metal2vulkan`'s (moved there, so a pipeline variant can
/// name it).
pub use metal2vulkan::metal::sampler::AddressModeMap;

/// How a mapped texture would be sampled that the runtime cannot map at bind
/// (`RuntimeGap::MappedTextureSampling`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MappedSampling {
    /// Through a constexpr sampler whose modes the map sends where a constexpr sampler cannot
    /// address (`AddressModeMap::constexpr_mode`: mirror-clamp-to-edge, or a clamp to zero or to
    /// the border from another mode), or whose modes the layout does not state. Any other is
    /// sampled through its slot's pipeline variant.
    ConstexprSampler,
    /// Through a runtime sampler slot that samples, in the same stage, a texture whose map gives
    /// it other modes: the slot's one state cannot be both.
    SharedSamplerSlot,
    /// By a function that samples textures through handles (an argument buffer's or a heap's),
    /// which can reach the texture, bound at a slot of the command or made resident for it: its
    /// sampler is read from memory.
    Handles,
}

/// The address-mode map each runtime sampler slot of one command is bound with: those whose
/// mapped modes differ from the bound sampler's, by slot. Empty for a command that binds no
/// mapped texture.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct SamplerMaps {
    maps: BTreeMap<SlotRef, AddressModeMap>,
}

impl SamplerMaps {
    /// `desc`, the sampler bound at `slot`, as the command binds it: its modes mapped where the
    /// slot samples a mapped texture.
    pub(super) fn sampler(&self, slot: SlotRef, desc: SamplerDescriptor) -> SamplerDescriptor {
        match self.maps.get(&slot) {
            Some(map) => map.sampler(desc),
            None => desc,
        }
    }

    /// `state`'s samplers as the command binds them, so the plan's specialization reads the
    /// mapped words.
    pub(super) fn apply(&self, state: &mut BindState) {
        for (slot, map) in &self.maps {
            let bound = state
                .stages
                .get_mut(&slot.stage)
                .and_then(|table| table.samplers.get_mut(&slot.index));
            if let Some(bound) = bound {
                bound.desc = map.sampler(bound.desc);
            }
        }
    }
}

/// One texture slot of a layout with the sampler it is sampled through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pair {
    /// A runtime sampler slot of the texture's stage.
    Slot { texture: SlotRef, sampler: SlotRef },
    /// A constexpr sampler, with its modes on `s`, `t` and `r` (`None`: a sampler whose modes
    /// the layout does not state).
    Constexpr {
        texture: SlotRef,
        modes: Option<[SamplerAddressMode; 3]>,
    },
}

/// The texture slots `layout` samples, each with its sampler, and whether a function of it
/// samples textures through handles.
pub(super) fn pairs(layout: &PipelineLayout) -> (Vec<Pair>, bool) {
    let mut pairs = Vec::new();
    let mut handles = false;
    for binding in layout.sets.iter().flat_map(|set| &set.bindings) {
        match binding.source {
            BindingSource::CombinedTextureSampler { texture, sampler }
            | BindingSource::SeparateTexture { texture, sampler } => {
                let pair = match sampler.sampler {
                    SamplerRef::Slot(index) => Pair::Slot {
                        texture,
                        sampler: SlotRef {
                            stage: texture.stage,
                            class: SlotClass::Sampler,
                            index,
                        },
                    },
                    // A constexpr sampler, and any other the layout names in time: its state
                    // is no bound sampler's.
                    other => Pair::Constexpr {
                        texture,
                        modes: other.constexpr_address_modes(),
                    },
                };
                if !pairs.contains(&pair) {
                    pairs.push(pair);
                }
            }
            BindingSource::HeapSampler { .. }
            | BindingSource::Heap(HandleClass::Sampler { .. }) => {
                handles = true;
            }
            _ => {}
        }
    }
    (pairs, handles)
}

/// The maps the runtime sampler slots of `pairs` take: each slot's sampler, `sampler_of`, mapped
/// by the map of every texture it samples, `map_of`. Refused where a constexpr sampler samples a
/// texture whose map changes one of its modes (or its modes are not stated), where one slot's
/// textures map it to different modes, and, where `handles`, where a texture of `reached` (those
/// a function sampling handles can reach) has a map.
pub(super) fn resolve(
    pairs: &[Pair],
    handles: bool,
    map_of: impl Fn(SlotRef) -> AddressModeMap,
    sampler_of: impl Fn(SlotRef) -> Option<SamplerDescriptor>,
    reached: impl IntoIterator<Item = AddressModeMap>,
) -> Result<SamplerMaps, RuntimeGap> {
    let refuse = |map, through| RuntimeGap::MappedTextureSampling { map, through };
    if handles {
        if let Some(map) = reached.into_iter().find(|map| !map.is_identity()) {
            return Err(refuse(map, MappedSampling::Handles));
        }
    }
    // Each runtime sampler slot's bound state, its modes as the first texture it samples maps
    // them, and that texture's map.
    let mut seen: BTreeMap<SlotRef, (SamplerDescriptor, [SamplerAddressMode; 3], AddressModeMap)> =
        BTreeMap::new();
    for pair in pairs {
        match *pair {
            Pair::Constexpr { texture, modes } => {
                // A constexpr sampler samples a mapped texture through the pipeline variant of
                // its slot (`VariantKey::address_maps`), whose state holds its modes mapped, as
                // Metal samples it (a constexpr sampler samples as a runtime one
                // of its modes): refused only where the map sends a mode where a constexpr
                // sampler cannot address, or its modes are not stated.
                let map = map_of(texture);
                let stated =
                    modes.is_some_and(|m| m.iter().all(|&mode| map.constexpr_mode(mode).is_some()));
                if !map.is_identity() && !stated {
                    return Err(refuse(map, MappedSampling::ConstexprSampler));
                }
            }
            Pair::Slot { texture, sampler } => {
                // An unbound slot binds the device's neutral sampler, which no map changes.
                let Some(desc) = sampler_of(sampler) else {
                    continue;
                };
                let map = map_of(texture);
                let mapped = map.sampler(desc);
                let modes = [
                    mapped.s_address_mode,
                    mapped.t_address_mode,
                    mapped.r_address_mode,
                ];
                match seen.get(&sampler) {
                    Some(&(_, first, other)) if first != modes => {
                        let named = if map.is_identity() { other } else { map };
                        return Err(refuse(named, MappedSampling::SharedSamplerSlot));
                    }
                    Some(_) => {}
                    None => {
                        seen.insert(sampler, (desc, modes, map));
                    }
                }
            }
        }
    }
    Ok(SamplerMaps {
        maps: seen
            .into_iter()
            .filter(|(_, (desc, modes, _))| {
                *modes
                    != [
                        desc.s_address_mode,
                        desc.t_address_mode,
                        desc.r_address_mode,
                    ]
            })
            .map(|(slot, (_, _, map))| (slot, map))
            .collect(),
    })
}

/// The maps one command of `layout` binds its runtime samplers with ([`resolve`]): `tables`
/// are its stages' argument tables, and `residency` what its encoder made resident. A command
/// whose tables bind no mapped texture, and whose encoder made none resident
/// (`Residency::maps_addresses`, kept as textures are made resident), takes none, and reads no
/// more of `layout` and no resident texture.
pub(super) fn sampler_maps(
    layout: &PipelineLayout,
    tables: &[(Stage, &ArgumentTable)],
    residency: Option<&Residency>,
) -> Result<SamplerMaps, RuntimeGap> {
    let mapped = |texture: &Texture| !texture.address_mode_map().is_identity();
    let bound = tables.iter().flat_map(|(_, table)| table.textures.values());
    let resident = || residency.into_iter().flat_map(Residency::textures);
    if !bound.clone().any(mapped) && !residency.is_some_and(Residency::maps_addresses) {
        return Ok(SamplerMaps::default());
    }
    let (pairs, handles) = pairs(layout);
    let table = |stage: Stage| {
        tables
            .iter()
            .find(|(of, _)| *of == stage)
            .map(|&(_, table)| table)
    };
    let texture_of =
        |slot: SlotRef| table(slot.stage).and_then(|table| table.textures.get(&slot.index));
    let sampler_of = |slot: SlotRef| {
        table(slot.stage)
            .and_then(|table| table.samplers.get(&slot.index))
            .map(|(state, clamps)| super::binding::bound_sampler(state, *clamps))
    };
    resolve(
        &pairs,
        handles,
        |slot| texture_of(slot).map_or(AddressModeMap::IDENTITY, Texture::address_mode_map),
        sampler_of,
        bound.chain(resident()).map(Texture::address_mode_map),
    )
}
