//! The Vulkan objects of one compiled variant of a pipeline state: its shader modules (one per
//! stage), immutable samplers, descriptor-set layouts and pipeline layout, taken when the variant
//! compiles; a `VkPipeline` per region specialization of a compute variant, or per graphics key
//! of a render variant (`graphics.rs`), made when a command first needs it; a command's set 0
//! from its command buffer's pools (`descriptor_arena.rs`); and its hold on the device's heap set
//! of its set-1 layout (`heap_sets.rs`), taken at its first command.
//!
//! Each object owns its device handle and is destroyed when its last holder drops it: a command
//! buffer retains what its commands use until it completes. A shader module is the device's,
//! shared by every variant whose stage has its exact SPIR-V (`content_keyed.rs`); so are the
//! immutable samplers, by their descriptions (`sampler_set`), and the set layouts and pipeline
//! layout, by their exact definition (`shared_layouts.rs`). Only the device's maps destroy those:
//! a variant's are shares.

use super::content_keyed::{program_call, ContentKeyed, Owned};
use super::graphics::Notices;
use super::kept::{Compiled, Instance, KeptObjects, KeptPipeline, Place};
use super::libraries::Linked;
use super::shared_layouts::{settle_layouts, shared_layout, SharedLayout};
use super::RegionKey;
use crate::binding::{Info, SetWrite};
use crate::degradation::{
    tested_slots_of, BoundFormats, BoundNotice, CompiledPipeline, Kinds, Occurrence, TestedFormats,
};
use crate::descriptor_arena::{DescriptorArena, SetNeeds};
use crate::device::{lock, DeviceShared};
use crate::error::{internal, vulkan, Error};
use crate::exit_guard;
use crate::generated_commands::INDIRECT_BINDABLE;
use crate::heap_sets::{self, HeapSet};
use crate::release;
use crate::shader_cache::driver::Making;
use crate::ycbcr_device::YCbCrSampler;
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{DispatchRegion, DynamicViewsModule};
use metal2vulkan::cache::{Hash256, SpecializationKey};
use metal2vulkan::error::{Feature, RuntimeDriverDefect, Unsupported};
use metal2vulkan::pipeline::{
    DescriptorSetLayout, PipelineLayout, RenderPipeline, SlotRef, StageModule,
};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// What a compute `VkPipeline` is made of: a stage's module, and the specialization, stage
/// flags and required subgroup size it is created under, with where its specialized notices
/// are announced (`Device::degrade_specialized`; by the command that makes it, naming what it binds
/// where it knows that, or by the dispatch that handed it to the pool).
struct ComputeStage<'a> {
    object: &'a StageObject,
    specialization: &'a metal2vulkan::cache::SpecializationKey,
    stage_flags: vk::PipelineShaderStageCreateFlags,
    required_subgroup_size: Option<u32>,
    notices: Notices<'a>,
}

/// One stage's shader module and entry point.
#[derive(Clone)]
pub(crate) struct StageObject {
    pub(super) stage: vk::ShaderStageFlags,
    pub(super) module: Arc<ShaderModule>,
    pub(super) entry: CString,
}

/// The immutable samplers of a layout, one per binding that bakes one, in the layout's order of
/// sets and bindings: the device's set of those samplers, shared by every variant instance whose
/// layout bakes the same ones ([`SamplerSets`]), or a host step's own. Counted against the
/// device's limit from their creation. It keeps no device functions and destroys nothing itself:
/// only the device's map destroys it, through the device, its count given back, where a program's
/// call settles the map once nothing else holds it, at a sweep for a host step's own, or at the
/// device's teardown (`content_keyed.rs`).
pub(crate) struct SamplerSet {
    samplers: Vec<vk::Sampler>,
}

impl Owned<DeviceShared> for SamplerSet {
    fn destroy(&self, shared: &DeviceShared) {
        self.release(shared);
        if !self.samplers.is_empty() {
            shared.counts.sampler_set_destroyed();
        }
    }
}

impl SamplerSet {
    /// Its samplers, one per binding that bakes one, in its layout's order.
    pub(super) fn samplers(&self) -> &[vk::Sampler] {
        &self.samplers
    }

    /// Destroy its samplers, giving their count back: the set's destruction, or a set whose
    /// creation failed part way (never made, so never counted destroyed).
    fn release(&self, shared: &DeviceShared) {
        for &sampler in &self.samplers {
            // SAFETY: the map lets a set go only when nothing else shares it, so no layout uses
            // it (a layout's objects hold the set until they destroyed the layout) and no pending
            // command does (a command buffer retains the objects that hold it until it
            // completes); or at the device's teardown, when no variant and no command buffer is
            // left (each holds the device).
            if unsafe { release::object(shared.vk(), sampler) } {
                shared.sampler_destroyed();
            }
        }
    }
}

/// The device's immutable sampler sets by the descriptors of their samplers
/// (`DeviceShared::sampler_sets`).
pub(crate) type SamplerSets = ContentKeyed<Vec<metal2vulkan::vk::SamplerDesc>, SamplerSet>;

/// The device's set of the samplers `descs` describes, made now where it holds none, and whether
/// it was made. In a program's call (`content_keyed::program_call`) a set made here enters the
/// device's map; a host step, a pool piece or another runtime thread only looks it up, and on a
/// miss makes a set of its own that no lookup finds, gone at a later sweep once nothing else holds
/// it: each call here sweeps. A layout that bakes no sampler takes an empty set, counted nowhere
/// and never in the map.
pub(crate) fn sampler_set(
    device: &Device,
    descs: Vec<metal2vulkan::vk::SamplerDesc>,
) -> Result<(Arc<SamplerSet>, bool), Error> {
    if descs.is_empty() {
        let empty = SamplerSet {
            samplers: Vec::new(),
        };
        return Ok((Arc::new(empty), false));
    }
    // A warm pass makes no sampler, nor holds the device's (`warm/pass.rs`): either would count
    // against the device's limit at a moment the program's order does not decide.
    if crate::warm::warming() {
        return Err(crate::warm::translation_only());
    }
    let shared = device.shared();
    // The device's Y′CbCr conversion of each sampler of a planes image (`texture/ycbcr_planes.rs`),
    // the one its view is made with, so the two are identically defined: made before the sets'
    // lock, as it takes the owned kernels' own.
    let conversions = descs
        .iter()
        .map(|desc| {
            desc.ycbcr
                .map(|sampler| {
                    let ycbcr = YCbCrSampler::planes(device, sampler).ok_or_else(|| {
                        internal("a layout bakes a planes sampler the device does not convert")
                    })?;
                    device.made_ycbcr_conversion(ycbcr)
                })
                .transpose()
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let program = program_call();
    // The lock is held across the creation, so two of the program's calls never make two sets of
    // the same samplers.
    let mut sets = lock(&shared.sampler_sets);
    let made = sets.get_or_make(descs.clone(), program, || {
        let mut set = SamplerSet {
            samplers: Vec::with_capacity(descs.len()),
        };
        for (desc, conversion) in descs.iter().zip(&conversions) {
            let mut chained = vk::SamplerYcbcrConversionInfo::default();
            let mut reduction = vk::SamplerReductionModeCreateInfo::default();
            let mut border = vk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
            let mut info =
                metal2vulkan::ash_interop::sampler_create_info(desc, &mut reduction, &mut border);
            if let Some(conversion) = *conversion {
                chained = chained.conversion(conversion);
                info = info.push_next(&mut chained);
            }
            // SAFETY: `info` and its chain are live for the call. Counted against the device's
            // limit (`DeviceShared::create_sampler`).
            match unsafe { shared.create_sampler(&info) } {
                Ok(sampler) => set.samplers.push(sampler),
                Err(error) => {
                    // Those made go back, and their count with them: no holder has the set.
                    set.release(shared);
                    return Err(error);
                }
            }
        }
        Ok(set)
    });
    let gone = sets.sweep();
    drop(sets);
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
    let (set, made) = made?;
    match made {
        true => shared.counts.sampler_set_made(),
        false => shared.counts.sampler_set_shared(),
    }
    Ok((set, made))
}

/// Let go of the device's sampler sets no variant holds any more (`content_keyed.rs`): in a
/// program's call only, where the kept variants applied their queue.
fn settle_sampler_sets(device: &Device) {
    debug_assert!(
        !super::pool::in_piece(),
        "the device's sampler sets settled in a pipeline pool's piece"
    );
    // A host step's variant settles nothing, as it takes nothing back (`Kept::take`).
    if !program_call() {
        return;
    }
    let shared = device.shared();
    let gone = lock(&shared.sampler_sets).settle();
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
}

impl StageObject {
    /// The stage's `VkShaderModule`.
    pub(super) fn module(&self) -> vk::ShaderModule {
        self.module.raw
    }
}

/// One `VkShaderModule`: the device's module of its SPIR-V, shared by every variant whose stage
/// has those exact words ([`ShaderModules`]), or a host step's own. It keeps no device functions
/// and has no destroying `Drop`: a holder's `Arc` is a share, and only the device's map destroys
/// it ([`Owned::destroy`]), once nothing else shares it or at the device's teardown
/// (`content_keyed.rs`), so a holder that outlives the device's teardown (a variant whose last
/// device handle goes before its stages) calls nothing. On MoltenVK a module keeps the libraries
/// its SPIR-V converted to, so every variant and state that shares it converts and compiles its
/// MSL once.
pub(crate) struct ShaderModule {
    raw: vk::ShaderModule,
}

impl Owned<DeviceShared> for ShaderModule {
    fn destroy(&self, shared: &DeviceShared) {
        // SAFETY: the map lets a module go only when nothing else shares it, so no pipeline is
        // being made from it (a making holds its variant's objects) and no pending command needs
        // it (a pipeline does not need its module once made); or at the device's teardown, when
        // no variant and no command buffer is left (each holds the device).
        unsafe { release::object(shared.vk(), self.raw) };
    }
}

/// The device's shader modules by the SHA-256 of their exact SPIR-V ([`words_digest`],
/// `DeviceShared::shader_modules`).
pub(crate) type ShaderModules = ContentKeyed<Hash256, ShaderModule>;

/// The SHA-256 of SPIR-V `words` (their length, then their bytes): what names a stage's module in
/// the device's map and its words in its makings' records, taken once per stage of a variant.
///
/// The only derivation of such a digest there is. A creation that holds the compiled function a
/// stage's words came from asks the function for the digest it already took with this very
/// derivation (`render::stage_words_digest`), so a sweep of states over one function pair takes
/// one hash a stage rather than one a creation; what the function gives back is what this
/// returns, so nothing named by the digest moves.
pub(crate) fn words_digest(words: &[u32]) -> Hash256 {
    note_words_digested();
    Making::new("words").words(words).finish()
}

/// The digests of stage words this process derived through [`words_digest`], which
/// `tests::render_stage_digests`' count reads: a run of sibling creations over one function pair
/// must add none to what its first creation derived. Carried only where that device test is
/// compiled, since nothing else reads it.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
static WORDS_DIGESTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`WORDS_DIGESTED`].
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn words_digests_derived() -> u64 {
    WORDS_DIGESTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Count a digest of stage words derived rather than read from a compiled function.
fn note_words_digested() {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    WORDS_DIGESTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The digest of what every `VkPipeline` of a variant of `layout` and `stages` reads besides its
/// making's own state, which its makings' records are named by (`shader_cache/driver.rs`): each
/// stage's stage, entry point, stage flags, required subgroup size, specialization layout and
/// SPIR-V words (By their digest, `words`, in stage order), and the layout, its immutable samplers
/// among it.
fn variant_content(layout: &PipelineLayout, stages: &[StageModule], words: &[Hash256]) -> Hash256 {
    let mut making = Making::new("variant").text(layout);
    for (stage, words) in stages.iter().zip(words) {
        making = making
            .text(&(
                stage.stage,
                &stage.entry_point,
                stage.stage_flags,
                stage.required_subgroup_size,
                &stage.specialization,
            ))
            .digest(words);
    }
    making.finish()
}

impl VariantObjects {
    /// The description of a making of `kind` of one of its `VkPipeline`s, begun with its
    /// content ([`variant_content`]); the making adds its own state.
    pub(super) fn making(&self, kind: &str) -> Making {
        Making::new(kind).digest(&self.content)
    }

    /// The flight of a making of one of its `VkPipeline`s whose own part of the driver's compile
    /// is `own` (its specialization, a region's stage): its content ([`variant_content`]) and
    /// that, which a twin's making shares (`flight.rs`).
    pub(super) fn flight(&self, own: &dyn std::fmt::Debug) -> super::flight::FlightKey {
        Making::new("flight")
            .digest(&self.content)
            .text(own)
            .finish()
    }

    /// The digests of `pipeline`'s (its compiled pipeline's) layout and stages
    /// ([`PartsContent`]), taken once.
    pub(super) fn parts_content(&self, pipeline: &RenderPipeline) -> &PartsContent {
        self.parts_content.get_or_init(|| PartsContent {
            layout: Making::new("layout").text(&pipeline.layout).finish(),
            stages: pipeline
                .stages
                .iter()
                .zip(self.words.iter())
                .map(|(stage, words)| {
                    Making::new("stage")
                        .text(&(stage.stage, &stage.entry_point, &stage.specialization))
                        .digest(words)
                        .finish()
                })
                .collect(),
        })
    }
}

/// The device's module of `spirv`, made now where it holds none. In a program's call
/// (`content_keyed::program_call`) a module made here enters the device's map; a host step or a
/// pool piece only looks it up, and on a miss makes a module of its own that no lookup finds, so
/// the map's lookups change only in the program's calls. Such a module of its own goes at a later
/// sweep, once nothing else holds it (`content_keyed.rs`, the exception): each call here sweeps.
pub(crate) fn shader_module(
    device: &Device,
    spirv: &[u32],
    digest: &Hash256,
) -> Result<Arc<ShaderModule>, Error> {
    let shared = device.shared();
    let program = program_call();
    // The lock is held across the creation, so two of the program's calls never make two
    // modules of the same words; `vkCreateShaderModule` copies the words and compiles nothing.
    let mut modules = lock(&shared.shader_modules);
    let made = modules.get_or_make(*digest, program, || {
        let info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: the code is live SPIR-V words for the call.
        let raw = unsafe { shared.vk().create_shader_module(&info, None) }.map_err(vulkan)?;
        Ok::<_, Error>(ShaderModule { raw })
    });
    let gone = modules.sweep();
    drop(modules);
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
    let (module, made) = made?;
    match made {
        true => shared.counts.shader_module_made(),
        false => shared.counts.shader_module_shared(),
    }
    Ok(module)
}

/// Let go of the device's modules no variant holds any more (`content_keyed.rs`): on a
/// program's thread only, where the kept variants applied their queue.
fn settle_modules(device: &Device) {
    debug_assert!(
        !super::pool::in_piece(),
        "the device's shader modules settled in a pipeline pool's piece"
    );
    // A host step's variant settles nothing, as it takes nothing back (`Kept::take`).
    if !program_call() {
        return;
    }
    let shared = device.shared();
    let gone = lock(&shared.shader_modules).settle();
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
}

/// Let go of the device's graphics pipeline libraries no pipeline holds any more
/// (`libraries.rs`): in a program's call only, as for its modules.
fn settle_libraries(device: &Device) {
    if !program_call() {
        return;
    }
    let shared = device.shared();
    if let Some(libraries) = &shared.libraries {
        libraries.settle(shared);
    }
}

/// The layouts and shader modules of one variant.
pub(crate) struct VariantObjects {
    pub(super) device: Device,
    /// The stages' modules, in the pipeline's stage order.
    pub(super) stages: Vec<StageObject>,
    samplers: Arc<SamplerSet>,
    /// Its set layouts and pipeline layout: the device's of their definition
    /// (`shared_layouts.rs`).
    layout: Arc<SharedLayout>,
    /// What one set of each layout set takes from a pool, by set number.
    set_needs: Vec<SetNeeds>,
    /// Whether its draws push set 0 rather than allocate it (`pushes_set_zero`).
    push_set_zero: bool,
    /// The set-0 bindings its layout makes dynamic ([`dynamic_bindings`]), by number, ascending:
    /// each command binds its set 0 with their offsets (`binding::Dynamic`).
    dynamic: Vec<u32>,
    /// The shape of its set 0 and push-constant ranges ([`set_zero_shape`]), which a draw's
    /// record compares with the last push's (`render/bound.rs`).
    set_zero_shape: u64,
    /// Where they do, a write of each element of each set-0 sampler binding that bakes its sampler
    /// (`immutable_pushes`), pushed with every draw's writes.
    immutable_pushes: Vec<SetWrite>,
    /// The stages with notes a specialization decides, announced as each `VkPipeline` of the
    /// variant is made (`Device::degrade_specialized`).
    pub(super) specialized: Vec<StageModule>,
    /// The kinds its making may announce, which a bind tests (`announce_bound`).
    kinds: Kinds,
    /// The texture slots its specialized notes test (`degradation::tested_slots_of`): those
    /// whose bound formats a bind's notices may name.
    tested: Vec<SlotRef>,
    /// The notices its last bind announced at every occurrence, and what they were made for
    /// (`announce_bound`).
    bound: Mutex<Option<BoundNotices>>,
    /// The formats at its tested slots under which its notes of the bound formats last named
    /// nothing (`degrade_bound_formats`).
    quiet_formats: Mutex<Option<TestedFormats>>,
    /// The layout's set 1, the heap, where it has one.
    heap_layout: Option<DescriptorSetLayout>,
    /// The device's heap set of that layout, from the variant's first command that binds it
    /// (`heap_sets.rs`).
    heap: Mutex<Option<Arc<HeapSet>>>,
    /// The variant instance its objects are, where its last release keeps them for a variant of
    /// the same compiled pipeline made again (`kept.rs`); `None` destroys them.
    keep: Option<Instance>,
    /// The digest its makings' records are named by, with each making's own state
    /// ([`variant_content`]).
    content: Hash256,
    /// Each stage's words' digest ([`words_digest`]), in stage order: taken once, for its
    /// modules, its content and its parts' content.
    words: Arc<[Hash256]>,
    /// The digests its graphics pipeline libraries are named by (`graphics.rs`), from its first
    /// making linked from libraries.
    parts_content: OnceLock<PartsContent>,
}

/// The digests of a variant's layout and of each of its stages (its stage, entry point,
/// specialization layout and words), which name the libraries of its pipelines' parts
/// (`GraphicsContent::names`).
pub(crate) struct PartsContent {
    pub(crate) layout: Hash256,
    pub(crate) stages: Vec<Hash256>,
}

/// The notices a bind of a variant announced at every occurrence
/// (`VariantObjects::announce_bound`), and the specializations and tested formats they were made
/// for.
struct BoundNotices {
    made_for: Vec<(SpecializationKey, Option<TestedFormats>)>,
    notices: Arc<[BoundNotice]>,
}

impl BoundNotices {
    /// Whether the notices were made for `specializations`: the same specializations in the
    /// same order, each with the same formats at the variant's tested slots, or with none known
    /// where none were.
    fn made_for(&self, specializations: &[(&SpecializationKey, Option<BoundFormats<'_>>)]) -> bool {
        self.made_for.len() == specializations.len()
            && self
                .made_for
                .iter()
                .zip(specializations)
                .all(|((made, tested), (key, formats))| {
                    made == *key
                        && match (tested, formats) {
                            (None, None) => true,
                            (Some(tested), Some(formats)) => {
                                tested.iter().all(|&(slot, format)| formats(slot) == format)
                            }
                            _ => false,
                        }
                })
    }
}

impl Drop for VariantObjects {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // A kept instance's objects go to the device's kept variants. The objects own no Vulkan
        // object of their own: their shares of the device's modules, layouts and samplers go
        // with them, and only the device's maps destroy those (`content_keyed.rs`).
        let Some(instance) = self.keep else {
            return;
        };
        let objects = KeptObjects {
            stages: self.stages.clone(),
            samplers: Arc::clone(&self.samplers),
            layout: Arc::clone(&self.layout),
            content: self.content,
            words: Arc::clone(&self.words),
        };
        lock(&self.device.shared().kept).objects_released(instance, objects);
    }
}

/// One `VkPipeline` of a variant: the one a compute variant's regions of one specialization
/// share, or a render variant's of one graphics key.
pub(crate) struct RawPipeline {
    pub(super) device: Device,
    pub(crate) raw: vk::Pipeline,
    /// How long the driver took to create it (an execution record's, `execution.rs`); zero for
    /// one taken back from a released variant (`kept.rs`).
    pub(crate) created_in: std::time::Duration,
    /// The variant instance whose layout made it and its place there, where its last release
    /// keeps it for that instance (`kept.rs`); `None` destroys it.
    pub(super) keep: Option<(Instance, Place)>,
    /// Made at its state's creation, its specialized notices not announced yet: the first draw
    /// that takes it announces them, with what it binds (`RenderVariant::first_use`).
    pub(super) unannounced: AtomicBool,
    /// The libraries it was linked from, held while it lives (`libraries.rs`); `None` for a
    /// pipeline made whole.
    pub(super) linked: Option<Linked>,
}

impl Drop for RawPipeline {
    fn drop(&mut self) {
        // As for `VariantObjects`.
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        if let Some((instance, place)) = self.keep.clone() {
            let shared = self.device.shared();
            let kept = KeptPipeline {
                place,
                raw: self.raw,
                unannounced: self.unannounced.load(Ordering::Acquire),
                linked: self.linked.clone(),
            };
            lock(&shared.kept).pipeline_released(instance, kept);
            return;
        }
        // SAFETY: as for `VariantObjects`.
        unsafe { release::object(self.device.shared().vk(), self.raw) };
    }
}

impl RawPipeline {
    /// `kept`, a `VkPipeline` of `instance` a released variant kept, taken back.
    pub(crate) fn taken_back(
        device: &Device,
        kept: KeptPipeline,
        instance: Instance,
    ) -> RawPipeline {
        RawPipeline {
            device: device.clone(),
            raw: kept.raw,
            created_in: std::time::Duration::ZERO,
            keep: Some((instance, kept.place)),
            unannounced: AtomicBool::new(kept.unannounced),
            linked: kept.linked,
        }
    }

    /// Whether its specialized notices wait for this, its first draw: once true, then false.
    pub(crate) fn first_drawn(&self) -> bool {
        self.unannounced.swap(false, Ordering::AcqRel)
    }

    /// The pipeline, kept at `place` of `objects`' instance once released, where `objects` are
    /// kept (`kept.rs`).
    pub(crate) fn kept_at(mut self, objects: &VariantObjects, place: Place) -> RawPipeline {
        self.keep = objects.keep.map(|instance| (instance, place));
        self
    }
}

impl VariantObjects {
    /// The shader modules of `stages` and the layouts of `layout`, a compiled pipeline's. Sets
    /// are numbered by position (the layout's set `n` is at index `n`), which the compile layer
    /// guarantees. Once they are made, the device announces what the stages record: their
    /// precision notes (but those a specialization decides, announced with each `VkPipeline`), and
    /// reconvergence that is not Metal's (`degradation.rs`), but the kinds it announces at every
    /// occurrence, which each bind announces (`announce_bound`); what the variant's `pipeline`
    /// records beside its stages counts among its kinds (`Device::degrade_pipeline`).
    pub(crate) fn create(
        device: &Device,
        layout: &PipelineLayout,
        stages: &[StageModule],
        pipeline: CompiledPipeline<'_>,
        digests: Option<&[Hash256]>,
    ) -> Result<VariantObjects, Error> {
        let push_set_zero = pushes_set_zero(device, layout, stages, pipeline);
        let dynamic = dynamic_bindings(device, layout, push_set_zero);
        let set_needs = needs_of(device, layout, push_set_zero, &dynamic)?;
        // One sampler per binding that bakes one, in the layout's order: the live set of the same
        // samplers where it holds one, else one made now (`sampler_set`).
        let descs: Vec<metal2vulkan::vk::SamplerDesc> = layout
            .sets
            .iter()
            .flat_map(|set| &set.bindings)
            .filter_map(|binding| binding.immutable_sampler)
            .collect();
        let (samplers, made) = sampler_set(device, descs)?;
        if made && exit_guard::in_host_step() {
            // A host step's variant whose samplers the device's map holds no set of: counted
            // where the step runs (the open break).
            device.shared().counts.host_step_sampler_set_made();
        }
        // Each stage's words' digest: the caller's where it has them, which is where it holds
        // the compiled functions the stages' words came from and they already carry the digest
        // (`render::stage_words_digest`); else taken here, one hash over every word of every
        // stage.
        let words: Arc<[Hash256]> = match digests {
            Some(digests) if digests.len() == stages.len() => Arc::from(digests),
            // A caller that gives digests gives one per stage, in stage order. Anything else is
            // a defect of its own, never a reason to name a module by another's content.
            Some(_) => return Err(internal("a variant's stage digests are not its stages'")),
            None => stages
                .iter()
                .map(|stage| words_digest(&stage.spirv))
                .collect(),
        };
        let shared_layout = shared_layout(device, layout, &samplers, (push_set_zero, &dynamic))?;
        let set_zero_shape = set_zero_shape(layout, push_set_zero, &dynamic);
        let mut objects = VariantObjects {
            device: device.clone(),
            stages: Vec::with_capacity(stages.len()),
            samplers,
            layout: shared_layout,
            set_needs,
            push_set_zero,
            dynamic,
            set_zero_shape,
            immutable_pushes: Vec::new(),
            specialized: stages
                .iter()
                .filter(|s| !s.specialized_notes.is_empty())
                .cloned()
                .collect(),
            heap_layout: layout.sets.get(1).cloned(),
            heap: Mutex::new(None),
            kinds: Kinds::of(stages, pipeline, understates_support(device)),
            tested: tested_slots_of(stages),
            bound: Mutex::new(None),
            quiet_formats: Mutex::new(None),
            keep: None,
            content: variant_content(layout, stages, &words),
            words: Arc::clone(&words),
            parts_content: OnceLock::new(),
        };
        for (stage, words) in stages.iter().zip(words.iter()) {
            let entry = CString::new(stage.entry_point.as_str())
                .map_err(|_| internal("an entry point name holds NUL"))?;
            let module = shader_module(device, &stage.spirv, words)?;
            objects.stages.push(StageObject {
                stage: stage.stage.into(),
                module,
                entry,
            });
        }
        if push_set_zero {
            objects.immutable_pushes = immutable_pushes(layout, &objects.samplers.samplers)?;
        }
        device.degrade_stages(stages, Occurrence::Made);
        Ok(objects)
    }

    /// [`create`](Self::create)'s objects of a variant of `compiled`, kept when released: those
    /// of a released instance of it taken back from the device's kept variants with the
    /// `VkPipeline`s that instance made (each at its place), else made now as a new instance
    /// (`kept.rs`).
    pub(crate) fn kept_or_create(
        device: &Device,
        compiled: Compiled,
        layout: &PipelineLayout,
        stages: &[StageModule],
        pipeline: CompiledPipeline<'_>,
        digests: Option<&[Hash256]>,
    ) -> Result<(VariantObjects, Vec<(Place, RawPipeline)>), Error> {
        let push_set_zero = pushes_set_zero(device, layout, stages, pipeline);
        let dynamic = dynamic_bindings(device, layout, push_set_zero);
        let set_needs = needs_of(device, layout, push_set_zero, &dynamic)?;
        let shared = device.shared();
        // A warm pass takes nothing back and keeps nothing (`warm/pass.rs`): the kept variants'
        // order is the program's (the owner's HITCH rule 1), and its objects go as it lets them.
        let warming = crate::warm::warming();
        let back = match warming {
            true => None,
            false => lock(&shared.kept).take(shared, compiled),
        };
        // The kept variants' queue applied (an evicted instance's modules let go): the device's
        // modules nothing holds now go, in the program's order.
        settle_modules(device);
        // And the device's pipeline layouts nothing holds now, then the sampler sets nothing
        // holds now, the evicted instances' layouts gone.
        settle_layouts(device);
        settle_sampler_sets(device);
        // And the device's graphics pipeline libraries nothing holds now.
        settle_libraries(device);
        let Some(back) = back else {
            let mut objects = VariantObjects::create(device, layout, stages, pipeline, digests)?;
            objects.keep = (!warming).then(|| Instance::new(compiled));
            return Ok((objects, Vec::new()));
        };
        device.shared().counts.variant_taken_back();
        let instance = back.instance;
        let immutable_pushes = match push_set_zero {
            true => immutable_pushes(layout, &back.objects.samplers.samplers)?,
            false => Vec::new(),
        };
        let objects = VariantObjects {
            device: device.clone(),
            stages: back.objects.stages,
            samplers: back.objects.samplers,
            layout: back.objects.layout,
            set_needs,
            push_set_zero,
            set_zero_shape: set_zero_shape(layout, push_set_zero, &dynamic),
            dynamic,
            immutable_pushes,
            specialized: stages
                .iter()
                .filter(|s| !s.specialized_notes.is_empty())
                .cloned()
                .collect(),
            heap_layout: layout.sets.get(1).cloned(),
            heap: Mutex::new(None),
            kinds: Kinds::of(stages, pipeline, understates_support(device)),
            tested: tested_slots_of(stages),
            bound: Mutex::new(None),
            quiet_formats: Mutex::new(None),
            keep: Some(instance),
            content: back.objects.content,
            words: back.objects.words,
            parts_content: OnceLock::new(),
        };
        let pipelines = back
            .pipelines
            .into_iter()
            .map(|kept| {
                let place = kept.place.clone();
                (place, RawPipeline::taken_back(device, kept, instance))
            })
            .collect();
        device.degrade_stages(stages, Occurrence::Made);
        Ok((objects, pipelines))
    }

    /// The kinds its making may announce (`Kinds::of`), which a bind tests.
    pub(crate) fn kinds(&self) -> &Kinds {
        &self.kinds
    }

    /// The format `formats` gives at each texture slot a specialized note of its stages tests:
    /// what a bind's notices name, taken while the bind's tables live.
    pub(crate) fn tested_formats(&self, formats: BoundFormats<'_>) -> TestedFormats {
        self.tested
            .iter()
            .map(|&slot| (slot, formats(slot)))
            .collect()
    }

    /// Announce what the notes of the bound formats of `stages`, the variant's, name at a command
    /// that binds `formats` (`Device::degrade_bound_formats`), but where the formats at its tested
    /// slots are those under which they last named nothing (`quiet_formats`): what they name is a
    /// function of those formats, of its stages, and of the kinds the device has yet to announce,
    /// which only shrink, so they name nothing again. A command whose formats are unknown (an
    /// indirect command's) announces as before.
    pub(crate) fn degrade_bound_formats(
        &self,
        stages: &[StageModule],
        formats: Option<BoundFormats<'_>>,
    ) {
        let Some(formats) = formats else {
            self.device.degrade_bound_formats(stages, None);
            return;
        };
        let quiet = lock(&self.quiet_formats)
            .as_ref()
            .is_some_and(|quiet| quiet.iter().all(|&(slot, format)| formats(slot) == format));
        if quiet {
            return;
        }
        if !self.device.degrade_bound_formats(stages, Some(formats)) {
            *lock(&self.quiet_formats) = Some(self.tested_formats(formats));
        }
    }

    /// Announce, at a dispatch's or draw's bind of the variant, the notices its making announces
    /// whose kinds the device's options ask for at every occurrence (`DeviceOptions::
    /// announce_each`), on the calling thread, with the making's object and reason
    /// (`Device::bound_notices`): `stages` are the variant's, `pipeline` its compiled pipeline,
    /// and `specializations` the specializations and bound formats of the `VkPipeline`s the bind
    /// takes. Its kinds are tested first (`Device::binds_each`), so a variant none of whose kinds
    /// is asked for costs an AND a word. The notices are made once and announced at each bind
    /// until the specializations, or the formats bound at the slots its notes test, change:
    /// they are a function of those alone, the variant's stages and pipeline fixed.
    pub(crate) fn announce_bound<'a>(
        &self,
        (stages, pipeline): (&[StageModule], CompiledPipeline<'_>),
        specializations: impl IntoIterator<Item = (&'a SpecializationKey, Option<BoundFormats<'a>>)>,
    ) {
        let device = &self.device;
        if !device.binds_each(&self.kinds) {
            return;
        }
        let specializations: Vec<_> = specializations.into_iter().collect();
        let notices = {
            let mut bound = lock(&self.bound);
            match bound.as_ref().filter(|b| b.made_for(&specializations)) {
                Some(b) => Arc::clone(&b.notices),
                None => {
                    device.shared().counts.bound_notices_made();
                    let notices: Arc<[BoundNotice]> = device
                        .bound_notices((stages, pipeline), &specializations)
                        .into();
                    let made_for = specializations
                        .iter()
                        .map(|&(s, f)| (s.clone(), f.map(|f| self.tested_formats(f))))
                        .collect();
                    *bound = Some(BoundNotices {
                        made_for,
                        notices: Arc::clone(&notices),
                    });
                    notices
                }
            }
        };
        // Announced with no lock held: the device's sink may call back.
        device.announce_bound(&notices);
    }

    /// The pipeline layout: the device's of its definition, which every variant of the same
    /// definition shares (`shared_layouts.rs`).
    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout.layout()
    }

    /// How many sets the layout has.
    pub(crate) fn set_count(&self) -> usize {
        self.layout.set_layouts().len()
    }

    /// Whether its draws push set 0 rather than allocate it (`pushes_set_zero`).
    pub(crate) fn pushes_set_zero(&self) -> bool {
        self.push_set_zero
    }

    /// The shape of its set 0 and push-constant ranges ([`set_zero_shape`]).
    pub(crate) fn set_zero_shape(&self) -> u64 {
        self.set_zero_shape
    }

    /// The set-0 bindings its layout makes dynamic, by number, ascending ([`dynamic_bindings`]).
    pub(crate) fn dynamic_bindings(&self) -> &[u32] {
        &self.dynamic
    }

    /// The writes its draws push for set 0's baked samplers, where they push set 0.
    pub(crate) fn immutable_pushes(&self) -> &[SetWrite] {
        &self.immutable_pushes
    }

    /// Whether its layout bakes an immutable sampler.
    pub(crate) fn bakes_samplers(&self) -> bool {
        !self.samplers.samplers.is_empty()
    }

    /// The layout of the set 0 its commands allocate: none where [`set_zero`](Self::set_zero)
    /// allocates none.
    pub(crate) fn set_zero_layout(&self) -> Option<vk::DescriptorSetLayout> {
        match (self.layout.set_layouts().first(), self.set_needs.first()) {
            (Some(&layout), Some(needs)) if !needs.is_empty() => Some(layout),
            _ => None,
        }
    }

    /// A command's set 0, from its command buffer's pools `arena`; none when the layout has no
    /// set, set 0 no descriptor (a pipeline that uses no set needs none bound), or its commands
    /// push set 0.
    pub(crate) fn set_zero(
        &self,
        arena: &mut DescriptorArena,
    ) -> Result<Option<vk::DescriptorSet>, Error> {
        match (self.layout.set_layouts().first(), self.set_needs.first()) {
            (Some(&layout), Some(needs)) if !needs.is_empty() => {
                Ok(Some(arena.allocate(&self.device, layout, needs)?))
            }
            _ => Ok(None),
        }
    }

    /// The device's heap set of the layout's set 1 (`heap_sets.rs`), which the variant holds
    /// from its first command that binds it; refused for a layout without one (`Internal`).
    pub(crate) fn heap_set(&self) -> Result<Arc<HeapSet>, Error> {
        let mut held = lock(&self.heap);
        if let Some(set) = held.as_ref() {
            return Ok(Arc::clone(set));
        }
        let layout = self
            .heap_layout
            .as_ref()
            .ok_or_else(|| internal("a heap set of a layout without one"))?;
        let set = heap_sets::heap_set(&self.device, layout)?;
        *held = Some(Arc::clone(&set));
        Ok(set)
    }

    /// The `VkPipeline` of `key`'s specialization, stage flags, and required subgroup size, of a
    /// compute variant (one stage), its specialized notices announced as `notices` says.
    pub(crate) fn pipeline(
        &self,
        key: &RegionKey,
        notices: Notices<'_>,
    ) -> Result<RawPipeline, Error> {
        self.compute_pipeline(key, false, notices)
    }

    /// The `VkPipeline` of a render variant's `COMPUTE` stage at `index` of its stages, an
    /// emulated object or mesh function's (`module`, the variant's stage there), under a mesh
    /// draw's complete `specialization` (`DrawCommands::specialization`), which binds `formats`.
    pub(crate) fn emulated_pipeline(
        &self,
        index: usize,
        module: &StageModule,
        (specialization, formats): (&metal2vulkan::cache::SpecializationKey, BoundFormats<'_>),
    ) -> Result<RawPipeline, Error> {
        let object = self
            .stages
            .get(index)
            .filter(|s| s.stage == vk::ShaderStageFlags::COMPUTE)
            .ok_or_else(|| internal("an emulated mesh pipeline without its compute stage"))?;
        let stage = ComputeStage {
            object,
            specialization,
            stage_flags: module.stage_flags.into(),
            required_subgroup_size: module.required_subgroup_size,
            notices: Notices::Now(Some(formats)),
        };
        self.made_pipeline(&stage, false)
    }

    /// [`pipeline`](Self::pipeline), created with `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`
    /// for an indirect execution set (`generated_execute.rs`): on a device that enabled
    /// `VK_EXT_device_generated_commands` (and `maintenance5`, whose flags it takes). The
    /// replay that makes it does not know the formats its commands bind.
    pub(crate) fn indirect_bindable_pipeline(
        &self,
        region: &DispatchRegion,
    ) -> Result<RawPipeline, Error> {
        self.compute_pipeline(&RegionKey::of(region), true, Notices::Now(None))
    }

    /// The compute pipeline of `key`, indirect-bindable where `bindable` says so, its specialized
    /// notices announced as `notices` says.
    fn compute_pipeline(
        &self,
        key: &RegionKey,
        bindable: bool,
        notices: Notices<'_>,
    ) -> Result<RawPipeline, Error> {
        let [kernel] = self.stages.as_slice() else {
            return Err(internal("a compute pipeline of other than one stage"));
        };
        let stage = ComputeStage {
            object: kernel,
            specialization: &key.specialization,
            stage_flags: key.stage_flags.into(),
            required_subgroup_size: key.required_subgroup_size,
            notices,
        };
        self.made_pipeline(&stage, bindable)
    }

    /// The compute pipeline of `c`, indirect-bindable where `bindable` says so.
    fn made_pipeline(&self, c: &ComputeStage<'_>, bindable: bool) -> Result<RawPipeline, Error> {
        let map: Vec<vk::SpecializationMapEntry> = c
            .specialization
            .entries
            .iter()
            .map(|&entry| entry.into())
            .collect();
        let spec = vk::SpecializationInfo::default()
            .map_entries(&map)
            .data(&c.specialization.data);
        let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(c.required_subgroup_size.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(c.object.module())
            .name(&c.object.entry)
            .flags(c.stage_flags);
        if !map.is_empty() {
            stage = stage.specialization_info(&spec);
        }
        if c.required_subgroup_size.is_some() {
            stage = stage.push_next(&mut subgroup);
        }
        let mut flags = vk::PipelineCreateFlags2CreateInfoKHR::default().flags(INDIRECT_BINDABLE);
        let mut info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.layout.layout());
        if bindable {
            info = info.push_next(&mut flags);
        }
        // Whether the making's cache held it and its one stage, where the making reports them
        // (`shader_cache::found_in_cache`; one stage's feedback, as
        // VUID-VkComputePipelineCreateInfo-pipelineStageCreationFeedbackCount-06566 allows).
        let mut whole = vk::PipelineCreationFeedback::default();
        let mut stage = [vk::PipelineCreationFeedback::default()];
        let shared = self.device.shared();
        let driver_reports = shared.shader_cache.driver_reports();
        // Its record is named by the variant's content and what this making adds to it.
        let making = self.making("compute").text(&(
            c.object.stage.as_raw(),
            &c.object.entry,
            c.specialization,
            c.stage_flags.as_raw(),
            c.required_subgroup_size,
            bindable,
        ));
        // The turn first (none where a piece's thread holds one), once no twin's making is in the
        // air (`flight.rs`): the driver's time alone is timed, not the wait for a turn or a twin.
        let flight = self.flight(&(
            c.object.stage.as_raw(),
            &c.object.entry,
            c.specialization,
            c.stage_flags.as_raw(),
            c.required_subgroup_size,
        ));
        let (turn, leader, followed) = shared.flights.board(flight);
        if followed {
            shared.counts.flight_followed();
        }
        let started = std::time::Instant::now();
        // With what the driver reports of the making: a pipeline found in its record's cache
        // reads no data back (`shader_cache/driver.rs`).
        let created = shared
            .make_pipelines_reported(making, |cache| {
                let mut feedback = vk::PipelineCreationFeedbackCreateInfo::default()
                    .pipeline_creation_feedback(&mut whole)
                    .pipeline_stage_creation_feedbacks(&mut stage);
                let mut info = info;
                if driver_reports {
                    info = info.push_next(&mut feedback);
                }
                // SAFETY: every pointer in `info` outlives the call; the pipeline cache is the
                // making's own, destroyed once the call returned.
                let made = unsafe { shared.vk().create_compute_pipelines(cache, &[info], None) };
                let found = driver_reports
                    .then(|| crate::shader_cache::found_in_cache(&whole, &stage))
                    .flatten();
                (made, found)
            })
            .map_err(|(_, error)| vulkan(error))?;
        let created_in = started.elapsed();
        drop(leader);
        drop(turn);
        if driver_reports {
            shared.shader_cache.driver_reported(&whole, &stage);
        }
        let raw = created
            .first()
            .copied()
            .ok_or_else(|| internal("the driver made no pipeline"))?;
        self.device.shared().counts.variant_pipeline_made();
        let pipeline = RawPipeline {
            device: self.device.clone(),
            raw,
            created_in,
            keep: None,
            // Its notices wait for its first dispatch where its state's creation made it.
            unannounced: AtomicBool::new(matches!(c.notices, Notices::FirstDraw)),
            linked: None,
        };
        if let Notices::Now(formats) = c.notices {
            self.device.degrade_specialized(
                &self.specialized,
                c.specialization,
                formats,
                Occurrence::Made,
            );
        }
        Ok(pipeline)
    }
}

/// The `VkDescriptorSetLayout` of `set`, each binding with its `immutable` samplers (none where
/// its list is empty or missing) and of its type in `types` (the set's own where the list is
/// shorter). A layout the driver refuses ([`set_support`]) is refused, but where the driver runs
/// what it refuses (`RuntimeDriverDefect::SetLayoutSupportUnderstated`), whose variant's making
/// announced it (`Device::degrade_pipeline`).
pub(crate) fn create_set_layout(
    device: &Device,
    set: &DescriptorSetLayout,
    samplers_and_types: (&[&[vk::Sampler]], &[vk::DescriptorType]),
    push: bool,
) -> Result<vk::DescriptorSetLayout, Error> {
    with_create_info(set, samplers_and_types, push, |info| {
        if layout_support(device, set, info) == SetSupport::Refused {
            return Err(Error::Unsupported(Unsupported::not_implemented(
                Feature::ArgumentBufferTextures,
            )));
        }
        // SAFETY: `info` and every array it points at are live for the call.
        unsafe {
            device
                .shared()
                .vk()
                .create_descriptor_set_layout(info, None)
        }
        .map_err(vulkan)
    })
}

/// Whether the device makes `set`'s layout (as a pooled set of its own binding types), asking its
/// driver where Vulkan does not promise it ([`set_support`]).
pub(crate) fn set_layout_support(device: &Device, set: &DescriptorSetLayout) -> SetSupport {
    with_create_info(set, (&[], &[]), false, |info| {
        layout_support(device, set, info)
    })
}

/// [`set_support`] of `set`, whose create info is `info`, on `device`.
fn layout_support(
    device: &Device,
    set: &DescriptorSetLayout,
    info: &vk::DescriptorSetLayoutCreateInfo<'_>,
) -> SetSupport {
    let understated = understates_support(device);
    let total = set.bindings.iter().map(|b| u64::from(b.count)).sum();
    set_support(total, understated, || {
        let mut support = vk::DescriptorSetLayoutSupport::default();
        // SAFETY: `info` and every array it points at are live for the call, and `support` heads
        // an empty chain.
        unsafe {
            device
                .shared()
                .vk()
                .get_descriptor_set_layout_support(info, &mut support)
        };
        support.supported == vk::TRUE
    })
}

/// Whether `device`'s driver understates its descriptor set layout support
/// (`RuntimeDriverDefect::SetLayoutSupportUnderstated`): it runs layouts its
/// `vkGetDescriptorSetLayoutSupport` refuses.
fn understates_support(device: &Device) -> bool {
    (device.properties().driver_defects).contains(&RuntimeDriverDefect::SetLayoutSupportUnderstated)
}

/// Whether the runtime makes, on the device `description` names, a descriptor set layout its
/// driver's `vkGetDescriptorSetLayoutSupport` refuses ([`SetSupport::Understated`]): where the
/// driver runs what the query refuses (`RuntimeDriverDefect::SetLayoutSupportUnderstated`,
/// `driver_defects.rs`). Validation's executor makes its own heap sets by it, so its sink passes
/// the validation layer's report of such a layout there as the runtime's does.
#[doc(hidden)]
pub fn makes_layouts_past_support(description: &metal2vulkan::device::DeviceDescription) -> bool {
    crate::driver_defects::of(description)
        .contains(&RuntimeDriverDefect::SetLayoutSupportUnderstated)
}

/// The least `maxPerSetDescriptors` Vulkan allows (its Required Limits): a set layout of at most
/// that many descriptors is one `vkGetDescriptorSetLayoutSupport` answers `VK_TRUE` for on every
/// device, so only a larger one is asked about.
pub(crate) const LEAST_PER_SET_DESCRIPTORS: u64 = 1024;

/// Whether a device makes a descriptor set layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SetSupport {
    /// It makes it: Vulkan promises it, or the driver's `vkGetDescriptorSetLayoutSupport`
    /// answers `VK_TRUE`.
    Supported,
    /// The driver's query answers `VK_FALSE`, and the driver runs the layout as Metal runs the
    /// argument buffer it encodes it in (`RuntimeDriverDefect::SetLayoutSupportUnderstated`): it
    /// is made, announced (`Degradation::SetPastLayoutSupport`), and Vulkan's validation layer
    /// reports it (`VUID-vkCreateDescriptorSetLayout-support-09582`).
    Understated,
    /// The driver's query answers `VK_FALSE`: not made (`Unsupported`), as Vulkan requires
    /// (`VUID-vkCreateDescriptorSetLayout-support-09582`: "If the descriptor layout exceeds the
    /// limits reported through the physical device limits, then vkGetDescriptorSetLayoutSupport
    /// must have returned VkDescriptorSetLayoutSupport with support equal to VK_TRUE").
    Refused,
}

/// [`SetSupport`] of a layout of `total` descriptors on a device whose driver understates its
/// support (`understated`), `supported` asking the driver: a layout of at most
/// [`LEAST_PER_SET_DESCRIPTORS`] is supported unasked ("If the descriptor set layout satisfies
/// the VkPhysicalDeviceMaintenance3Properties::maxPerSetDescriptors limit, this command is
/// guaranteed to return VK_TRUE"); any other is what the driver answers.
pub(crate) fn set_support(
    total: u64,
    understated: bool,
    supported: impl FnOnce() -> bool,
) -> SetSupport {
    if total <= LEAST_PER_SET_DESCRIPTORS || supported() {
        SetSupport::Supported
    } else if understated {
        SetSupport::Understated
    } else {
        SetSupport::Refused
    }
}

/// Whether a layout of `total` descriptors may be one [`set_support`] answers
/// [`SetSupport::Understated`] for on a device whose driver understates its support
/// (`understated`): one past [`LEAST_PER_SET_DESCRIPTORS`] there, which the driver is asked about.
/// The kinds a variant may announce (`Kinds::of`) hold `Degradation::SetPastLayoutSupport` by
/// it, so a bind announces the notice whatever else the variant records.
pub(crate) fn may_pass_support(total: u64, understated: bool) -> bool {
    understated && total > LEAST_PER_SET_DESCRIPTORS
}

/// `f` of the `VkDescriptorSetLayoutCreateInfo` of `set` as [`create_set_layout`] makes it.
fn with_create_info<R>(
    set: &DescriptorSetLayout,
    (immutable, types): (&[&[vk::Sampler]], &[vk::DescriptorType]),
    push: bool,
    f: impl FnOnce(&vk::DescriptorSetLayoutCreateInfo<'_>) -> R,
) -> R {
    let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = set
        .bindings
        .iter()
        .enumerate()
        .map(|(n, binding)| {
            let ty = types.get(n).copied().unwrap_or(binding.ty.into());
            let info = vk::DescriptorSetLayoutBinding::default()
                .binding(binding.binding)
                .descriptor_type(ty)
                .descriptor_count(binding.count)
                .stage_flags(binding.stages.into());
            match immutable.get(n).filter(|samplers| !samplers.is_empty()) {
                Some(samplers) => info.immutable_samplers(samplers),
                None => info,
            }
        })
        .collect();
    let flags: Vec<vk::DescriptorBindingFlags> =
        set.bindings.iter().map(|b| b.flags.into()).collect();
    let mut flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let mut set_flags: vk::DescriptorSetLayoutCreateFlags = set.flags.into();
    if push {
        set_flags |= vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR;
    }
    let mut info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(set_flags)
        .bindings(&bindings);
    if flags.iter().any(|f| !f.is_empty()) {
        info = info.push_next(&mut flags_info);
    }
    f(&info)
}

/// The shape of `layout`'s set 0 and push-constant ranges, where its draws push set 0
/// (`push_set_zero`) or not, its bindings numbered in `dynamic` of their dynamic types: a hash of
/// what Vulkan's compatibility for set 0 compares (the set-0 `VkDescriptorSetLayout`'s create
/// info, each binding's number, type, count, stages, baked sampler and flags and the set's flags,
/// and the push-constant ranges), which two variants of the same
/// shape would create identically defined, compatible for set 0 (Pipeline Layout
/// Compatibility). Variants share a layout only where it is identically defined whole
/// (`shared_layouts.rs`), so a draw's record compares shapes to count the pushes that carry set 0
/// whole across a switch between two layouts of equal shapes (which differ in a later set, or
/// bake a host step's own samplers; `WorkCounts::set_zero_equal_shape_switches`). Taken once, as
/// the variant's objects are made.
fn set_zero_shape(layout: &PipelineLayout, push_set_zero: bool, dynamic: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    push_set_zero.hash(&mut hasher);
    if let Some(set) = layout.sets.first() {
        set.flags.hash(&mut hasher);
        for binding in &set.bindings {
            binding.binding.hash(&mut hasher);
            binding.ty.hash(&mut hasher);
            dynamic.contains(&binding.binding).hash(&mut hasher);
            binding.count.hash(&mut hasher);
            binding.stages.hash(&mut hasher);
            binding.immutable_sampler.hash(&mut hasher);
            binding.flags.hash(&mut hasher);
        }
    }
    layout.push_constants.hash(&mut hasher);
    hasher.finish()
}

/// What one set of each of `layout`'s sets takes from a pool, by set number: nothing for set 0
/// where it is pushed (`push_set_zero`), and set 0's bindings numbered in `dynamic` of their
/// dynamic types.
fn needs_of(
    device: &Device,
    layout: &PipelineLayout,
    push_set_zero: bool,
    dynamic: &[u32],
) -> Result<Vec<SetNeeds>, Error> {
    layout
        .sets
        .iter()
        .enumerate()
        .map(|(index, set)| match (push_set_zero, index) {
            (true, 0) => Ok(SetNeeds {
                counts: Vec::new(),
                flags: vk::DescriptorPoolCreateFlags::empty(),
            }),
            (false, 0) => needs_typed(set, dynamic, |sampler| {
                device.shared().ycbcr_formats.planes_descriptors(sampler)
            }),
            _ => set_needs(set),
        })
        .collect()
}

/// The set-0 bindings of `layout` a variant makes dynamic, by number, ascending: those whose
/// descriptor can sit at a set that stays as it was while only the offsets its commands bind
/// change, the offsets given as each command binds the set
/// (`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC`, `..._STORAGE_BUFFER_DYNAMIC`). None where its
/// commands push set 0 (a push descriptor set layout holds no dynamic buffer). Else each uniform
/// or storage buffer binding of one descriptor that each stage using it uses through one variable
/// (`DescriptorBinding::variables`) and no stage the pipeline adds reads (a tessellation control
/// or geometry stage, whose variables the layout does not count), in binding order while the
/// device's `maxDescriptorSetUniformBuffersDynamic` and `maxDescriptorSetStorageBuffersDynamic`
/// take them; every other binding stays of its own type.
///
/// A binding each stage using it uses through **several** variables (a slot reached through its
/// word view and its narrow or `uvec4` unit views) stays static where the device does
/// not take such a binding dynamically, which the device answers itself
/// ([`takes_multi_variable_dynamic`]): it is one driver's shader-translation defect, not a limit
/// of Vulkan, which bounds the variables at a binding nowhere.
fn dynamic_bindings(device: &Device, layout: &PipelineLayout, push_set_zero: bool) -> Vec<u32> {
    let Some(set) = layout.sets.first().filter(|_| !push_set_zero) else {
        return Vec::new();
    };
    // Asked of the device only where a binding needs the answer, so a layout of one variable a
    // binding -- every layout but the few with a slot read or written through several types --
    // compiles nothing.
    let several =
        set.bindings.iter().any(|b| b.variables > 1) && takes_multi_variable_dynamic(device);
    dynamic_numbers(set, device.shared().dynamic_buffers, several)
}

/// [`dynamic_bindings`]' choice over `set`, with `[uniform, storage]` dynamic buffers left of the
/// device's limits and `several` whether a binding of more than one variable may be dynamic.
fn dynamic_numbers(
    set: &DescriptorSetLayout,
    [mut uniform, mut storage]: [u32; 2],
    several: bool,
) -> Vec<u32> {
    let added = vk::ShaderStageFlags::TESSELLATION_CONTROL
        | vk::ShaderStageFlags::TESSELLATION_EVALUATION
        | vk::ShaderStageFlags::GEOMETRY;
    let mut dynamic = Vec::new();
    for binding in &set.bindings {
        let stages: vk::ShaderStageFlags = binding.stages.into();
        let flags: vk::DescriptorBindingFlags = binding.flags.into();
        let variables = match binding.variables {
            0 => false,
            1 => true,
            _ => several,
        };
        if binding.count != 1 || !variables || stages.intersects(added) || !flags.is_empty() {
            continue;
        }
        let left = match binding.ty.into() {
            vk::DescriptorType::UNIFORM_BUFFER => &mut uniform,
            vk::DescriptorType::STORAGE_BUFFER => &mut storage,
            _ => continue,
        };
        if *left > 0 {
            *left -= 1;
            dynamic.push(binding.binding);
        }
    }
    dynamic
}

/// Whether `device` takes a set-0 buffer binding whose entry point uses it through more than one
/// variable as a dynamic descriptor: the device's own answer, asked once and kept
/// (`DeviceShared::dynamic_multi_variable`).
///
/// Vulkan bounds the variables at a binding nowhere, dynamic or not, so this asks about the
/// driver: MoltenVK's SPIRV-Cross writes Metal that does not compile for such a binding (it
/// declares one variable's local alone). Read in 1.4.1's source, and measured on **1.4.2**
/// (Homebrew's, `driverVersion` 10402) on an Apple M3 Max, where `vkCreateComputePipelines`
/// answers `VK_ERROR_INITIALIZATION_FAILED` for a kernel of three and of four variables at one
/// dynamic binding. A driver without the defect must not pay for it -- a descriptor set of its
/// own per command -- so the device answers for itself rather than being named.
///
/// It answers by creating one compute pipeline of exactly that shape, the crate-owned self-check
/// module with one variable per unit view the device's storage views can declare, so any plan's
/// binding is a subset of it (`bind::DynamicViewsModule`), and destroying it. A creation that
/// fails, for any reason, answers no, which is what every device answered before it was asked.
/// What the driver reports inside that failed creation is its answer to the runtime's own
/// question and names none of the program's work (MoltenVK 1.4.2 reports the Metal it could not
/// compile and the failed pipeline as two errors), so the sink is handed one informational line
/// carrying those messages ([`SELF_CHECK_REFUSED`]) in place of the driver's errors; a creation
/// that succeeds hands on whatever the driver reported in it, as any other making does.
///
/// The check proves the driver **compiles** the shape. A driver that compiled it and then applied
/// the dynamic offset to one of its variables alone would still be wrong, and none is measured to
/// (`docs/FAST-PATHS.md`).
fn takes_multi_variable_dynamic(device: &Device) -> bool {
    if let Some(&answered) = device.shared().dynamic_multi_variable.get() {
        return answered;
    }
    // A host step makes no object of its own and must not compile here (`exit_guard`): its
    // variant takes the static form every device took before, and the answer stays unasked for a
    // later variant outside a step.
    if exit_guard::in_host_step() {
        return false;
    }
    *device
        .shared()
        .dynamic_multi_variable
        .get_or_init(|| multi_variable_dynamic_creates(device).unwrap_or(false))
}

/// Create the self-check pipeline of [`takes_multi_variable_dynamic`] and destroy it: whether the
/// device made it.
fn multi_variable_dynamic_creates(device: &Device) -> Result<bool, Error> {
    let module = metal2vulkan::bind::dynamic_views_module(device.compile_device())
        .map_err(Error::Compile)?;
    let shared = device.shared();
    let raw = shared.vk();
    let mut shader = vk::ShaderModule::null();
    let mut set_layout = vk::DescriptorSetLayout::null();
    let mut layout = vk::PipelineLayout::null();
    let mut pipeline = vk::Pipeline::null();
    let made = (|| {
        let info = vk::ShaderModuleCreateInfo::default().code(&module.spirv);
        // SAFETY: the code is live SPIR-V words for the call.
        shader = unsafe { raw.create_shader_module(&info, None) }.map_err(vulkan)?;
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(DynamicViewsModule::BINDING)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `info` and its array are live for the call.
        set_layout = unsafe { raw.create_descriptor_set_layout(&info, None) }.map_err(vulkan)?;
        let set_layouts = [set_layout];
        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: as above.
        layout = unsafe { raw.create_pipeline_layout(&info, None) }.map_err(vulkan)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: every pointer in `info` outlives the call; no pipeline cache, since the
        // self-check's pipeline is never made again.
        let (created, heard) = crate::device::holding(|| unsafe {
            raw.create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        });
        match created {
            Ok(made) => {
                pipeline = made.first().copied().unwrap_or(vk::Pipeline::null());
                for (sink, message) in &heard {
                    sink.deliver(message);
                }
                Ok(true)
            }
            Err((_, result)) => {
                refused_self_check(result, &heard);
                Ok(false)
            }
        }
    })();
    // SAFETY: nothing uses the objects: no command recorded them, and the pipeline, where it was
    // made, was never bound.
    unsafe {
        release::object(raw, pipeline);
        release::object(raw, layout);
        release::object(raw, set_layout);
        release::object(raw, shader);
    }
    made
}

/// The id name of the informational line a device hands its sink where the driver did not make
/// the self-check's pipeline of [`takes_multi_variable_dynamic`]: the driver's result and the
/// messages it reported inside that creation, which the sink is not handed as the driver's own.
pub(crate) const SELF_CHECK_REFUSED: &str = "metal2vulkan-self-check-refused";

/// Tell the sink that heard the driver's messages `heard` inside the self-check's creation,
/// informationally, that the driver did not make its pipeline ([`SELF_CHECK_REFUSED`]): its
/// `result`, and those messages, by id name and text, in the order the driver raised them. A
/// creation that reported nothing says nothing, as before.
fn refused_self_check(result: vk::Result, heard: &[(crate::MessageSink, crate::DebugMessage)]) {
    let Some((sink, _)) = heard.first() else {
        return;
    };
    let mut text = format!(
        "the device does not take a set-0 buffer binding of several variables as a dynamic \
         descriptor: vkCreateComputePipelines of the self-check module answered {result:?}, \
         reporting {} message(s) of its own",
        heard.len()
    );
    for (_, message) in heard {
        text.push_str(&format!("\n{}: {}", message.id_name, message.text));
    }
    crate::driver_facts::inform_through(Some(sink), SELF_CHECK_REFUSED, text);
}

/// The device's answer of [`takes_multi_variable_dynamic`], for the check that the restriction is
/// the device's and not every device's (`tests::dynamic_multi_variable`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn asks_multi_variable_dynamic(device: &Device) -> bool {
    takes_multi_variable_dynamic(device)
}

/// The descriptor types a pushed set 0 takes: those a set-0 write (`binding::Info`) carries, but
/// the dynamic buffers, inline uniform blocks and mutable descriptors a push descriptor set layout
/// cannot hold (`VkDescriptorSetLayoutCreateInfo`'s valid usage).
const PUSHED_TYPES: [vk::DescriptorType; 9] = [
    vk::DescriptorType::SAMPLER,
    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    vk::DescriptorType::SAMPLED_IMAGE,
    vk::DescriptorType::STORAGE_IMAGE,
    vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
    vk::DescriptorType::STORAGE_TEXEL_BUFFER,
    vk::DescriptorType::UNIFORM_BUFFER,
    vk::DescriptorType::STORAGE_BUFFER,
    vk::DescriptorType::INPUT_ATTACHMENT,
];

/// Whether a variant of `layout` and `stages` pushes its set 0 (`vkCmdPushDescriptorSet`) at
/// each draw or dispatch, rather than allocating a set from its command buffer's pools, writing it
/// and binding it: the same descriptors, at the same bindings. Where the device has the push
/// command (Vulkan 1.4, or `VK_KHR_push_descriptor`, `binding::push_descriptor_command`), the
/// compiled layout's set 0 fits `maxPushDescriptors` (`push_descriptor_eligible`), takes no set or
/// binding flag and only descriptor types a push descriptor set layout takes (`PUSHED_TYPES`),
/// and the variant binds set 0 at one bind point: a render variant of no compute stage at the
/// graphics bind point, its draws' (`render::Recording`), where one that emulates a mesh or
/// tessellation stage dispatches with set 0 too; a kernel's variant, of compute stages alone
/// (a `CompiledPipeline::Kernel`), at the compute bind point, its dispatches' and tile
/// dispatches' (`pending_dispatch::Recording`, `render::tile_dispatch`).
fn pushes_set_zero(
    device: &Device,
    layout: &PipelineLayout,
    stages: &[StageModule],
    pipeline: CompiledPipeline<'_>,
) -> bool {
    let compute = |stage: &StageModule| {
        let flags: vk::ShaderStageFlags = stage.stage.into();
        flags.contains(vk::ShaderStageFlags::COMPUTE)
    };
    let one_point = match pipeline {
        CompiledPipeline::Render(_) => !stages.iter().any(compute),
        CompiledPipeline::Kernel(_) => !stages.is_empty() && stages.iter().all(compute),
    };
    device.shared().push_descriptor.is_some()
        && layout.push_descriptor_eligible
        && one_point
        && layout.sets.first().is_some_and(|set| {
            let flags: vk::DescriptorSetLayoutCreateFlags = set.flags.into();
            flags.is_empty()
                && !set.bindings.is_empty()
                && set.bindings.iter().all(|binding| {
                    let flags: vk::DescriptorBindingFlags = binding.flags.into();
                    flags.is_empty() && PUSHED_TYPES.contains(&binding.ty.into())
                })
        })
}

/// A write of each element of each sampler binding of `layout`'s set 0 that bakes its sampler, of
/// the sampler `samplers` holds for it (one per binding that bakes one, in the layout's order of
/// sets and bindings, as `VariantObjects::create` takes them). A pushed set takes them, which a
/// pool-allocated one holds from its layout. The Vulkan specification, `vkCmdPushDescriptorSet`:
/// "This includes immutable sampler descriptors, which must be pushed before they are accessed by
/// a pipeline (the immutable samplers are pushed, rather than the samplers in pDescriptorWrites)."
/// A combined image sampler binding that bakes its sampler takes the draw's image write, whose
/// sampler Vulkan ignores there.
fn immutable_pushes(
    layout: &PipelineLayout,
    samplers: &[vk::Sampler],
) -> Result<Vec<SetWrite>, Error> {
    let mut baked = samplers.iter().copied();
    let mut pushes = Vec::new();
    for (index, set) in layout.sets.iter().enumerate() {
        for binding in set
            .bindings
            .iter()
            .filter(|b| b.immutable_sampler.is_some())
        {
            let sampler = baked
                .next()
                .ok_or_else(|| internal("a sampler set shorter than its layout"))?;
            let ty: vk::DescriptorType = binding.ty.into();
            if index == 0 && ty == vk::DescriptorType::SAMPLER {
                pushes.extend((0..binding.count).map(|element| SetWrite {
                    binding: binding.binding,
                    element,
                    ty,
                    info: Info::Image(vk::DescriptorImageInfo::default().sampler(sampler)),
                }));
            }
        }
    }
    Ok(pushes)
}

/// What a pool that allocates one set of `set` needs: each binding's count summed per
/// descriptor type, since a set takes every binding's whole count; and `UPDATE_AFTER_BIND` when
/// the set layout is `UPDATE_AFTER_BIND_POOL`.
pub(crate) fn set_needs(set: &DescriptorSetLayout) -> Result<SetNeeds, Error> {
    needs_typed(set, &[], |_| 1)
}

/// [`set_needs`] of `set` with its bindings numbered in `dynamic` of their dynamic types, a
/// combined image sampler binding whose immutable sampler is a planes image's Y′CbCr conversion
/// (`SamplerDesc::ycbcr`) taking `planes` of its sampler descriptors for each of its count
/// (`combinedImageSamplerDescriptorCount`, `ycbcr_device.rs`).
fn needs_typed(
    set: &DescriptorSetLayout,
    dynamic: &[u32],
    planes: impl Fn(metal2vulkan::bind::YCbCrPlanesSampler) -> u32,
) -> Result<SetNeeds, Error> {
    let mut counts: std::collections::BTreeMap<i32, u32> = std::collections::BTreeMap::new();
    let mut flags = vk::DescriptorPoolCreateFlags::empty();
    let set_flags: vk::DescriptorSetLayoutCreateFlags = set.flags.into();
    if set_flags.contains(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL) {
        flags |= vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
    }
    for binding in &set.bindings {
        let ty: vk::DescriptorType = binding.ty.into();
        let ty = match dynamic.contains(&binding.binding) {
            true => crate::binding::dynamic_type(ty),
            false => ty,
        };
        let each = match binding.immutable_sampler.and_then(|desc| desc.ycbcr) {
            Some(sampler) => planes(sampler),
            None => 1,
        };
        let count = counts.entry(ty.as_raw()).or_default();
        *count = binding
            .count
            .checked_mul(each)
            .and_then(|taken| count.checked_add(taken))
            .ok_or_else(|| internal("the layout's descriptor count overflows"))?;
    }
    Ok(SetNeeds {
        counts: counts.into_iter().filter(|&(_, count)| count > 0).collect(),
        flags,
    })
}
