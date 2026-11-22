//! The census of compiled and executed modules: what a module records of the product's
//! selections, and what a candidate run records of the modules it executed and where.
//!
//! A module's census ([`ModuleCensus`]) is its stage, the subgroup requirement its simd tier states
//! ([`SubgroupShape`]), its `StrategyCensus` names, its precision notes, and the containment
//! encodings it compiled under ([`Containment`]). The containment encoding is taken from what the
//! module compiled for, never from its census names, which do not all show it. The census names
//! an access that reaches buffer memory by device address apart from one through a descriptor view
//! (`.psb`, `.psb_and_view`: the product's `access::BufferReach`), and a side that may be an arena
//! (`.arena`, `access::SideReach`), with a memory intrinsic's destination and source named apart
//! (`.to_<side>`, `.from_<side>`): so the name and the module's encoding together say which check
//! each access took.
//!
//! A run's census ([`ExecutedCensus`]) is the class of the device it ran on ([`ExecutionClass`]:
//! a CPU ICD, or a GPU's driver, [`HostClass`]), the product build that compiled its modules, and
//! their censuses. The tier table compiles each committed profile twice where it states a
//! robustness feature ([`Leg`]): as stated, which selects the robust image encodings where it
//! states image robustness, and with every robustness feature removed ([`robustness_off`]), which
//! selects the in-shader ones. Buffers are checked in the shader on both legs.

use crate::candidate::engine::artifact::ExecutedArtifact;
use crate::case::Stage;
use crate::observation_contract::ComparisonClass;
use metal2vulkan::bind::BindingPlan;
use metal2vulkan::device::VkFeature;
use metal2vulkan::device::{DeviceDescription, DeviceProfile};
use metal2vulkan::pipeline::{PrecisionNote, StageModule};
use metal2vulkan::vk;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod tier_table;

/// What a module asks of the device's subgroups: the requirement its simd tier states, carried by
/// `StageModule::required_subgroup_size` and `REQUIRE_FULL_SUBGROUPS`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(deny_unknown_fields)]
pub struct SubgroupShape {
    /// `requiredSubgroupSize`, when the tier pins one.
    pub required_size: Option<u32>,
    /// Full subgroups are required.
    pub full_subgroups: bool,
}

impl SubgroupShape {
    /// The shape of a linked stage module.
    pub fn of_stage_module(module: &StageModule) -> SubgroupShape {
        SubgroupShape {
            required_size: module.required_subgroup_size,
            full_subgroups: module
                .stage_flags
                .contains(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS),
        }
    }
}

/// The robustness features a tier-table profile's robustness-off leg removes
/// ([`Leg::RobustnessOff`]). Image robustness drops the in-shader clamp and guard of images
/// (`robust_image_access`); buffer robustness chooses no encoding, as the shader checks every
/// descriptor view. The engine executor's own context enables the ones its device's description
/// states that the runtime enables on some device (the image ones), all or none.
pub const ROBUSTNESS_FEATURES: [VkFeature; 4] = [
    VkFeature::RobustBufferAccess,
    VkFeature::RobustBufferAccess2,
    VkFeature::RobustImageAccess,
    VkFeature::RobustImageAccess2,
];

/// How buffers reached through descriptor views (and texel buffers) are contained: the shader
/// checks every one, on every profile. Its out-of-bounds targets' mechanism
/// (`executor_contract::out_of_bounds`) states each access's check, where this module-wide label
/// says what the descriptor views take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferEncoding {
    /// The module declares no buffer (no variable in the `StorageBuffer` or `Uniform` storage
    /// class, no texel buffer), so no buffer encoding applies to it, whatever its profile's.
    Unused,
    /// Checked in the shader.
    Checked,
    /// Left to robust descriptors, the shader emitting no check: a stored row may state it; no
    /// compile does.
    Robust,
}

/// How image reads and writes are contained (`DeviceProfile::device_reads_texels` and
/// `device_drops_texel_writes`, which the product's texture family reads). A texel buffer is no
/// image here: buffer robustness, not image robustness, contains it, so it counts toward
/// [`BufferEncoding`] alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEncoding {
    /// The module declares no image type (`OpTypeImage`) but a texel buffer's, so it accesses no
    /// image and no image encoding applies to it, whatever its profile's.
    Unused,
    /// Reads clamped and writes guarded in the shader: no image robustness is enabled.
    Checked,
    /// `robustImageAccess` without the second level: writes are left to the device, which discards
    /// them out of bounds; reads are clamped in the shader, since the device's out-of-bounds texel
    /// is not Metal's.
    RobustWrites,
    /// `robustImageAccess2` is enabled: reads and writes are left to the device, no clamp, no
    /// guard.
    Robust,
}

/// The containment encodings a module compiled under: the leaf encodings of its descriptor
/// accesses. They are those of the profile it compiled for, which every lowering that reads them
/// reads (the access targets' checks, `lower::texture::slot`).
///
/// Census names do not all show the encoding they compiled under, so the report keys its rows by
/// this, taken from the compile, not by the names. An access by device address (a `BufferForm::Psb`
/// slot or a wide pointer) or to an arena is checked in the shader whatever the profile; the census
/// names it apart from a descriptor access of the same strategy, so the name, not the module, says
/// which accesses these encodings apply to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(from = "ContainmentRecord")]
pub struct Containment {
    pub buffers: BufferEncoding,
    pub images: ImageEncoding,
}

/// A stored [`Containment`]. Rows of executor v34 also state `device_addresses`, the module-level
/// device-address part, which naming each device-address access in the census superseded: it is
/// read and dropped. Such a row is not current (`ENGINE_EXECUTOR_ABI` v37), so it is evidence
/// for no row whatever it states.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContainmentRecord {
    buffers: BufferEncoding,
    images: ImageEncoding,
    #[serde(default, rename = "device_addresses")]
    _device_addresses: Option<bool>,
}

impl From<ContainmentRecord> for Containment {
    fn from(record: ContainmentRecord) -> Containment {
        Containment {
            buffers: record.buffers,
            images: record.images,
        }
    }
}

impl Containment {
    /// The encodings of a module of SPIR-V `words` compiled for `profile`.
    pub fn of_compile(profile: &DeviceProfile, words: &[u32]) -> Containment {
        let declared = Declared::of(words);
        Containment {
            buffers: if !declared.buffers {
                BufferEncoding::Unused
            } else {
                BufferEncoding::Checked
            },
            images: if !declared.images {
                ImageEncoding::Unused
            } else if profile.device_reads_texels(false) {
                ImageEncoding::Robust
            } else if profile.device_drops_texel_writes(false) {
                ImageEncoding::RobustWrites
            } else {
                ImageEncoding::Checked
            },
        }
    }

    /// `buffers:<b>,images:<i>`, as the report prints it.
    pub fn label(self) -> String {
        format!(
            "buffers:{},images:{}",
            match self.buffers {
                BufferEncoding::Unused => "unused",
                BufferEncoding::Checked => "checked",
                BufferEncoding::Robust => "robust",
            },
            match self.images {
                ImageEncoding::Unused => "unused",
                ImageEncoding::Checked => "checked",
                ImageEncoding::RobustWrites => "robust_writes",
                ImageEncoding::Robust => "robust",
            },
        )
    }
}

/// The resources a SPIR-V module declares that a containment encoding can apply to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Declared {
    /// A variable in the `StorageBuffer` or `Uniform` storage class, or a texel buffer
    /// (`OpTypeImage` of `Dim` `Buffer`, contained by buffer robustness).
    buffers: bool,
    /// An image type other than a texel buffer's (`OpTypeImage` of a `Dim` other than `Buffer`),
    /// contained by image robustness.
    images: bool,
}

impl Declared {
    /// What `words` declare. Words that do not parse as instructions after the header declare
    /// both: the reading that never merges one encoding into another.
    fn of(words: &[u32]) -> Declared {
        const OP_TYPE_IMAGE: u32 = 25;
        const OP_VARIABLE: u32 = 59;
        const DIM_BUFFER: u32 = 5;
        const UNIFORM: u32 = 2;
        const STORAGE_BUFFER: u32 = 12;
        let mut out = Declared {
            buffers: false,
            images: false,
        };
        let mut at = 5;
        while let Some(&word) = words.get(at) {
            let count = (word >> 16) as usize;
            let Some(operands) = words.get(at + 1..at + count).filter(|_| count > 0) else {
                break;
            };
            match word & 0xffff {
                OP_TYPE_IMAGE if operands.get(2) == Some(&DIM_BUFFER) => out.buffers = true,
                OP_TYPE_IMAGE => out.images = true,
                OP_VARIABLE => {
                    out.buffers |= matches!(operands.get(2), Some(&(UNIFORM | STORAGE_BUFFER)));
                }
                _ => {}
            }
            at += count;
        }
        if at != words.len() || words.len() < 5 {
            return Declared {
                buffers: true,
                images: true,
            };
        }
        out
    }
}

/// The census of one compiled or executed stage module.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleCensus {
    pub stage: Stage,
    /// The module's subgroup shape, where the executor compiled it itself: `None` for a run
    /// through the runtime, which keeps its modules to itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subgroup: Option<SubgroupShape>,
    /// `StrategyCensus` names. A run through the runtime, which keeps the translator's census of
    /// its modules to itself, records only the float sums and products the case's AIR calls,
    /// under the names the translator's census gives them
    /// (`check::interface::float_reductions`): what its comparison class is derived from.
    pub strategies: BTreeSet<String>,
    /// The module's precision notes: with `strategies`, what its observations' comparison class
    /// is derived from. An executed module's are its notes under the specialization it ran with and
    /// the values its plan pushed (`StageModule::precision_notes_bound`), a compiled one's every
    /// note it records.
    pub precision: BTreeSet<PrecisionNote>,
    /// The containment encodings it compiled under. A row recorded before executors recorded them
    /// (executor ABI v33 and earlier) has none, and is evidence for no row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub containment: Option<Containment>,
}

impl ModuleCensus {
    /// The census of a linked stage module compiled for `profile`, or `None` for a stage Metal
    /// has no function of (a post-tessellation vertex function's module is its evaluation stage),
    /// as a pipeline created it under `plan`'s specialization and a dispatch or draw ran it with
    /// the values `plan` pushes: its precision notes are those that hold of them
    /// (`StageModule::precision_notes_bound`, `BindingPlan::specialized_value`). `None`: every
    /// note the module records, the answer where the plan is not known.
    pub fn of_stage_module_under(
        module: &StageModule,
        profile: &DeviceProfile,
        plan: Option<&BindingPlan>,
    ) -> Option<ModuleCensus> {
        let stage = [
            (vk::ShaderStage::COMPUTE, Stage::Kernel),
            (vk::ShaderStage::VERTEX, Stage::Vertex),
            (vk::ShaderStage::TESSELLATION_EVALUATION, Stage::Vertex),
            (vk::ShaderStage::FRAGMENT, Stage::Fragment),
            (vk::ShaderStage::TASK_EXT, Stage::Object),
            (vk::ShaderStage::MESH_EXT, Stage::Mesh),
        ]
        .into_iter()
        .find(|(vk_stage, _)| *vk_stage == module.stage)
        .map(|(_, stage)| stage)?;
        Some(ModuleCensus {
            stage,
            subgroup: Some(SubgroupShape::of_stage_module(module)),
            strategies: module.strategies.entries.keys().cloned().collect(),
            precision: match plan {
                Some(plan) => module
                    .precision_notes_bound(&plan.specialization, &|key| plan.specialized_value(key))
                    .into_iter()
                    .collect(),
                None => module.precision_notes.iter().copied().collect(),
            },
            containment: Some(Containment::of_compile(profile, &module.spirv)),
        })
    }

    /// The census of each module of a linked pipeline's `modules` (in pipeline order) that runs
    /// a Metal function, compiled for `profile`. A tessellation pipeline's synthesized vertex and
    /// control stages run none and are left out: the pipeline is tessellated exactly when it has an
    /// evaluation stage. `Err` names a module of a stage no Metal function has.
    pub fn of_pipeline_modules(
        modules: &[StageModule],
        profile: &DeviceProfile,
    ) -> Result<Vec<ModuleCensus>, String> {
        ModuleCensus::of_pipeline_modules_under(modules, profile, None)
    }

    /// [`ModuleCensus::of_pipeline_modules`] of a pipeline a dispatch or draw ran under `plan`
    /// ([`ModuleCensus::of_stage_module_under`]); `None` where it is not known.
    pub fn of_pipeline_modules_under(
        modules: &[StageModule],
        profile: &DeviceProfile,
        plan: Option<&BindingPlan>,
    ) -> Result<Vec<ModuleCensus>, String> {
        let tessellated = modules
            .iter()
            .any(|m| m.stage == vk::ShaderStage::TESSELLATION_EVALUATION);
        modules
            .iter()
            .filter(|m| {
                !tessellated
                    || (m.stage != vk::ShaderStage::VERTEX
                        && m.stage != vk::ShaderStage::TESSELLATION_CONTROL)
            })
            .map(|m| {
                ModuleCensus::of_stage_module_under(m, profile, plan).ok_or_else(|| {
                    format!(
                        "the linked module's stage {:?} has no authored stage",
                        m.stage
                    )
                })
            })
            .collect()
    }
}

/// A Vulkan driver (`VkDriverId`): the host class of a GPU (NVIDIA proprietary, RADV, ANV, ...).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HostClass(pub vk::DriverId);

impl std::fmt::Debug for HostClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Where a module runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionClass {
    /// A CPU Vulkan implementation (`VK_PHYSICAL_DEVICE_TYPE_CPU`: lavapipe, SwiftShader).
    CpuIcd,
    /// A GPU of one host class.
    Host(HostClass),
}

impl ExecutionClass {
    /// The class of a device, or `None` for a description that names no device (a capability
    /// profile, such as `runtime-floor`, that no host is).
    pub fn of_description(description: &DeviceDescription) -> Option<ExecutionClass> {
        if description.device_type == Some(vk::PhysicalDeviceType::CPU) {
            return Some(ExecutionClass::CpuIcd);
        }
        description
            .driver_id
            .map(|driver| ExecutionClass::Host(HostClass(driver)))
    }
}

/// The census of what one candidate run executed, and where (the evidence side).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutedCensus {
    /// `None` when the opened device's description names neither a CPU type nor a driver.
    pub device: Option<ExecutionClass>,
    /// The product build that compiled the modules (`TRANSLATOR_FINGERPRINT`): the one currency
    /// rule (`index::candidate_staleness`) holds a row whose census names another product stale, so
    /// evidence is current only for the build that recorded it.
    pub product: String,
    /// The committed capability profile `VK_LAYER_KHRONOS_profiles` restricted the device to,
    /// if any (`--profiles-layer P`). `device` is then the device that executed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restricted_to: Option<String>,
    pub modules: Vec<ModuleCensus>,
    /// How the run contained each out-of-bounds target of its case, and what escaped: the facts the
    /// candidate's verdict on such a case reads (`executor_contract::out_of_bounds::matches`).
    /// `None` for a case that declares no target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_of_bounds: Option<crate::executor_contract::out_of_bounds::OutOfBoundsRun>,
    /// How the run drew its case's rate-mapped render pass: `None` for a case whose pass has no
    /// rate map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_map_pass: Option<RateMapPassRun>,
    /// What the run executed, by content, and under which product runtime: C3's currency holds a
    /// row of another build current while its artifact is the one the authored compile census
    /// computes for that build. `None` for a run of an executor before ABI v129.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<ExecutedArtifact>,
}

/// How a candidate drew a rate-mapped render pass: region by region (the option (a), on every
/// device), into the product's compile of the map, each draw recorded once per region, with the
/// plan's near-equivalent, which the row's comparison class carries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateMapPassRun {
    /// The screen size: the pass's viewport, in screen coordinates.
    pub screen_size: [u32; 2],
    /// The physical size of the product's compile of the map: the extent the pass draws into.
    pub compiled_physical_size: [u32; 2],
    /// The regions each draw of the pass was recorded in.
    pub regions: u32,
    /// The pass's precision notes: with the modules', what the
    /// row's comparison class is derived from.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub precision: BTreeSet<PrecisionNote>,
}

impl ExecutedCensus {
    /// The comparison class of an observation of these modules, derived from their notes and
    /// strategies: what a recorded class must equal.
    pub fn comparison_class(&self) -> ComparisonClass {
        ComparisonClass::derive(
            self.modules
                .iter()
                .flat_map(|module| module.precision.iter().copied())
                .chain(
                    self.rate_map_pass
                        .iter()
                        .flat_map(|pass| pass.precision.iter().copied()),
                ),
            self.modules
                .iter()
                .flat_map(|module| module.strategies.iter().map(String::as_str)),
        )
    }
}

/// Which robustness a tier-table entry compiles a committed profile under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Leg {
    /// The profile as committed, with the robustness features it states: the robust image
    /// encodings where it states them.
    Stated,
    /// The profile without any of [`ROBUSTNESS_FEATURES`]: the in-shader image encodings. A
    /// profile has this leg only when it states one of them.
    RobustnessOff,
}

/// `description` without any robustness feature ([`ROBUSTNESS_FEATURES`]): its robustness-off
/// leg, or `None` when it states none, so that leg is its stated one.
pub fn robustness_off(description: &DeviceDescription) -> Option<DeviceDescription> {
    let mut off = description.clone();
    off.features
        .retain(|feature| !ROBUSTNESS_FEATURES.contains(feature));
    (off.features.len() != description.features.len()).then_some(off)
}
