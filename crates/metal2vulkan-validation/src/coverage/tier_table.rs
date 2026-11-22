//! The tier table and a compile of one entry for it, for the tests that check the product's
//! containment and guards on every committed profile: each committed profile as stated and in its
//! robustness-off leg ([`committed_profiles`]), and the census and out-of-bounds containment of an
//! entry compiled for one ([`compile_module`]).

use super::*;
use crate::candidate::engine::{function_constants, graphics};
use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsTarget};
use crate::case::AuthoredCase;
use crate::check::loop_free::DecodedModule;
use crate::engine::{bounded_detail, Outcome};
use crate::executor_contract::out_of_bounds::{self, Mechanism};
use crate::literal::LiteralResources;
use metal2vulkan::device::{Device, DevicePolicy, DeviceProfile};
use metal2vulkan::error::Error;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::pipeline::{EntryDescriptor, EntryRef, LoweringInputs};
use metal2vulkan::pipeline::{StageInputDescriptor, VertexDescriptor};
use metal2vulkan::Budget;
use std::time::{Duration, Instant};

/// A committed device profile in the tier table, in one [`Leg`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TierProfile {
    /// The committed profile's stem.
    pub stem: String,
    pub leg: Leg,
    pub class: Option<ExecutionClass>,
    /// `subgroupSize`: the width of a module that asks for full subgroups without pinning one.
    pub subgroup_size: u32,
}

impl TierProfile {
    /// The entry of `description` (the committed profile `stem`) in `leg`.
    pub fn of_description(stem: &str, leg: Leg, description: &DeviceDescription) -> TierProfile {
        TierProfile {
            stem: stem.into(),
            leg,
            class: ExecutionClass::of_description(description),
            subgroup_size: description.subgroup.size,
        }
    }

    /// The entry's name in the report: the stem, with `+robustness-off` for that leg.
    pub fn label(&self) -> String {
        match self.leg {
            Leg::Stated => self.stem.clone(),
            Leg::RobustnessOff => format!("{}+robustness-off", self.stem),
        }
    }
}

/// A tier-table entry: a committed profile in one leg, and the profile lowering compiles for: the
/// resident-table domains, whose containment mechanisms (the root table, descriptor views, the
/// robustness legs) these tests check.
pub struct CommittedProfile {
    pub tier: TierProfile,
    pub profile: DeviceProfile,
}

/// The tier table: every committed profile (`validation/profiles`), sorted by stem, as stated
/// and, where it states a robustness feature, in its robustness-off leg ([`robustness_off`]).
pub fn committed_profiles() -> Result<Vec<CommittedProfile>, String> {
    let mut out = Vec::new();
    for stem in crate::engine::committed_profiles()? {
        let description = crate::engine::load_description(&stem)?;
        let derive = |description: DeviceDescription| {
            Device::from_queries(description, DevicePolicy::RESIDENT_TABLES)
                .map(|device| device.profile().clone())
                .map_err(|error| format!("{stem}: {error}"))
        };
        out.push(CommittedProfile {
            tier: TierProfile::of_description(&stem, Leg::Stated, &description),
            profile: derive(description.clone())?,
        });
        if let Some(off) = robustness_off(&description) {
            out.push(CommittedProfile {
                tier: TierProfile::of_description(&stem, Leg::RobustnessOff, &off),
                profile: derive(off)?,
            });
        }
    }
    Ok(out)
}

/// The translation ceiling of one attempt (AGENTS.md).
const ATTEMPT_DEADLINE: Duration = Duration::from_secs(20);

/// Why a compile produced no module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// A typed product refusal (the outcome's class and key): the entry selects nothing on the
    /// profile.
    Refused(String),
    /// Anything else (a deadline, a budget, an invariant, a panic, a stage the report has no row
    /// for): what the entry selects is unknown.
    Failed(String),
}

/// Whether a compile error's outcome is a typed product refusal: the product decided the entry
/// does not compile on the profile. Every `Failed` outcome, and a budget breach, is a defect or a
/// transient, not an answer.
fn is_typed_refusal(outcome: &Outcome) -> bool {
    !matches!(outcome, Outcome::Translated | Outcome::Failed { .. }) && !outcome.is_defect()
}

/// What a module compiles with: the function constants, and the vertex descriptor a vertex
/// case's executors draw it with. A vertex function's fetch projection is one of
/// its lowering inputs, decided by the descriptor and the profile (on a leg without buffer
/// robustness every attribute is pulled from a checked storage buffer, so the module declares
/// one), so a case's row compile takes the descriptor its executed module compiled under.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseInputs {
    pub constants: FunctionConstantValues,
    /// The Metal oracle's vertex descriptor of a vertex case with vertex inputs or tessellation
    /// inputs (`graphics::vertex_descriptor` of `graphics::vertex_buffers`); `None` for every
    /// other case and for a fixture entry, which compile with default lowering inputs.
    pub vertex_descriptor: Option<VertexDescriptor>,
    /// The stage-input descriptor of a kernel case with stage inputs, as both executors fetch
    /// them (`executor_contract::stage_inputs`); `None` for every other case and for
    /// a fixture entry.
    pub stage_input: Option<StageInputDescriptor>,
    /// The case's out-of-bounds targets, whose containment the compile reports;
    /// none for a fixture entry.
    pub out_of_bounds: Vec<OutOfBoundsTarget>,
}

impl CaseInputs {
    /// The default constants, no vertex descriptor, and no out-of-bounds target: what a fixture
    /// entry compiles with.
    pub fn defaults() -> CaseInputs {
        CaseInputs {
            constants: FunctionConstantValues::defaults(),
            vertex_descriptor: None,
            stage_input: None,
            out_of_bounds: Vec::new(),
        }
    }
}

/// The [`CaseInputs`] of `case`, whose AIR assembles to `bitcode`: the constants its literals map
/// to (`function_constants`), and, for a vertex case that feeds vertex buffers, the vertex
/// descriptor of the buffers its render executors bind, at the oracle's slots for its entry's
/// AIR-layer interface (the derivation `graphics::render_case` uses).
pub fn case_inputs(case: &AuthoredCase, bitcode: &[u8]) -> Result<CaseInputs, String> {
    let literal = LiteralResources::prepare(case)?;
    let constants = function_constants(&literal).map_err(|error| error.to_string())?;
    let feeds_vertex_buffers = !case.vertex_inputs.is_empty() || case.tessellation.is_some();
    let fetches_stage_inputs = case.stage == Stage::Kernel && !case.kernel_stage_inputs.is_empty();
    let interface = if (case.stage == Stage::Vertex && feeds_vertex_buffers) || fetches_stage_inputs
    {
        let decoded = DecodedModule::from_bitcode(&case.entry, bitcode)
            .map_err(|evidence| format!("the AIR does not decode: {evidence}"))?;
        Some(crate::check::entry_interface(case, &decoded)?.2)
    } else {
        None
    };
    let vertex_descriptor = match &interface {
        Some(interface) if case.stage == Stage::Vertex => {
            let buffers = graphics::vertex_buffers(case, &literal, interface)
                .map_err(|error| error.to_string())?;
            graphics::vertex_descriptor(&buffers)
        }
        _ => None,
    };
    let stage_input = match &interface {
        Some(interface) if fetches_stage_inputs => {
            use crate::executor_contract::stage_inputs as fetch;
            fetch::descriptor(&fetch::authored(&case.kernel_stage_inputs, interface)?)
        }
        _ => None,
    };
    Ok(CaseInputs {
        constants,
        vertex_descriptor,
        stage_input,
        out_of_bounds: case
            .out_of_bounds
            .as_ref()
            .map(|declaration| declaration.targets.clone())
            .unwrap_or_default(),
    })
}

/// How a compiled module contains one out-of-bounds target of its case on a tier-table entry:
/// the containment form the module exercises on it and the mechanism the module
/// and the entry's device give it (`executor_contract::out_of_bounds::containment`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetContainment {
    pub form: ContainmentForm,
    pub mechanism: Mechanism,
}

/// A module a compile produced ([`compile_module`]): its census, and how it contains each
/// out-of-bounds target of the case it was compiled for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledModule {
    pub census: ModuleCensus,
    pub out_of_bounds: Vec<TargetContainment>,
}

/// Compile entry `entry` of `bitcode` with `inputs` for `profile` (the 20-s attempt deadline)
/// into the census of its module: default lowering inputs without a vertex descriptor, and the
/// ones a render pipeline's vertex stage takes from the descriptor otherwise
/// (`DeviceProfile::vertex_entry_inputs`); with how the module contains each out-of-bounds
/// target of `inputs` on `profile`'s device (`out_of_bounds::containment`). A
/// target the module's interface does not account for fails the compile: what the case
/// exercises is unknown. The containment encodings are read off the function linked alone
/// (`DeviceProfile::link_standalone`).
pub fn compile_module(
    bitcode: &[u8],
    entry: &str,
    inputs: &CaseInputs,
    profile: &DeviceProfile,
) -> Result<CompiledModule, CompileError> {
    let budget = Budget::with_deadline(Instant::now().checked_add(ATTEMPT_DEADLINE));
    let refusal = |error: Error| {
        let outcome = Outcome::from(&error);
        let reason = format!(
            "{} {} {}",
            outcome.label(),
            outcome.key(),
            bounded_detail(&error.to_string())
        );
        if is_typed_refusal(&outcome) {
            CompileError::Refused(reason)
        } else {
            CompileError::Failed(reason)
        }
    };
    let mut descriptor = EntryDescriptor::new(bitcode, EntryRef::Name(entry));
    descriptor.constants = inputs.constants.clone();
    let lowering = match &inputs.vertex_descriptor {
        None => {
            let mut lowering = LoweringInputs::new();
            lowering.stage_input = inputs.stage_input.clone();
            lowering
        }
        Some(vertex) => profile
            .vertex_entry_inputs(&descriptor, Some(vertex), &budget)
            .map_err(refusal)?,
    };
    let (compiled, _) = profile.compile_entry(&descriptor, &lowering, &budget);
    let function = compiled.map_err(refusal)?;
    use metal2vulkan::pipeline::Stage as Sym;
    let stage = match function.stage() {
        Sym::Kernel => Stage::Kernel,
        Sym::Vertex => Stage::Vertex,
        Sym::Fragment => Stage::Fragment,
        // An object or mesh function compiles on a profile that enables mesh shading.
        Sym::Object => Stage::Object,
        Sym::Mesh => Stage::Mesh,
        other => {
            return Err(CompileError::Failed(format!(
                "a {other:?} function has no authored stage"
            )))
        }
    };
    let attributes = out_of_bounds::Fetch::of_function(&function);
    let module = out_of_bounds::ModuleFacts {
        stage: function.stage(),
        slots: function.slots(),
        attributes: &attributes,
        resident_roots: function.reads_resident_roots(),
        handle_table: function.resolves_handles(),
        // Every committed profile's device takes the resident-table domains (`committed_profiles`).
        policy: DevicePolicy::RESIDENT_TABLES,
        strategies: function.strategies(),
    };
    let out_of_bounds = inputs
        .out_of_bounds
        .iter()
        .map(|target| {
            out_of_bounds::containment(target, &module, profile)
                .map(|contained| TargetContainment {
                    form: contained.form,
                    mechanism: contained.mechanism,
                })
                .map_err(CompileError::Failed)
        })
        .collect::<Result<_, _>>()?;
    let linked = profile.link_standalone(&function).map_err(refusal)?;
    Ok(CompiledModule {
        census: ModuleCensus {
            stage,
            subgroup: Some(SubgroupShape {
                required_size: function.required_subgroup_size(),
                full_subgroups: function.requires_full_subgroups(),
            }),
            strategies: function.strategies().entries.keys().cloned().collect(),
            precision: function.precision_notes().iter().copied().collect(),
            containment: Some(Containment::of_compile(profile, &linked.words)),
        },
        out_of_bounds,
    })
}

impl Containment {
    /// The encodings of a module of SPIR-V `words` compiled for `profile`.
    pub(crate) fn of_compile(profile: &DeviceProfile, words: &[u32]) -> Containment {
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
    pub(crate) fn label(self) -> String {
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
pub(crate) struct Declared {
    /// A variable in the `StorageBuffer` or `Uniform` storage class, or a texel buffer
    /// (`OpTypeImage` of `Dim` `Buffer`, contained by buffer robustness).
    pub(crate) buffers: bool,
    /// An image type other than a texel buffer's (`OpTypeImage` of a `Dim` other than `Buffer`),
    /// contained by image robustness.
    pub(crate) images: bool,
}

impl Declared {
    /// What `words` declare. Words that do not parse as instructions after the header declare
    /// both: the reading that never merges one encoding into another.
    pub(crate) fn of(words: &[u32]) -> Declared {
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

impl ExecutionClass {
    /// The class of a device, or `None` for a description that names no device (a capability
    /// profile, such as `runtime-floor`, that no host is).
    pub(crate) fn of_description(description: &DeviceDescription) -> Option<ExecutionClass> {
        if description.device_type == Some(vk::PhysicalDeviceType::CPU) {
            return Some(ExecutionClass::CpuIcd);
        }
        description
            .driver_id
            .map(|driver| ExecutionClass::Host(HostClass(driver)))
    }
}
