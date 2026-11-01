//! Tessellation pipelines: the fetch projection of a post-tessellation vertex function and the
//! stages link adds around its evaluation stage.
//!
//! **The projection.** Every attribute a post-tessellation vertex function reads (a patch input
//! or a control-point field) is pulled: a tessellation evaluation shader has no fixed-function
//! vertex input. So the projection is each read attribute's descriptor entry as it stands
//! (`VertexFetch::pulled`), after the numeric-class check a vertex function's projection makes
//! (`link::vertex`): a float-class format read into an integer value, or an unnormalized integer
//! format read into a float one, is not a conversion whose Metal meaning is established. Which step
//! functions the evaluation shader fetches is `lower::tessellation`'s. The pipeline's
//! `VertexInputState` lists every attribute pulled and no fixed-function binding
//! ([`vertex_input_state`]).
//!
//! **The control stage** ([`control`]). A render pipeline whose vertex function is
//! post-tessellation draws each patch as one vertex of a `PATCH_LIST` of one control point
//! through a synthesized vertex stage and a synthesized tessellation control stage that reads the
//! patch's factors from the encoder's factor buffer, its patch index from the draw's patch index
//! buffer when the draw has one, and its control points' indices from the draw's control-point
//! index buffer when the pipeline has a control-point index type. [`control_plan`] checks the
//! descriptor's tessellation state (`None` is Metal's defaults) against Metal's rules and the
//! profile:
//!
//! - `maxTessellationFactor` between 1 and 64, a power of two for `pow2`, even for the fractional
//!   partitions (`InvalidDescriptor(MaxTessellationFactor)` otherwise), and at most the device's
//!   `maxTessellationGenerationLevel` (`Unsupported{Tessellation; Limit}`);
//! - a pipeline of vertex amplification above 1 (multiview) with tessellation stages needs
//!   `multiviewTessellationShader` (`Unsupported{VertexAmplification; Feature}` without it).
//!
//! Link numbers the factor buffer (`SyntheticBinding::TessellationFactors`), the patch index
//! buffer (`SyntheticBinding::PatchIndices`), and, for a pipeline with a control-point index
//! type, the control-point index buffer (`SyntheticBinding::ControlPointIndices`), storage
//! buffers of the control stage ([`ControlPlan::slots`]), among set 0's bindings and places the control stage's push range
//! ([`control::PUSH_BYTES`]) at the first 4-byte-aligned offset no other stage's range covers
//! ([`push_offset`]), within `maxPushConstantsSize` (`Unsupported{Tessellation; Limit}`
//! otherwise). A fragment function that reads `[[primitive_id]]` behind the evaluation stage is
//! not implemented ([`check_fragment`]): Vulkan's primitive id there counts patches, and what
//! Metal's is for a tessellated triangle has not been read.

pub mod control;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use crate::budget::Meter;
use crate::device::requirements::{Requirements, VkFeature, VkLimit};
use crate::device::{DeviceProfile, DriverWorkaround};
use crate::error::{
    Capability, DescriptorError, Error, Feature, LowerFamily, RuntimeInput, Unsupported,
};
use crate::lower::stage_io::fetch::encoding;
use crate::metal::{TessellationControlPointIndexType, TessellationPartitionMode as Partition};
use crate::pipeline::artifact::{
    PrecisionNote, Reconvergence, SpecializationLayout, StageModule, StrategyCensus,
};
use crate::pipeline::descriptor::{TessellationDescriptor, VertexDescriptor};
use crate::pipeline::graphics::{EmulatedTessellation, TessellationState};
use crate::pipeline::interface::{StageInterface, TessellationStage};
use crate::pipeline::layout::PushConstantRange;
use crate::pipeline::lower_inputs::{VertexFetch, VertexFetchProjection};
use crate::pipeline::stage_io::{IoNumeric, StageBuiltin};
use crate::pipeline::sym::ResourceSlot;
use crate::pipeline::vertex::{PulledAttribute, VertexInputState, VertexStride};
use crate::spirv::serialize::{SymbolTable, ENTRY_NAME};
use crate::vk;

use control::Control;

/// The projection of `attributes` (the Metal attribute indices a post-tessellation vertex
/// function reads, with the numeric class it reads each as) for vertex descriptor `desc`: every
/// one pulled. An attribute the descriptor lacks is an invalid descriptor, and so is any of its
/// attributes whose layout it lacks, read or not (`VertexDescriptor::check_layouts`).
pub fn patch_fetch_projection(
    desc: Option<&VertexDescriptor>,
    attributes: &BTreeMap<u32, IoNumeric>,
) -> Result<VertexFetchProjection, Error> {
    if let Some(d) = desc {
        d.check_layouts()?;
    }
    let mut out = VertexFetchProjection::new();
    for (&a, &numeric) in attributes {
        let attr = desc
            .and_then(|d| d.attributes.get(&a))
            .ok_or(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                attribute: a,
            }))?;
        let layout = desc.and_then(|d| d.layouts.get(&attr.buffer_index)).ok_or(
            Error::InvalidDescriptor(DescriptorError::VertexLayout {
                layout: attr.buffer_index,
            }),
        )?;
        let float = encoding(attr.format)?.is_float();
        if float != (numeric == IoNumeric::Float) {
            return Err(
                Unsupported::not_implemented(Feature::Lowering(LowerFamily::Tessellation)).into(),
            );
        }
        out.attributes.insert(a, VertexFetch::pulled(attr, layout));
    }
    Ok(out)
}

/// The vertex input state of the evaluation stage `vs` for descriptor `desc`: every attribute it
/// reads pulled, as its projection states it. A stage compiled under another projection (a
/// `Compiled` stage source whose inputs do not fit this descriptor) is an invalid descriptor, as
/// for a vertex function (`link::vertex::vertex_input_state`).
pub fn vertex_input_state(
    vs: &StageInterface,
    desc: Option<&VertexDescriptor>,
) -> Result<VertexInputState, Error> {
    let used = vs.io.attributes();
    let numeric: BTreeMap<u32, IoNumeric> = used.iter().map(|(a, (n, _))| (*a, *n)).collect();
    let projection = patch_fetch_projection(desc, &numeric)?;
    let mut state = VertexInputState::default();
    for (&a, (_, fetch)) in &used {
        if projection.attributes.get(&a) != Some(fetch) {
            return Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                attribute: a,
            }));
        }
        // The projection holds exactly the descriptor's attributes and layouts it read.
        let (Some(attr), Some(layout)) = (
            desc.and_then(|d| d.attributes.get(&a)),
            desc.and_then(|d| {
                d.attributes
                    .get(&a)
                    .and_then(|attr| d.layouts.get(&attr.buffer_index))
            }),
        ) else {
            return Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                attribute: a,
            }));
        };
        state.pulled.push(PulledAttribute {
            attribute: a,
            buffer_index: attr.buffer_index,
            format: attr.format,
            offset: attr.offset,
            step_function: layout.step_function,
            step_rate: layout.step_rate,
            stride: match layout.stride {
                Some(s) => VertexStride::Static(s),
                None => VertexStride::PerBind,
            },
        });
    }
    Ok(state)
}

/// A tessellation pipeline's control stage as link plans it: the descriptor's tessellation
/// state (Metal's defaults when it states none) and the evaluation stage's interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlPlan {
    pub desc: TessellationDescriptor,
    pub stage: TessellationStage,
}

impl ControlPlan {
    /// The control stage of this plan with its push fields at `push_offset`.
    pub fn control(&self, push_offset: u32) -> Control {
        Control {
            domain: self.stage.domain,
            partition: self.desc.partition_mode,
            winding: self.desc.output_winding_order,
            max_factor: self.desc.max_tessellation_factor,
            step: self.desc.factor_step_function,
            scale_enabled: self.desc.factor_scale_enabled,
            push_offset,
            control_points: self.stage.control_points,
            index_type: self.desc.control_point_index_type,
        }
    }

    /// The control stage's binding symbols: the factor and patch index buffers, and the
    /// control-point index buffer for a control-point index type.
    pub fn slots(&self) -> Vec<ResourceSlot> {
        let mut slots = vec![control::FACTOR_SLOT, control::PATCH_INDEX_SLOT];
        if self.desc.control_point_index_type != TessellationControlPointIndexType::None {
            slots.push(control::CONTROL_POINT_INDEX_SLOT);
        }
        slots
    }

    /// The pipeline's `TessellationState` with the control stage's push fields at `push_offset`.
    pub fn state(&self, push_offset: u32) -> TessellationState {
        TessellationState {
            patch_control_points: 1,
            domain: self.stage.domain,
            control_points: self.stage.control_points,
            factor_step_function: self.desc.factor_step_function,
            push_offset,
            control_point_index_type: self.desc.control_point_index_type,
        }
    }
}

/// Check the descriptor's tessellation state `desc` (`None`: Metal's defaults) of a pipeline
/// whose evaluation stage is `stage` and whose `maxVertexAmplificationCount` is `amplification`
/// on profile `p` (see the module docs).
pub fn control_plan(
    desc: Option<&TessellationDescriptor>,
    stage: TessellationStage,
    amplification: u32,
    p: &DeviceProfile,
) -> Result<ControlPlan, Error> {
    let desc = checked(desc)?;
    let max = desc.max_tessellation_factor;
    let f = p.fields();
    let level = f.limits.max_tessellation_generation_level;
    if max > level {
        return Err(Unsupported::needs(
            Feature::Tessellation,
            Capability::Limit {
                limit: VkLimit::MaxTessellationGenerationLevel,
                required: u64::from(max),
                available: u64::from(level),
            },
        )
        .into());
    }
    if amplification > 1
        && !f
            .enabled
            .features
            .contains(&VkFeature::MultiviewTessellationShader)
    {
        return Err(Unsupported::needs(
            Feature::VertexAmplification,
            Capability::Feature(VkFeature::MultiviewTessellationShader),
        )
        .into());
    }
    Ok(ControlPlan { desc, stage })
}

/// The descriptor's tessellation state `desc` (`None`: Metal's defaults), held to Metal's rules: a
/// `maxTessellationFactor` between 1 and 64, a power of two for `pow2` and even for the
/// fractional partitions (`InvalidDescriptor(MaxTessellationFactor)` otherwise).
fn checked(desc: Option<&TessellationDescriptor>) -> Result<TessellationDescriptor, Error> {
    let desc = desc.copied().unwrap_or_default();
    let max = desc.max_tessellation_factor;
    let valid = (1..=64).contains(&max)
        && match desc.partition_mode {
            Partition::Pow2 => max.is_power_of_two(),
            Partition::FractionalOdd | Partition::FractionalEven => max % 2 == 0,
            _ => true,
        };
    if !valid {
        return Err(Error::InvalidDescriptor(
            DescriptorError::MaxTessellationFactor {
                factor: max,
                partition_mode: desc.partition_mode,
            },
        ));
    }
    Ok(desc)
}

/// The emulated tessellation state of a pipeline whose evaluation stage `stage` was compiled for
/// a device without `tessellationShader` (`TessellationStage::emulated`): the descriptor's
/// tessellation state `desc` (`None`: Metal's defaults), held to Metal's rules as
/// [`control_plan`] holds it, but not to the device's tessellation limits (the emulation uses no
/// tessellation stage), and the state the stage baked in (`NeedsInput(TessellationDescriptor)`
/// otherwise: compiled for another state, the function draws another tessellation).
pub fn emulated_state(
    desc: Option<&TessellationDescriptor>,
    stage: TessellationStage,
) -> Result<EmulatedTessellation, Error> {
    let desc = checked(desc)?;
    if stage.emulated != Some(desc) {
        return Err(Error::NeedsInput(RuntimeInput::TessellationDescriptor));
    }
    Ok(EmulatedTessellation {
        domain: stage.domain,
        control_points: stage.control_points,
        descriptor: desc,
    })
}

/// A fragment stage `fs` behind an evaluation stage: one that reads `[[primitive_id]]`, from the
/// builtin or from its varying (`IoRole::PrimitiveId`), is not implemented (the module docs).
pub fn check_fragment(fs: Option<&StageInterface>) -> Result<(), Error> {
    let reads = |f: &StageInterface| {
        f.io.builtins.contains(&StageBuiltin::PrimitiveId)
            || f.io
                .inputs
                .iter()
                .any(|v| v.role == crate::pipeline::stage_io::IoRole::PrimitiveId)
    };
    if fs.is_some_and(reads) {
        return Err(Unsupported::not_implemented(Feature::Tessellation).into());
    }
    Ok(())
}

/// The byte offset of a synthesized stage's push range of `bytes` beside the stages' `ranges`
/// (the control stage's [`PUSH_BYTES`](control::PUSH_BYTES) of words, a band's evaluation
/// stage's `vec4`): the first offset, a multiple of `align`, whose `bytes` no range covers, within
/// `maxPushConstantsSize`; refused as `feature` needing a larger limit otherwise.
pub fn push_offset(
    ranges: &[PushConstantRange],
    (bytes, align): (u32, u32),
    feature: Feature,
    p: &DeviceProfile,
) -> Result<u32, Error> {
    let max = p.fields().limits.max_push_constants_size;
    let free = |at: u32| {
        ranges
            .iter()
            .all(|r| at + bytes <= r.offset || at >= r.offset.saturating_add(r.size))
    };
    (0..max)
        .step_by(align as usize)
        .take_while(|&at| at + bytes <= max)
        .find(|&at| free(at))
        .ok_or_else(|| {
            let end = ranges
                .iter()
                .map(|r| u64::from(r.offset) + u64::from(r.size))
                .max()
                .unwrap_or(0);
            Unsupported::needs(
                feature,
                Capability::Limit {
                    limit: VkLimit::MaxPushConstantsSize,
                    required: end + u64::from(bytes),
                    available: u64::from(max),
                },
            )
            .into()
        })
}

/// Note what control stage `c` takes without a Metal observation ([`Control::notes`]) on the
/// pipeline's evaluation stage among `modules`, and, on profile `p` whose tessellator is not
/// Apple's (`DriverWorkaround::TessellatorNotMetals`), the device's triangle order
/// (`PrecisionNote::TessellationTriangleOrder`): its output is what the readings change, and the
/// synthesized stages are no function's module (a census leaves them out).
pub fn note_evaluation_stage(c: &Control, p: &DeviceProfile, modules: &mut [StageModule]) {
    let mut notes = c.notes();
    if p.workarounds()
        .contains(&DriverWorkaround::TessellatorNotMetals)
    {
        notes.push(PrecisionNote::TessellationTriangleOrder);
    }
    for m in modules
        .iter_mut()
        .filter(|m| m.stage == vk::ShaderStage::TESSELLATION_EVALUATION)
    {
        // Unconditional on the stage: no specialization removes them.
        m.specialized_notes.retain(|s| !notes.contains(&s.note));
        m.precision_notes.extend(notes.iter().copied());
        m.precision_notes.sort_unstable();
        m.precision_notes.dedup();
    }
}

/// The synthesized vertex and tessellation control stages of `c` (in pipeline order) as final
/// modules under link table `table`, each with the requirements of its module: verified by the
/// owned verifier, serialized, and relocated, as a compiled function's module is.
pub fn synthesized_stages(
    c: &Control,
    table: &impl SymbolTable,
    p: &DeviceProfile,
    m: &mut Meter,
) -> Result<Vec<(StageModule, Requirements)>, Error> {
    let modules = [
        (control::vertex_module()?, vk::ShaderStage::VERTEX),
        (
            control::control_module(c)?,
            vk::ShaderStage::TESSELLATION_CONTROL,
        ),
    ];
    let mut out = Vec::with_capacity(modules.len());
    for (module, stage) in modules {
        let stats = crate::spirv::verify::verify(&module, p, m)?;
        let module = crate::spirv::serialize::serialize_for(module, p)?;
        let words = crate::spirv::serialize::relocate(&module, table)?;
        out.push((
            StageModule {
                stage,
                entry_point: ENTRY_NAME.to_owned(),
                spirv: words.into(),
                specialization: SpecializationLayout::default(),
                required_subgroup_size: None,
                stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
                reconvergence: Reconvergence::Guaranteed,
                precision_notes: Vec::new(),
                strategies: StrategyCensus::default(),
                constant_blob: None,
                rounded_colors: Vec::new(),
                specialized_notes: Vec::new(),
                own_pixel_textures: Vec::new(),
                snapshot_textures: Vec::new(),
                private_bytes: 0,
                stats,
            },
            module.requirements,
        ));
    }
    Ok(out)
}
