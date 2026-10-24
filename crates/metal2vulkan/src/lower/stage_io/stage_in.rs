//! Kernel `[[stage_in]]`: each attribute a kernel reads is fetched once at entry, as a pulled
//! vertex attribute is (`super::fetch`), from the buffer its compute pipeline's stage-input
//! descriptor (`LoweringInputs::stage_input`) names in the kernel's buffer argument table, and the
//! stage-in region's `[[grid_origin]]` and `[[grid_size]]` are read from the region the binder
//! writes (`PushField::StageInRegion`).
//!
//! **The element** of a layout's step, from Metal's definitions and what an M4 and an M3 Max read
//! past them:
//!
//! - `Constant`: none; the attribute's offset alone ("fetches attribute data once").
//! - `ThreadPositionInGridX`/`Y` at step rate `n`: the region's origin plus the thread's position
//!   on that axis divided by `n` (`setStageInRegion:`: "the GPU adds the values specified by the
//!   region's origin to the thread position in the grid"), the quotient clamped to the region's
//!   last element, its size less 1: a thread past the region reads the region's last element.
//!   The size less 1 is a 32-bit difference, so a region of size 0 on the axis clamps nothing.
//! - `...Indexed` at rate 1: the index the index buffer holds at the thread's position clamped
//!   the same way, of the descriptor's `indexType` at its `indexBufferIndex`, plus the region's
//!   origin (the origin is added after the lookup, as MoltenVK's tessellation relies on, and not
//!   to the position looked up). An indexed step at a rate other than 1 is not implemented: how
//!   the rate meets the index lookup is not established.
//! - A step rate of 0 on a thread-position step is an invalid descriptor (no element is every
//!   0th thread), and so is a vertex or tessellation step in a compute layout
//!   (`StageInFetch::of`): Metal's compute pipeline creation stops at an assertion for one.
//!
//! **The stride** of a layout is its descriptor's, compiled in, whatever stride the encoder binds
//! the buffer with: Metal reads a bound stride only for a layout of
//! `MTLBufferLayoutStrideDynamic` (an M4 read a fixed 16 under a bound 8).
//! Such a layout (`StageInputLayoutDescriptor::stride` `None`) reads the bound one from
//! `PushField::StageInStride`, which the binder writes, as 0 for a buffer bound without one.
//!
//! **Formats** decode as a vertex format of the same raw value (`MTLAttributeFormat` and
//! `MTLVertexFormat` list the same formats), and they meet the attribute's numeric class exactly
//! as a pulled vertex attribute's do (`fetch::class_match`, `fetch::decode`): an unnormalized 8- or
//! 16-bit integer format reaches a float attribute as each component's integer value, which Metal's
//! stage-in fetch and its fixed-function vertex fetch answer the same bits for.

use crate::air::abi::{AbiInt, ParamBinding, Slot};
use crate::error::{DescriptorError, Error, Feature, RuntimeInput, Unsupported};
use crate::lower::arith::Form;
use crate::lower::dispatch::Projection;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::contain;
use crate::metal::{AttributeFormat, IndexType, StepFunction, VertexFormat};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::stage_io::{StageInFetch, StageInInput};
use crate::pipeline::sym::{PushField, Stage};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;

use super::fetch::{self, IndexRead, PulledRead, Step, Stride};
use super::plan::{io_form, io_type, param_ty};
use super::{not_lowered, PlannedParam, Source};

/// Record that the kernel reads the stage-in region, which the binder writes into
/// `PushField::StageInRegion`.
fn read_region(cx: &mut SelectCx<'_>) {
    cx.push.insert(PushField::StageInRegion);
    cx.families.stage_io.region = true;
}

/// Plan the used kernel `[[stage_in]]` parameter `b` of attribute location `location`.
pub(super) fn param(cx: &mut SelectCx<'_>, b: &ParamBinding, location: &Slot) -> Result<(), Error> {
    let AbiInt::Const(attribute) = location.index else {
        return Err(not_lowered());
    };
    let Some(d) = cx.inputs.stage_input.as_ref() else {
        return Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor {
            attribute,
        }));
    };
    let fetch = StageInFetch::of(d, attribute)?;
    let ty = param_ty(cx, b)?;
    let form = io_form(cx, ty)?;
    let read = select(cx, attribute, fetch, form)?;
    cx.families.stage_io.params.push(PlannedParam {
        param: b.param,
        source: Source::Pulled(read),
    });
    Ok(())
}

/// Plan the pulled read of attribute `attribute`, fetched by `f`, into a parameter of form
/// `form`: its step, its source buffers (the access family chooses their slots), and the push
/// fields and region dependency it needs (see the module docs).
fn select(
    cx: &mut SelectCx<'_>,
    attribute: u32,
    f: StageInFetch,
    form: Form,
) -> Result<PulledRead, Error> {
    // `MTLAttributeFormat` lists `MTLVertexFormat`'s formats at the same values.
    let named = |value: u32| {
        let ty = AttributeFormat::METAL_NAME;
        match f.format {
            AttributeFormat::Invalid => DescriptorError::PlaceholderEnumValue { ty, value },
            _ => DescriptorError::UnknownEnumValue { ty, value },
        }
    };
    let format = match f.format {
        AttributeFormat::Invalid => None,
        _ => VertexFormat::try_from(f.format.as_raw()).ok(),
    }
    .ok_or_else(|| Error::InvalidDescriptor(named(f.format.as_raw())))?;
    let encoding = fetch::encoding(format)?;
    if form.is_bool() {
        return Err(not_lowered());
    }
    // The class crossing a stage-input attribute takes is a pulled vertex attribute's
    // (`fetch::class_match`): an unnormalized 8- or 16-bit integer format into a float attribute,
    // which `fetch::decode` converts, and which Metal's stage-in fetch answers the same bits for as
    // its fixed-function vertex fetch. The crossings that function refuses are a *vertex*
    // descriptor's refusals, which a compute layout's are not measured to be, so they stay
    // `NotImplemented` here rather than becoming an invalid descriptor.
    if fetch::class_match(encoding, form.is_float(), attribute).is_err() {
        return Err(not_lowered());
    }
    let layout = f.buffer_index;
    let rate = |rate: u32| -> Result<u32, Error> {
        match rate {
            0 => Err(Error::InvalidDescriptor(
                DescriptorError::StageInputStepRate {
                    layout,
                    step_rate: 0,
                },
            )),
            n => Ok(n),
        }
    };
    let step = match f.step_function {
        StepFunction::Constant => Step::Constant,
        StepFunction::ThreadPositionInGridX | StepFunction::ThreadPositionInGridY => {
            let axis = u8::from(f.step_function == StepFunction::ThreadPositionInGridY);
            Step::Grid {
                axis,
                rate: rate(f.step_rate)?,
            }
        }
        StepFunction::ThreadPositionInGridXIndexed | StepFunction::ThreadPositionInGridYIndexed => {
            let axis = u8::from(f.step_function == StepFunction::ThreadPositionInGridYIndexed);
            if rate(f.step_rate)? != 1 {
                return Err(Unsupported::not_implemented(Feature::KernelStageIn).into());
            }
            let Some((index_type, buffer)) = f.index else {
                return Err(cx.invariant("an indexed stage-input fetch without its index buffer"));
            };
            let bytes = match index_type {
                IndexType::UInt16 => 2,
                IndexType::UInt32 => 4,
            };
            Step::Indexed {
                axis,
                index: IndexRead { buffer, bytes },
            }
        }
        StepFunction::PerVertex
        | StepFunction::PerInstance
        | StepFunction::PerPatch
        | StepFunction::PerPatchControlPoint => {
            return Err(cx.invariant(
                "a stage-input fetch of a vertex step function (StageInFetch::of refuses it)",
            ))
        }
    };
    if let Step::Indexed { index, .. } = step {
        cx.push.insert(PushField::SlotBindOffset(fetch::slot(
            Stage::Kernel,
            index.buffer,
        )));
        cx.families.stage_io.buffers.insert(index.buffer);
    }
    if matches!(step, Step::Grid { .. } | Step::Indexed { .. }) {
        // The element clamps to the region and adds its origin; the thread position in the grid
        // is a grid builtin.
        read_region(cx);
        cx.push.insert(PushField::DispatchRegion);
        cx.families.stage_io.reads_grid_position = true;
    }
    // A view's read adds the bind residual; `access::finish_slots` drops the field for a source
    // buffer it reaches by device address, as for any slot.
    cx.push.insert(PushField::SlotBindOffset(fetch::slot(
        Stage::Kernel,
        f.buffer_index,
    )));
    // A fixed stride is the descriptor's, whatever the bind's; a dynamic layout's is the one the
    // encoder binds the buffer with, which the binder pushes. A constant step's is `Some(0)`
    // (`StageInFetch::of`): it reads none.
    let stride = match f.stride {
        Some(s) => Stride::Static(s),
        None => {
            let field = PushField::StageInStride(f.buffer_index);
            cx.push.insert(field);
            cx.facts.insert(RuntimeFact::StageInStride {
                layout: f.buffer_index,
            });
            Stride::Push(field)
        }
    };
    let sp = &mut cx.families.stage_io;
    sp.buffers.insert(f.buffer_index);
    // The compute pipeline checks the descriptor it is built with against this record.
    if let Err(at) = sp
        .interface
        .stage_in
        .binary_search_by_key(&attribute, |p| p.index)
    {
        sp.interface
            .stage_in
            .insert(at, StageInInput::new(attribute, io_type(form), f));
    }
    Ok(PulledRead {
        attribute,
        encoding,
        buffer: f.buffer_index,
        offset: f.offset,
        step,
        stride,
        form,
        check: contain::encoding(cx.profile),
    })
}

/// Plan the used `[[grid_origin]]` (`size` false) or `[[grid_size]]` (`size` true) parameter
/// `b`: the stage-in region's origin or size, as an integer of 1 to 3 lanes.
pub(super) fn region_param(
    cx: &mut SelectCx<'_>,
    b: &ParamBinding,
    size: bool,
) -> Result<(), Error> {
    let ty = param_ty(cx, b)?;
    let v = crate::air::ir::Value::Arg(crate::air::ir::ArgIdx::new(b.param));
    let lowered = cx.lowered_repr(v, ty)?;
    let projection = Projection::of_uvec3(&lowered).ok_or_else(not_lowered)?;
    read_region(cx);
    cx.families.stage_io.params.push(PlannedParam {
        param: b.param,
        source: Source::StageInRegion { size, projection },
    });
    Ok(())
}

/// The stage-in region's origin (`size` false) or size (`size` true) projected by `projection`,
/// read at the use.
pub(super) fn region_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    size: bool,
    projection: Projection,
) -> BResult<SValue> {
    let region = cx.push_field(fb, PushField::StageInRegion)?;
    let v = fb.composite_extract(region, &[u32::from(size)])?;
    let xyz = fb.vector_shuffle(v, v, &[0, 1, 2])?;
    match crate::lower::dispatch::project(fb, xyz, projection)? {
        Some(x) => Ok(x),
        None => Err(cx.fail("a stage-in region value without its projection")),
    }
}
