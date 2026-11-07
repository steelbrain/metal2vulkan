//! How both executors fetch a kernel case's `[[stage_in]]` data: the Metal oracle's
//! `MTLStageInputOutputDescriptor` and `setStageInRegion:`, and the engine candidate's
//! `ComputePipelineDescriptor::stage_input` and `BindState::stage_in_region`, derived once here, so
//! the two agree.
//!
//! Each authored stage input's data is its own buffer at the slot the shared slot rule gives its
//! location ([`stage_input_slots`]), read as the case's format at offset 0 with the case's stride,
//! stepping by thread position in the grid on x at rate 1 over the case's stage-in region
//! (`Dispatch::stage_in_region`), or without one a region at the origin the size of the dispatch
//! grid: thread `(x, y, z)` reads element `x`, or under a region the origin plus `x` clamped to
//! the region's size less 1. An input of a dynamic stride has a layout of
//! `MTLBufferLayoutStrideDynamic`, and both executors bind its data with the stride form at the
//! case's stride ([`bound_strides`]). Changing it changes the oracle's dispatches, which is a Metal
//! oracle version (`crate::metal::ORACLE_ABI`) and an engine executor version together.

use std::collections::BTreeMap;

use crate::case::indirect_commands::HostStageInRegion;
use crate::case::{AttributeFormat, KernelStageInput};
use crate::check::interface::{bound_constants, constant_int, AirInterface, NumShape};
use crate::check::loop_free::DecodedModule;
use crate::executor_contract::buffer_slots::stage_input_slots;
use metal2vulkan::bind::StageInRegion;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::metal;
use metal2vulkan::pipeline::{
    StageInputAttributeDescriptor, StageInputDescriptor, StageInputLayoutDescriptor,
};
use metal2vulkan_air::abi::ParamRole;

/// Every attribute's byte offset in its element: its data is its own buffer.
pub const OFFSET: u32 = 0;
/// Every layout's step: a new element per thread position in the grid on x.
pub const STEP_FUNCTION: metal::StepFunction = metal::StepFunction::ThreadPositionInGridX;
/// Every layout's step rate.
pub const STEP_RATE: u32 = 1;

/// One stage input as both executors fetch it: its location, the Metal buffer index its data is
/// bound at, its format, and the stride it is read at ([`OFFSET`], [`STEP_FUNCTION`], and
/// [`STEP_RATE`] are every input's): its layout's, or, where `dynamic`, the stride its data is
/// bound with, its layout's being `MTLBufferLayoutStrideDynamic`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInputFetch {
    pub location: u32,
    pub slot: u32,
    pub format: AttributeFormat,
    pub stride: u32,
    pub dynamic: bool,
}

/// The stage-in region of a dispatch of `grid` threads (`setStageInRegion:`): `authored`, a
/// case's `Dispatch::stage_in_region`, or at the origin the size of the grid.
pub fn region(authored: Option<HostStageInRegion>, grid: [u32; 3]) -> StageInRegion {
    match authored {
        Some(HostStageInRegion { origin, size }) => StageInRegion::new(origin, size),
        None => StageInRegion::new([0; 3], grid),
    }
}

/// The fetches of a case's authored stage inputs `inputs` for an entry of AIR-layer interface
/// `interface`, in location order. A location the entry does not declare is an error.
pub fn authored(
    inputs: &[KernelStageInput],
    interface: &AirInterface,
) -> Result<Vec<StageInputFetch>, String> {
    let slots = stage_input_slots(interface)?;
    let mut out = inputs
        .iter()
        .map(|input| {
            let slot = *slots.get(&input.location).ok_or_else(|| {
                format!(
                    "kernel stage input {} is no AIR stage input",
                    input.location
                )
            })?;
            Ok(StageInputFetch {
                location: input.location,
                slot,
                format: input.format,
                stride: input.stride,
                dynamic: input.dynamic_stride,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    out.sort_by_key(|fetch| fetch.location);
    Ok(out)
}

/// The fetches a kernel of AIR-layer interface `interface` implies without an authored case (a
/// census or L4 row), in location order: each `[[stage_in]]` location it declares, read as the
/// format that reads exactly its IR type (32-bit floats, halves, and 32/16/8-bit integers as
/// unsigned, the class the product declares every integer input as, of 1 to 4 lanes) with that
/// format's product storage-buffer stride, at its slot. A location that is not a constant or an IR
/// type no format reads is an error.
pub fn implied(interface: &AirInterface) -> Result<Vec<StageInputFetch>, String> {
    let slots = stage_input_slots(interface)?;
    let mut out = Vec::new();
    for param in &interface.params {
        let ParamRole::StageIn { location } = &param.binding.role else {
            continue;
        };
        let location = constant_int(location.index).ok_or_else(|| {
            format!(
                "kernel stage input at parameter {} has no constant location",
                param.binding.param
            )
        })?;
        let format = param.value.and_then(exact_format).ok_or_else(|| {
            format!(
                "kernel stage input {location} has IR type {:?}, which no format reads exactly",
                param.value
            )
        })?;
        let slot = *slots
            .get(&location)
            .ok_or_else(|| format!("kernel stage input {location} has no slot"))?;
        out.push(StageInputFetch {
            location,
            slot,
            format,
            stride: format.storage_buffer_stride(),
            dynamic: false,
        });
    }
    out.sort_by_key(|fetch| fetch.location);
    Ok(out)
}

/// The stage-input descriptor kernel `entry` of `bitcode` implies ([`implied`] of its interface
/// under the function-constant values `constants` the row compiles with; none: Metal's
/// undefined, the plain census's), `None` when it declares no stage input.
pub fn implied_descriptor(
    bitcode: &[u8],
    entry: &str,
    constants: &FunctionConstantValues,
) -> Result<Option<StageInputDescriptor>, String> {
    let decoded = DecodedModule::from_bitcode(entry, bitcode)
        .map_err(|evidence| format!("the AIR does not decode: {evidence}"))?;
    let abi = decoded
        .abi()
        .map_err(|error| format!("the AIR metadata ABI does not decode: {error}"))?;
    let interface = AirInterface::new(decoded.module(), abi, entry, &bound_constants(constants))?;
    Ok(descriptor(&implied(&interface)?))
}

/// The format that reads exactly a value of IR shape `shape`.
fn exact_format(shape: NumShape) -> Option<AttributeFormat> {
    use AttributeFormat as F;
    let row: [F; 4] = match (shape.float, shape.bits) {
        (true, 32) => [F::Float, F::Float2, F::Float3, F::Float4],
        (true, 16) => [F::Half, F::Half2, F::Half3, F::Half4],
        (false, 32) => [F::Uint, F::Uint2, F::Uint3, F::Uint4],
        (false, 16) => [F::Ushort, F::Ushort2, F::Ushort3, F::Ushort4],
        (false, 8) => [F::Uchar, F::Uchar2, F::Uchar3, F::Uchar4],
        _ => return None,
    };
    let lane = usize::try_from(shape.lanes).ok()?.checked_sub(1)?;
    row.get(lane).copied()
}

/// The product's stage-input descriptor of `fetches`, `None` when there are none (a kernel
/// without stage inputs is compiled without one, as the oracle's pipeline has none).
pub fn descriptor(fetches: &[StageInputFetch]) -> Option<StageInputDescriptor> {
    if fetches.is_empty() {
        return None;
    }
    let mut d = StageInputDescriptor::new();
    for fetch in fetches {
        d.attributes.insert(
            fetch.location,
            StageInputAttributeDescriptor::new(attribute_format(fetch.format), OFFSET, fetch.slot),
        );
        let mut layout = StageInputLayoutDescriptor::new((!fetch.dynamic).then_some(fetch.stride));
        layout.step_function = STEP_FUNCTION;
        layout.step_rate = STEP_RATE;
        d.layouts.insert(fetch.slot, layout);
    }
    Some(d)
}

/// The stride each slot of a dynamic layout among `fetches` binds its data with, by Metal buffer
/// index: the stride form's (`setBuffer:offset:attributeStride:atIndex:`), which both executors
/// use for these slots and the plain form for every other.
pub fn bound_strides(fetches: &[StageInputFetch]) -> BTreeMap<u32, u32> {
    fetches
        .iter()
        .filter(|fetch| fetch.dynamic)
        .map(|fetch| (fetch.slot, fetch.stride))
        .collect()
}

/// The Metal attribute format of an authored format.
pub fn attribute_format(format: AttributeFormat) -> metal::AttributeFormat {
    use metal::AttributeFormat as M;
    use AttributeFormat as F;
    match format {
        F::Char => M::Char,
        F::Char2 => M::Char2,
        F::Char3 => M::Char3,
        F::Char4 => M::Char4,
        F::Uchar => M::UChar,
        F::Uchar2 => M::UChar2,
        F::Uchar3 => M::UChar3,
        F::Uchar4 => M::UChar4,
        F::Short => M::Short,
        F::Short2 => M::Short2,
        F::Short3 => M::Short3,
        F::Short4 => M::Short4,
        F::Ushort => M::UShort,
        F::Ushort2 => M::UShort2,
        F::Ushort3 => M::UShort3,
        F::Ushort4 => M::UShort4,
        F::Half => M::Half,
        F::Half2 => M::Half2,
        F::Half3 => M::Half3,
        F::Half4 => M::Half4,
        F::Float => M::Float,
        F::Float2 => M::Float2,
        F::Float3 => M::Float3,
        F::Float4 => M::Float4,
        F::Uint => M::UInt,
        F::Uint2 => M::UInt2,
        F::Uint3 => M::UInt3,
        F::Uint4 => M::UInt4,
        F::Int => M::Int,
        F::Int2 => M::Int2,
        F::Int3 => M::Int3,
        F::Int4 => M::Int4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::interface::tests::param;
    use metal2vulkan_air::abi::{AbiInt, Access, BufferArg, Slot};

    fn slot(index: u32) -> Slot {
        Slot {
            index: AbiInt::Const(index),
            count: AbiInt::Const(1),
        }
    }

    /// Each authored input is its own buffer at the shared slot rule's slot (the lowest one no
    /// AIR argument occupies, in parameter order), read at offset 0 with its stride, stepping per
    /// thread on x; an undeclared location is an error.
    #[test]
    fn authored_inputs_fetch_one_element_per_thread_from_their_own_slot() {
        let buffer = ParamRole::Buffer(BufferArg {
            location: slot(0),
            access: Access::Read,
            space: metal2vulkan_air::ir::AddrSpace::Device,
            buffer_size: None,
            layout: None,
            raster_order_group: None,
        });
        let interface = AirInterface::synthetic(
            metal2vulkan_air::abi::Stage::Kernel {
                max_total_threads: None,
                work_group_size: None,
            },
            vec![
                param(0, ParamRole::StageIn { location: slot(3) }),
                param(1, buffer),
                param(2, ParamRole::StageIn { location: slot(1) }),
            ],
            vec![],
            vec![],
        );
        let input = |location, format, stride| KernelStageInput {
            location,
            format,
            stride,
            dynamic_stride: false,
            bytes_b64: String::new(),
        };
        let inputs = [
            input(3, AttributeFormat::Float3, 16),
            input(1, AttributeFormat::Ushort2, 4),
        ];
        let fetches = authored(&inputs, &interface).unwrap();
        assert_eq!(
            fetches,
            [
                StageInputFetch {
                    location: 1,
                    slot: 2,
                    format: AttributeFormat::Ushort2,
                    stride: 4,
                    dynamic: false,
                },
                StageInputFetch {
                    location: 3,
                    slot: 1,
                    format: AttributeFormat::Float3,
                    stride: 16,
                    dynamic: false,
                },
            ]
        );
        let d = descriptor(&fetches).unwrap();
        assert_eq!(d.attributes[&3].buffer_index, 1);
        assert_eq!(d.attributes[&3].format, metal::AttributeFormat::Float3);
        assert_eq!(d.attributes[&3].offset, 0);
        assert_eq!(d.layouts[&2].stride, Some(4));
        assert_eq!(d.layouts[&1].step_function, STEP_FUNCTION);
        assert_eq!(d.layouts[&1].step_rate, 1);
        assert_eq!(descriptor(&[]), None);
        assert!(authored(&[input(7, AttributeFormat::Float, 4)], &interface).is_err());
        assert_eq!(bound_strides(&fetches), BTreeMap::new());
        // A dynamic stride: the layout's is `MTLBufferLayoutStrideDynamic`, and the slot binds
        // the data at the case's stride.
        let dynamic = [KernelStageInput {
            dynamic_stride: true,
            ..input(3, AttributeFormat::Float3, 16)
        }];
        let fetches = authored(&dynamic, &interface).unwrap();
        assert!(fetches[0].dynamic);
        assert_eq!(descriptor(&fetches).unwrap().layouts[&1].stride, None);
        assert_eq!(bound_strides(&fetches), BTreeMap::from([(1, 16)]));
    }

    /// Both executors set a case's authored stage-in region, and without one the grid's from the
    /// origin.
    #[test]
    fn the_region_is_the_cases_or_the_grids() {
        assert_eq!(
            region(None, [8, 2, 1]),
            StageInRegion::new([0; 3], [8, 2, 1])
        );
        let authored = HostStageInRegion {
            origin: [2, 1, 0],
            size: [6, 0, 1],
        };
        assert_eq!(
            region(Some(authored), [8, 2, 1]),
            StageInRegion::new([2, 1, 0], [6, 0, 1])
        );
    }

    /// A row's stage inputs read the format of their IR type, unsigned for integers, at the
    /// product storage-buffer stride; a type no format reads is an error.
    #[test]
    fn a_row_implies_the_format_that_reads_its_ir_type() {
        let input = |at, location, float, bits, lanes| {
            let mut p = param(
                at,
                ParamRole::StageIn {
                    location: slot(location),
                },
            );
            p.value = Some(NumShape { float, bits, lanes });
            p
        };
        let kernel = |params| {
            AirInterface::synthetic(
                metal2vulkan_air::abi::Stage::Kernel {
                    max_total_threads: None,
                    work_group_size: None,
                },
                params,
                vec![],
                vec![],
            )
        };
        let interface = kernel(vec![input(0, 2, true, 32, 3), input(1, 0, false, 16, 2)]);
        assert_eq!(
            implied(&interface).unwrap(),
            [
                StageInputFetch {
                    location: 0,
                    slot: 1,
                    format: AttributeFormat::Ushort2,
                    stride: 4,
                    dynamic: false,
                },
                StageInputFetch {
                    location: 2,
                    slot: 0,
                    format: AttributeFormat::Float3,
                    stride: 16,
                    dynamic: false,
                },
            ]
        );
        assert!(implied(&kernel(vec![input(0, 0, false, 64, 1)])).is_err());
        assert!(implied(&kernel(vec![input(0, 0, true, 32, 5)])).is_err());
        assert_eq!(implied(&kernel(vec![])).unwrap(), []);
    }

    /// Every authored format is the Metal format of the same name, whose value `MTLVertexFormat`
    /// shares.
    #[test]
    fn authored_formats_are_metal_formats() {
        use crate::candidate::engine::graphics::vertex_format;
        let mut formats = 0;
        for scalar in [
            "char", "uchar", "short", "ushort", "half", "float", "uint", "int",
        ] {
            for lanes in ["", "2", "3", "4"] {
                let name = format!("{scalar}{lanes}");
                let format = AttributeFormat::from_air_type_name(&name).expect(&name);
                formats += 1;
                assert_eq!(
                    attribute_format(format).as_raw(),
                    vertex_format(format).as_raw(),
                    "{format:?}"
                );
            }
        }
        assert_eq!(formats, 32);
    }
}
