//! The buffer-slot rule both executors share: which Metal buffer-table slots an entry's AIR-layer
//! interface and a case occupy, and where the executor's own buffers go (the kernel `[[stage_in]]`
//! data, vertex-fetched inputs, the fuel and producer-policy buffers). The Metal oracle and the
//! engine candidate read it here, so their slot choices agree.

use crate::case::AuthoredCase;
use crate::check::interface::AirInterface;
use metal2vulkan_air::abi::ParamRole;
use std::collections::BTreeMap;

#[derive(Default)]
pub struct NativeBufferSlots(u32);

impl NativeBufferSlots {
    pub fn reserve(&mut self, base: u32, count: u32) -> Result<(), String> {
        if u64::from(base) + u64::from(count) > 31 {
            return Err(format!(
                "native buffer range {base}+{count} exceeds 31 slots"
            ));
        }
        self.0 |= ((1u32 << count) - 1) << base;
        Ok(())
    }

    /// The highest free slot.
    pub fn allocate(&mut self) -> Result<u32, String> {
        let available = !self.0 & 0x7fff_ffff;
        if available == 0 {
            return Err("native buffer slots are exhausted".into());
        }
        let index = 31 - available.leading_zeros();
        self.0 |= 1 << index;
        Ok(index)
    }

    /// The lowest free slot.
    fn allocate_lowest(&mut self) -> Result<u32, String> {
        let available = !self.0 & 0x7fff_ffff;
        if available == 0 {
            return Err("native buffer slots are exhausted".into());
        }
        let index = available.trailing_zeros();
        self.0 |= 1 << index;
        Ok(index)
    }

    /// The buffer-table slots the entry's AIR arguments occupy: buffers, argument buffers,
    /// acceleration structures, and function tables over their declared ranges. Every
    /// declaration counts, live or not: a slot only a function-constant alternative binds is
    /// still not free for an oracle buffer.
    fn arguments(interface: &AirInterface) -> Result<Self, String> {
        let mut slots = Self::default();
        for param in &interface.params {
            let occupies = matches!(
                param.binding.role,
                ParamRole::Buffer(_)
                    | ParamRole::ArgumentBuffer(_)
                    | ParamRole::AccelerationStructure { .. }
                    | ParamRole::FunctionTable(_)
                    | ParamRole::IntersectionTable(_)
            );
            if let Some((first, count)) = param.slots().filter(|_| occupies) {
                slots.reserve(first, count.max(1))?;
            }
        }
        Ok(slots)
    }

    /// Every slot the entry occupies: its AIR arguments and its kernel `[[stage_in]]` buffers.
    pub fn from_interface(interface: &AirInterface) -> Result<Self, String> {
        let mut slots = Self::arguments(interface)?;
        for slot in stage_input_slots(interface)?.into_values() {
            slots.reserve(slot, 1)?;
        }
        Ok(slots)
    }

    pub fn for_case(case: &AuthoredCase, interface: &AirInterface) -> Result<Self, String> {
        let mut slots = Self::from_interface(interface)?;
        for binding in case
            .buffers
            .iter()
            .map(|r| r.binding)
            .chain(case.acceleration_structures.iter().map(|r| r.binding))
            .chain(case.visible_function_tables.iter().map(|r| r.binding))
            .chain(case.intersection_function_tables.iter().map(|r| r.binding))
        {
            slots.reserve(binding, 1)?;
        }
        for array in &case.device_buffer_arrays {
            slots.reserve(array.binding, array.length)?;
        }
        Ok(slots)
    }
}

/// The buffer slot the oracle binds each kernel `[[stage_in]]` location's data at, by location.
/// AIR gives a kernel stage input no slot; the pipeline's stage-input descriptor names one. Each
/// takes the lowest slot no AIR argument occupies, in parameter order, so the choice is a
/// function of the entry's signature.
pub fn stage_input_slots(interface: &AirInterface) -> Result<BTreeMap<u32, u32>, String> {
    let mut slots = NativeBufferSlots::arguments(interface)?;
    let mut inputs = interface
        .params
        .iter()
        .filter(|param| matches!(param.binding.role, ParamRole::StageIn { .. }))
        .collect::<Vec<_>>();
    inputs.sort_by_key(|param| param.binding.param);
    let mut out = BTreeMap::new();
    for param in inputs {
        let (location, _) = param.slots().ok_or_else(|| {
            format!(
                "kernel stage input at parameter {} has no constant location",
                param.binding.param
            )
        })?;
        if out.insert(location, slots.allocate_lowest()?).is_some() {
            return Err(format!("AIR binds kernel stage input {location} twice"));
        }
    }
    Ok(out)
}

/// The buffer slot each vertex-fetched input binds its data at, by location: the vertex
/// attributes and a tessellation stage's control-point and patch inputs. Every slot the entry
/// and the case occupy ([`NativeBufferSlots::for_case`]) is taken first; then the inputs, in
/// location order, each take the highest free slot.
pub fn vertex_input_slots(
    case: &AuthoredCase,
    interface: &AirInterface,
) -> Result<BTreeMap<u32, u32>, String> {
    let locations = case
        .vertex_inputs
        .iter()
        .chain(case.tessellation.iter().flat_map(|tessellation| {
            tessellation
                .control_points
                .iter()
                .chain(&tessellation.patch_inputs)
        }))
        .map(|input| input.location);
    input_slots(NativeBufferSlots::for_case(case, interface)?, locations)
}

/// The buffer slot each vertex-fetched input at `locations` binds its data at, given the slots
/// already taken (`slots`): in location order, each takes the highest free slot. The one rule
/// behind [`vertex_input_slots`] and the vertex descriptor of an L4 row (`acceptance::row`).
pub fn input_slots(
    mut slots: NativeBufferSlots,
    locations: impl IntoIterator<Item = u32>,
) -> Result<BTreeMap<u32, u32>, String> {
    let mut locations = locations.into_iter().collect::<Vec<_>>();
    locations.sort_unstable();
    let mut out = BTreeMap::new();
    for location in locations {
        if out.insert(location, slots.allocate()?).is_some() {
            return Err(format!("vertex input location {location} is bound twice"));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::interface::tests::{object, of_text};

    #[test]
    fn native_buffer_slots_reserve_full_ranges_and_reject_overflow() {
        let mut slots = NativeBufferSlots::default();
        slots.reserve(29, 2).unwrap();
        assert_eq!(slots.allocate().unwrap(), 28);
        assert!(slots.reserve(u32::MAX, 2).is_err());
        assert!(slots.reserve(0, u32::MAX).is_err());
        assert_eq!(slots.allocate().unwrap(), 27);
        assert_eq!(slots.allocate_lowest().unwrap(), 0);
        slots.reserve(0, 31).unwrap();
        assert!(slots.allocate().is_err());
        assert!(slots.allocate_lowest().is_err());
    }

    #[test]
    fn native_buffer_slots_include_unauthored_array_elements() {
        let mut case = crate::case::narrow_vertex_test_case(String::new(), String::new());
        case.device_buffer_arrays
            .push(crate::case::DeviceBufferArrayResource {
                binding: 28,
                length: 3,
                elements: vec![],
            });
        let interface = AirInterface::synthetic(
            metal2vulkan_air::abi::Stage::Vertex { patch: None },
            vec![],
            vec![],
            vec![],
        );
        let mut slots = NativeBufferSlots::for_case(&case, &interface).unwrap();
        assert_eq!(slots.allocate().unwrap(), 27);
    }

    #[test]
    fn air_arguments_reserve_their_declared_ranges() {
        use crate::check::interface::tests::param;
        use metal2vulkan_air::abi::{AbiInt, Slot};
        let slot = |index, count| Slot {
            index: AbiInt::Const(index),
            count: AbiInt::Const(count),
        };
        let mut table = object(29);
        table.location = slot(29, 2);
        let interface = AirInterface::synthetic(
            metal2vulkan_air::abi::Stage::Kernel {
                max_total_threads: None,
                work_group_size: None,
            },
            vec![
                param(0, ParamRole::FunctionTable(table)),
                param(
                    1,
                    ParamRole::StageIn {
                        location: slot(0, 1),
                    },
                ),
            ],
            vec![],
            vec![],
        );
        let mut slots = NativeBufferSlots::from_interface(&interface).unwrap();
        // 29 and 30 are the table's; 0 is the stage input's.
        assert_eq!(slots.allocate().unwrap(), 28);
        assert_eq!(slots.allocate_lowest().unwrap(), 1);
    }

    #[test]
    fn vertex_inputs_take_the_highest_slots_the_entry_and_case_leave_free() {
        use crate::case::{
            AttributeFormat, AttributeInput, BufferResource, DeviceBufferArrayResource,
            ResourceRole,
        };
        use crate::check::interface::tests::{param, vertex_buffers_at};
        use metal2vulkan_air::abi::{AbiInt, Access, BufferArg, Slot};
        let mut case = crate::case::narrow_vertex_test_case(String::new(), "v".into());
        case.buffers.push(BufferResource {
            binding: 30,
            role: ResourceRole::Input,
            bytes_b64: Some("AAAAAA==".into()),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        });
        case.device_buffer_arrays.push(DeviceBufferArrayResource {
            binding: 26,
            length: 3,
            elements: vec![],
        });
        case.vertex_inputs.push(AttributeInput {
            location: 5,
            format: AttributeFormat::Float,
            stride: 4,
            dynamic_stride: false,
            bytes_b64: "AAAAAA==".into(),
        });
        // The case holds 0, 30, and 26..=28; the entry's buffer holds 25.
        assert_eq!(
            vertex_input_slots(&case, &vertex_buffers_at(&[25])).unwrap(),
            BTreeMap::from([(0, 29), (1, 24), (5, 23)])
        );

        // An argument buffer, a gated-off alternative, and every slot of a multi-slot range are
        // the entry's too, though reflection names none of them as a buffer slot.
        let slot = |index, count| Slot {
            index: AbiInt::Const(index),
            count: AbiInt::Const(count),
        };
        let buffer = |location| BufferArg {
            location,
            access: Access::Read,
            space: metal2vulkan_air::ir::AddrSpace::Device,
            buffer_size: None,
            layout: None,
            raster_order_group: None,
        };
        let mut gated_off = param(1, ParamRole::Buffer(buffer(slot(24, 1))));
        gated_off.live = false;
        let mut table = object(21);
        table.location = slot(21, 3);
        let interface = AirInterface::synthetic(
            metal2vulkan_air::abi::Stage::Vertex { patch: None },
            vec![
                param(0, ParamRole::ArgumentBuffer(buffer(slot(29, 1)))),
                gated_off,
                param(2, ParamRole::FunctionTable(table)),
            ],
            vec![],
            vec![],
        );
        assert_eq!(
            vertex_input_slots(&case, &interface).unwrap(),
            BTreeMap::from([(0, 25), (1, 20), (5, 19)])
        );

        case.device_buffer_arrays[0].length = 6;
        assert!(vertex_input_slots(&case, &vertex_buffers_at(&[])).is_err());
    }

    #[test]
    fn kernel_stage_inputs_take_the_lowest_slots_the_air_leaves_free() {
        // The output buffer holds slot 0, so location 0's data goes to slot 1.
        let ll = include_str!("../../fixtures/public/kernel_stage_input_word.ll");
        let interface = of_text(ll, "stage_input_word", &BTreeMap::new()).unwrap();
        assert_eq!(
            stage_input_slots(&interface).unwrap(),
            BTreeMap::from([(0, 1)])
        );
        let mut slots = NativeBufferSlots::from_interface(&interface).unwrap();
        assert_eq!(slots.allocate_lowest().unwrap(), 2);
    }
}
