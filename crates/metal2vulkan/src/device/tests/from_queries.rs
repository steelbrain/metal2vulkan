//! `Device::from_queries` decides by Vulkan queries alone: the description's identity changes
//! nothing it derives.

use super::*;

/// The identities each committed profile is also read under, each with a device type: MoltenVK
/// on an Apple GPU, the NVIDIA driver on a GA10x, RADV, Intel's Mesa driver and none.
fn identities() -> [(Option<vk::DriverId>, u32, u32, vk::PhysicalDeviceType); 5] {
    use vk::PhysicalDeviceType as T;
    [
        (
            Some(vk::DriverId::MOLTENVK),
            0x106b,
            0x1_0009,
            T::INTEGRATED_GPU,
        ),
        (
            Some(vk::DriverId::NVIDIA_PROPRIETARY),
            0x10de,
            0x2204,
            T::DISCRETE_GPU,
        ),
        (
            Some(vk::DriverId::MESA_RADV),
            0x1002,
            0x744c,
            T::DISCRETE_GPU,
        ),
        (
            Some(vk::DriverId::INTEL_OPEN_SOURCE_MESA),
            0x8086,
            0x9a49,
            T::INTEGRATED_GPU,
        ),
        (None, 0, 0, T::CPU),
    ]
}

/// On every committed profile, descriptions that differ only in `driver_id`, `vendor_id`,
/// `device_id` and the device type give one profile, whose only workarounds are the two every
/// device but Apple's takes and which reads YCbCr planes in place.
#[test]
fn from_queries_derives_one_profile_whatever_the_identity() {
    for (stem, _) in committed_profiles() {
        let desc = profile(&stem);
        let derived: Vec<Option<DeviceProfile>> = identities()
            .into_iter()
            .map(|(driver, vendor, device, kind)| {
                let mut other = desc.clone();
                other.driver_id = driver;
                other.vendor_id = vendor;
                other.device_id = device;
                other.device_type = Some(kind);
                Device::from_queries(other, DevicePolicy::default())
                    .ok()
                    .map(|d| d.profile().clone())
            })
            .collect();
        for profile in &derived[1..] {
            assert_eq!(profile, &derived[0], "{stem}");
        }
        let Some(queried) = &derived[0] else {
            continue;
        };
        assert_eq!(
            queried.workarounds(),
            [
                DriverWorkaround::ClippedCoverageNotMetals,
                DriverWorkaround::TessellatorNotMetals
            ],
            "{stem}"
        );
        assert!(queried.fields().ycbcr_in_place, "{stem}");
    }
}

/// The owned synthetic kernel of `from_queries_keeps_the_narrow_mask_on_ga10x`: two loaded words'
/// low 16 bits sign-extended, summed, clamped by `llvm.smin` then `llvm.smax` to exactly 16 bits'
/// range, truncated and zero-extended.
const NARROW_CLAMP: &str = r#"; Owned synthetic kernel.
source_filename = "narrow_clamp.metal"

declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)

define void @narrow_clamp(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {
entry:
  %w = zext i32 %gid to i64
  %xp = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %w
  %x = load i32, ptr addrspace(1) %xp, align 4
  %yp = getelementptr inbounds i32, ptr addrspace(1) %xp, i64 1
  %y = load i32, ptr addrspace(1) %yp, align 4
  %x16 = trunc i32 %x to i16
  %y16 = trunc i32 %y to i16
  %a = sext i16 %x16 to i32
  %b = sext i16 %y16 to i32
  %s = add nsw i32 %a, %b
  %s1 = call i32 @llvm.smin.i32(i32 %s, i32 32767)
  %s2 = call i32 @llvm.smax.i32(i32 %s1, i32 -32768)
  %t = trunc i32 %s2 to i16
  %z = zext i16 %t to i32
  %op = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %w
  store i32 %z, ptr addrspace(1) %op, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @narrow_clamp, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// The owned synthetic kernel of `from_queries_records_the_device_denormals_of_a_float_product`:
/// a `float` product and a `half` product, each stored back.
const MIXED_PRODUCTS: &str = r#"; Owned synthetic kernel.
source_filename = "mixed_products.metal"

define void @mixed_products(ptr addrspace(1) %f, ptr addrspace(1) %h, i32 %gid) {
entry:
  %w = zext i32 %gid to i64
  %fp = getelementptr inbounds float, ptr addrspace(1) %f, i64 %w
  %x = load float, ptr addrspace(1) %fp, align 4
  %y = fmul float %x, %x
  store float %y, ptr addrspace(1) %fp, align 4
  %hp = getelementptr inbounds half, ptr addrspace(1) %h, i64 %w
  %a = load half, ptr addrspace(1) %hp, align 2
  %b = fmul half %a, %a
  store half %b, ptr addrspace(1) %hp, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @mixed_products, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"f"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"h"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// On every committed profile `Device::from_queries` derives, a kernel's `float` product beside a
/// `half` one ([`MIXED_PRODUCTS`]) records `PrecisionNote::DeviceDenormals` and requires no
/// `VkProperty::MetalFloatDenormals`, whether its entry declares the 16-bit float controls (where
/// the device computes the `half` product in 16 bits) or not: no Vulkan query states that a
/// device reads an `f32` denormal as Metal does, and a driver may keep one under the 16-bit
/// controls, so the note is static.
#[test]
fn from_queries_records_the_device_denormals_of_a_float_product() {
    use crate::device::requirements::VkProperty;
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::testing::{compile, llvm_assemble_text, AssembleError};
    let bc = match llvm_assemble_text(MIXED_PRODUCTS) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(error) => panic!("mixed_products: {error:?}"),
    };
    let mut compiled = 0;
    for (stem, _) in committed_profiles() {
        let Ok(device) = Device::from_queries(profile(&stem), DevicePolicy::default()) else {
            continue;
        };
        let a = compile(&bc, "mixed_products", device.profile())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            a.precision.contains(&PrecisionNote::DeviceDenormals),
            "{stem}"
        );
        assert!(
            !(a.requirements.properties).contains(&VkProperty::MetalFloatDenormals),
            "{stem}"
        );
        compiled += 1;
    }
    assert!(compiled >= 10, "{compiled} profiles compiled the kernel");
}

/// A narrow mask over a signed clamp, to exactly 16 bits' range, of a sum of two values
/// sign-extended from 16 bits ([`NARROW_CLAMP`]), on an NVIDIA Ampere GA10x description (the RTX
/// 50's capture under the RTX 3090's identity): `Device::from_queries` clears its high bits with
/// `OpBitwiseAnd`, as on every device, and emits no `OpBitFieldUExtract` for it.
#[test]
fn from_queries_clears_a_narrow_value_by_an_and_on_ga10x() {
    use crate::budget::SPIRV_WORDS;
    use crate::lower::select::LowerInputs;
    use crate::pipeline::compile::{compile_bitcode_kept, EntryRef};
    use crate::pipeline::testing::{llvm_assemble_text, meter, AssembleError};
    use crate::spirv::ir::{BinOp, Inst};
    let bc = match llvm_assemble_text(NARROW_CLAMP) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(error) => panic!("narrow_clamp: {error:?}"),
    };
    let mut desc = profile("nvidia-rtx50-linux");
    desc.driver_id = Some(vk::DriverId::NVIDIA_PROPRIETARY);
    desc.vendor_id = 0x10de;
    desc.device_id = 0x2204;
    // The bitwise ANDs and the unsigned extracts of the kernel compiled on `p`.
    let counts = |p: &DeviceProfile| {
        let entry = EntryRef::Name("narrow_clamp");
        let inputs = LowerInputs::new();
        let compiled = compile_bitcode_kept(&bc, entry, p, &inputs, &mut meter(), SPIRV_WORDS);
        let (_, _, module) = compiled.expect("compiled");
        let insts = || module.blocks().iter().flat_map(|b| b.insts.iter());
        let ands = insts()
            .filter(|n| {
                matches!(
                    n.inst,
                    Inst::Binary {
                        op: BinOp::BitwiseAnd,
                        ..
                    }
                )
            })
            .count();
        let extracts = insts()
            .filter(|n| matches!(n.inst, Inst::BitFieldExtract { signed: false, .. }))
            .count();
        (ands, extracts)
    };
    let queried = Device::from_queries(desc, DevicePolicy::default()).unwrap();
    let (ands, extracts) = counts(queried.profile());
    assert!(ands > 0, "from_queries: no AND");
    assert_eq!(extracts, 0, "from_queries: an extract");
}
