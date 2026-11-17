//! Lane reads end to end: on a profile taking `DriverWorkaround::UniformBufferPerLaneReadsSerialize`
//! a `constant` slot a uniform view could read keeps its storage view where some read reaches it
//! at an address that may differ between the lanes that run it together, and keeps its uniform
//! view where every read is at a uniform address; a profile without the fact gives every such slot
//! its uniform view. The fact comes from the description's measured set alone (the runtime's view
//! check), never from the GPU's identity.
//!
//! The kernels read a `constant` table of 64 words (`air.buffer_size` 256, a `Bounded` window):
//! `lane_table` at the thread's position, `loop_table` in a loop over a bound read from the table,
//! after an early return of the threads past it (a counter every lane that runs the loop holds
//! alike), and `field_table` at one constant offset.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::path::Path;

use crate::device::profile::{DeviceProfile, DriverWorkaround};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::pipeline::interface::{BufferForm, DescriptorView, SlotInterface};
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{compile, link_and_validate, llvm_assemble_text, AssembleError};

const TABLES: &str = r#"define void @lane_table(ptr addrspace(2) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, i32 noundef %tid) {
entry:
  %i = and i32 %tid, 63
  %ix = zext i32 %i to i64
  %p = getelementptr inbounds [64 x i32], ptr addrspace(2) %0, i64 0, i64 %ix
  %v = load i32, ptr addrspace(2) %p, align 4
  %tx = zext i32 %tid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %tx
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}
define void @loop_table(ptr addrspace(2) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, i32 noundef %tid) {
entry:
  %n = load i32, ptr addrspace(2) %0, align 4
  %n6 = and i32 %n, 63
  %in = icmp ult i32 %tid, %n6
  br i1 %in, label %head, label %exit
head:
  %i = phi i32 [ 1, %entry ], [ %i1, %head ]
  %acc = phi i32 [ 0, %entry ], [ %acc1, %head ]
  %ix = zext i32 %i to i64
  %p = getelementptr inbounds [64 x i32], ptr addrspace(2) %0, i64 0, i64 %ix
  %v = load i32, ptr addrspace(2) %p, align 4
  %acc1 = add i32 %acc, %v
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, %n6
  br i1 %more, label %head, label %done
done:
  %tx = zext i32 %tid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %tx
  store i32 %acc1, ptr addrspace(1) %o, align 4
  br label %exit
exit:
  ret void
}
define void @field_table(ptr addrspace(2) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, i32 noundef %tid) {
entry:
  %p = getelementptr inbounds [64 x i32], ptr addrspace(2) %0, i64 0, i64 3
  %v = load i32, ptr addrspace(2) %p, align 4
  %tx = zext i32 %tid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %tx
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}
!air.kernel = !{!1, !2, !3}
!air.version = !{!17}
!air.language_version = !{!18}
!1 = !{ptr @lane_table, !10, !11}
!2 = !{ptr @loop_table, !10, !11}
!3 = !{ptr @field_table, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 256, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 256, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Table", !"air.arg_name", !"table"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

fn assemble() -> Option<Vec<u8>> {
    match llvm_assemble_text(TABLES) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the tables: {why}"),
    }
}

/// The profile of the committed capture `stem`, under the device ID `id` where one is given, its
/// description's measured set `measured`.
fn profile(stem: &str, id: Option<u32>, measured: &[DriverWorkaround]) -> DeviceProfile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if let Some(id) = id {
        desc.device_id = id;
    }
    desc.measured.extend(measured.iter().copied());
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The form of the table's slot (Metal buffer 0) in `entry` on `p`, the module validated.
fn table_form(bc: &[u8], entry: &str, p: &DeviceProfile, what: &str) -> BufferForm {
    let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{what} {entry}: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what} {entry}: L3: {e}"));
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index: 0,
    };
    match a.interface.slots.get(&slot) {
        Some(SlotInterface::Buffer(b)) => b.form,
        other => panic!("{what} {entry}: {other:?}"),
    }
}

/// Where the description measured the fact, NVIDIA's capture or RADV's, the table read at the
/// thread's position keeps its storage view; the loop's counter after an early return, and a
/// constant offset, keep the uniform view.
#[test]
fn a_per_lane_constant_read_keeps_its_storage_view_where_the_fact_was_measured() {
    let Some(bc) = assemble() else {
        return;
    };
    let fact = DriverWorkaround::UniformBufferPerLaneReadsSerialize;
    let storage = BufferForm::Descriptor(DescriptorView::Storage);
    let uniform = BufferForm::Descriptor(DescriptorView::UniformWords);
    for what in ["nvidia-rtx5080-laptop", "amd-780m-radv"] {
        let p = profile(what, None, &[fact]);
        assert!(p.workarounds().contains(&fact), "{what}");
        assert_eq!(table_form(&bc, "lane_table", &p, what), storage, "{what}");
        assert_eq!(table_form(&bc, "loop_table", &p, what), uniform, "{what}");
        assert_eq!(table_form(&bc, "field_table", &p, what), uniform, "{what}");
    }
}

/// The identity decides nothing: NVIDIA's captures under the IDs of the three GPUs the fact was
/// first measured on (the RTX 5080 Laptop GPU, the RTX 3090 and the RTX 2080), with nothing
/// measured, give every kernel's table its uniform view; and so does RADV, the negative control.
#[test]
fn without_the_measured_fact_a_per_lane_constant_read_keeps_its_uniform_view() {
    let Some(bc) = assemble() else {
        return;
    };
    let uniform = BufferForm::Descriptor(DescriptorView::UniformWords);
    for (stem, id) in [
        ("nvidia-rtx5080-laptop", None),
        ("nvidia-rtx5080-laptop", Some(0x2204)),
        ("nvidia-rtx5080-laptop", Some(0x1e87)),
        ("amd-780m-radv", None),
        ("amd-rdna3-radv", None),
    ] {
        let p = profile(stem, id, &[]);
        assert!(!p
            .workarounds()
            .contains(&DriverWorkaround::UniformBufferPerLaneReadsSerialize));
        for entry in ["lane_table", "loop_table", "field_table"] {
            assert_eq!(
                table_form(&bc, entry, &p, stem),
                uniform,
                "{stem} {id:?} {entry}"
            );
        }
    }
}

/// Both view facts compose: the table read at the thread's position keeps its storage view (the
/// lane read first), the loop's indexed reads, uniform among the active lanes, take the uniform
/// texel buffer, and the constant offset alone keeps the uniform buffer.
#[test]
fn both_view_facts_compose() {
    let Some(bc) = assemble() else {
        return;
    };
    let both = [
        DriverWorkaround::UniformBufferPerLaneReadsSerialize,
        DriverWorkaround::ExternalUniformReadsSkipACache,
    ];
    let p = profile("nvidia-rtx5080-laptop", None, &both);
    let what = "both";
    let form = |entry| table_form(&bc, entry, &p, what);
    assert_eq!(
        form("lane_table"),
        BufferForm::Descriptor(DescriptorView::Storage)
    );
    assert_eq!(
        form("loop_table"),
        BufferForm::Descriptor(DescriptorView::UniformTexel)
    );
    assert_eq!(
        form("field_table"),
        BufferForm::Descriptor(DescriptorView::UniformWords)
    );
}
