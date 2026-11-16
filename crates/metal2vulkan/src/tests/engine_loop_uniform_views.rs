//! A kernel's read-only slot read in a loop at a per-instance uniform offset, whose memory no
//! write of the dispatch reaches, is a uniform view on a profile that takes
//! `DriverWorkaround::UniformAddressStorageReadsSlow` (the Iris Xe RPL-P under ANV), and a storage
//! view everywhere else (`lower::access::loop_uniform_views`).
//!
//! The kernel is a four-tap convolution of our own: `out[gid] = sum(weights[i] * source[gid +
//! i])`, the counter `i` running from 0 to 3 alike in every thread. Its weights are read at the
//! counter, its source at a per-thread offset, and its output written. Each test pins one part:
//! the form and the set-0 descriptor type on the profile that takes the workaround and on the one
//! of the same driver and capabilities that does not (the UHD Graphics 770 of RPL-S); the two
//! proofs the uniform view needs from the function (an unaliased argument, nothing writing it);
//! the variant a shape binding more than `maxUniformBufferRange` asks for; and, on the CPU
//! interpreter, the same bytes through either view.
//!
//! `llvm-as` assembles the kernel; without it the tests skip visibly.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::bind::{BindState, BoundBuffer, DescriptorWrite, StageTable};
use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy, DriverWorkaround};
use crate::library::Library;
use crate::pipeline::interface::{BufferForm, DescriptorView, SlotInterface};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{llvm_assemble, typed_module_with, AssembleError, Scratch};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, RuntimeFact,
    StageSource, UniformViewChoice,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// The committed profile that takes the workaround, and one of the same driver and capabilities
/// that does not.
const IRIS: &str = "intel-irisxe-rplp-anv";
const RPLS: &str = "intel-uhd770-rpls-anv";

const ENTRY: &str = "tap_loop";

/// The kernel, its weights argument declared with `weights_attrs` and `weights_access`, its exit
/// running `exit_extra` before its store.
fn kernel(weights_attrs: &str, weights_access: &str, exit_extra: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @tap_loop(ptr addrspace(1) noundef {weights_attrs} %0, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %1, ptr addrspace(1) noundef "air-buffer-no-alias" %2, i32 noundef %3) {{
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %sum, %loop ]
  %iz = zext i32 %i to i64
  %wp = getelementptr inbounds float, ptr addrspace(1) %0, i64 %iz
  %w = load float, ptr addrspace(1) %wp, align 4
  %x = add i32 %3, %i
  %xz = zext i32 %x to i64
  %sp = getelementptr inbounds float, ptr addrspace(1) %1, i64 %xz
  %s = load float, ptr addrspace(1) %sp, align 4
  %m = fmul float %w, %s
  %sum = fadd float %acc, %m
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop

exit:
{exit_extra}  %gz = zext i32 %3 to i64
  %dp = getelementptr inbounds float, ptr addrspace(1) %2, i64 %gz
  store float %sum, ptr addrspace(1) %dp, align 4
  ret void
}}

!air.kernel = !{{!9}}
!air.version = !{{!17}}
!air.language_version = !{{!18}}
!9 = !{{ptr @tap_loop, !10, !11}}
!10 = !{{}}
!11 = !{{!12, !13, !14, !15}}
!12 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"{weights_access}", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"weights"}}
!13 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"source"}}
!14 = !{{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!15 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!17 = !{{i32 2, i32 8, i32 0}}
!18 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The kernel as the tests read it: its weights an unaliased `device const float *`.
fn tap_loop() -> String {
    kernel(r#"readonly "air-buffer-no-alias""#, "air.read", "")
}

/// `text` assembled through a scratch file this test owns and removes.
fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-loop-uniform-views-{name}"));
    let ll = scratch.path().join(format!("{name}.ll"));
    std::fs::write(&ll, text).expect("write");
    match llvm_assemble(&ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn sref(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// The kernel's pipeline on `dev`, compiled under `inputs`.
fn pipeline(dev: &Device, bc: &[u8], inputs: &LoweringInputs) -> ComputePipeline {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(ENTRY).expect("function");
    let compiled = dev
        .compile_function(&FunctionDescriptor::new(f), inputs, &Budget::standard())
        .expect("compile");
    dev.compile_compute_pipeline(
        &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
        &Budget::standard(),
    )
    .expect("pipeline")
}

fn form(p: &ComputePipeline, index: u32) -> BufferForm {
    match p.interface.slots.get(&sref(index)) {
        Some(SlotInterface::Buffer(b)) => b.form,
        other => panic!("slot {index}: {other:?}"),
    }
}

fn has_fact(p: &ComputePipeline, fact: RuntimeFact) -> bool {
    p.dependencies.iter().any(|d| d.fact == fact)
}

/// A bound buffer of `length` bytes at a GPU address of its own.
fn buf(index: u32, length: u64) -> BoundBuffer {
    BoundBuffer::Buffer {
        length,
        offset: 0,
        gpu_address: 0x10_0000_0000 + u64::from(index) * 0x1_0000_0000,
        device_address: None,
        attribute_stride: None,
    }
}

/// The kernel's bind state: its weights `weights` bytes long, its source and output 1 KiB.
fn state(weights: u64) -> BindState {
    let mut t = StageTable::default();
    t.buffers.insert(0, buf(0, weights));
    t.buffers.insert(1, buf(1, 1024));
    t.buffers.insert(2, buf(2, 1024));
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, t);
    state
}

/// The set-0 descriptor type the plan of `p` for `shape` writes at buffer slot `index`.
fn set0_type(
    p: &ComputePipeline,
    dev: &Device,
    s: &BindState,
    index: u32,
) -> crate::vk::DescriptorType {
    let plan = p.plan(dev, &s.shape()).expect("plan");
    let source = |write: &DescriptorWrite| {
        (p.layout.sets.iter())
            .filter(|set| set.set == write.set)
            .flat_map(|set| &set.bindings)
            .find(|b| b.binding == write.binding)
            .map(|b| b.source)
    };
    (plan.descriptors.iter())
        .find(|w| w.set == 0 && source(w) == Some(BindingSource::Buffer(sref(index))))
        .map(|w| w.ty)
        .unwrap_or_else(|| panic!("no set-0 write of slot {index}"))
}

/// On the Iris the weights are a uniform view whose set-0 binding is a `UNIFORM_BUFFER`, its
/// window (an unsized pointer read at the counter) unbounded, so it states its range fact as
/// `Viewed` and no `BufferRangeLimit`; the source, read at a per-thread offset, and the written
/// output stay storage views. On the RPL-S, the same driver and capabilities without the
/// workaround, the weights are a storage view and a `STORAGE_BUFFER`, as before.
#[test]
fn a_loop_uniform_slot_is_a_uniform_view_where_the_profile_takes_the_workaround() {
    let Some(bc) = assemble("tap_loop", &tap_loop()) else {
        return;
    };
    let uniform = BufferForm::Descriptor(DescriptorView::UniformWords);
    let storage = BufferForm::Descriptor(DescriptorView::Storage);
    let small = state(16);
    let iris = device(IRIS);
    assert!(iris
        .profile()
        .workarounds()
        .contains(&DriverWorkaround::UniformAddressStorageReadsSlow));
    let p = pipeline(&iris, &bc, &LoweringInputs::new());
    assert_eq!(
        [form(&p, 0), form(&p, 1), form(&p, 2)],
        [uniform, storage, storage],
        "{IRIS}"
    );
    assert!(has_fact(
        &p,
        RuntimeFact::UniformViewRange {
            slot: sref(0),
            choice: UniformViewChoice::Viewed,
        }
    ));
    assert!(!has_fact(
        &p,
        RuntimeFact::BufferRangeLimit { slot: sref(0) }
    ));
    assert_eq!(
        set0_type(&p, &iris, &small, 0),
        crate::vk::DescriptorType::UNIFORM_BUFFER
    );
    assert_eq!(
        set0_type(&p, &iris, &small, 1),
        crate::vk::DescriptorType::STORAGE_BUFFER
    );

    let rpls = device(RPLS);
    assert!(!rpls
        .profile()
        .workarounds()
        .contains(&DriverWorkaround::UniformAddressStorageReadsSlow));
    let p = pipeline(&rpls, &bc, &LoweringInputs::new());
    assert_eq!(
        [form(&p, 0), form(&p, 1), form(&p, 2)],
        [storage, storage, storage],
        "{RPLS}"
    );
    assert!(!p
        .dependencies
        .iter()
        .any(|d| matches!(d.fact, RuntimeFact::UniformViewRange { .. })));
    assert_eq!(
        set0_type(&p, &rpls, &small, 0),
        crate::vk::DescriptorType::STORAGE_BUFFER
    );
}

/// The uniform view needs both of the function's proofs that no write of the dispatch reaches
/// the weights: on the Iris, weights whose argument the AIR does not declare unaliased (no
/// `air-buffer-no-alias`), and weights the kernel also writes, stay storage views.
#[test]
fn a_loop_uniform_view_needs_an_unaliased_argument_nothing_writes() {
    let iris = device(IRIS);
    let storage = BufferForm::Descriptor(DescriptorView::Storage);
    let cases = [
        ("aliased", kernel("readonly", "air.read", "")),
        (
            "written",
            kernel(
                r#""air-buffer-no-alias""#,
                "air.read_write",
                "  store float %sum, ptr addrspace(1) %0, align 4\n",
            ),
        ),
    ];
    for (what, text) in cases {
        let Some(bc) = assemble(what, &text) else {
            return;
        };
        let p = pipeline(&iris, &bc, &LoweringInputs::new());
        assert_eq!(form(&p, 0), storage, "{what}");
        assert!(
            !has_fact(
                &p,
                RuntimeFact::UniformViewRange {
                    slot: sref(0),
                    choice: UniformViewChoice::Viewed,
                }
            ),
            "{what}"
        );
    }
}

/// A shape binding the weights a buffer whose length class does not fit `maxUniformBufferRange`
/// (1 GiB on the Iris) asks for the variant that keeps them a storage view
/// (`VariantKey::storage_view_slots`); a small one asks for none. The variant's weights are a
/// storage view stated `Kept`, and its plan for the same shape states the same key.
#[test]
fn a_large_buffer_at_a_loop_uniform_view_asks_for_its_storage_variant() {
    let Some(bc) = assemble("tap_loop_large", &tap_loop()) else {
        return;
    };
    let iris = device(IRIS);
    let base = pipeline(&iris, &bc, &LoweringInputs::new());
    let small = base.plan(&iris, &state(16).shape()).expect("plan");
    assert!(small.variant.is_empty(), "small weights need no variant");
    let large = state(2 << 30);
    let plan = base.plan(&iris, &large.shape()).expect("plan");
    assert_eq!(plan.variant.storage_view_slots, BTreeSet::from([sref(0)]));
    let inputs = plan.variant.lowering_inputs(Stage::Kernel);
    let variant = pipeline(&iris, &bc, &inputs);
    assert_eq!(
        form(&variant, 0),
        BufferForm::Descriptor(DescriptorView::Storage)
    );
    assert!(has_fact(
        &variant,
        RuntimeFact::UniformViewRange {
            slot: sref(0),
            choice: UniformViewChoice::Kept,
        }
    ));
    let again = variant.plan(&iris, &large.shape()).expect("variant plan");
    assert_eq!(again.variant, plan.variant, "a fixed point");
}

/// On the CPU interpreter, eight threads of the Iris's base module (the weights' uniform view)
/// and of its storage-view variant write the same output, the convolution's own.
#[test]
fn a_loop_uniform_view_reads_the_bytes_the_storage_view_does() {
    let Some(bc) = assemble("tap_loop_bytes", &tap_loop()) else {
        return;
    };
    let iris = device(IRIS);
    let p = iris.profile();
    let weights = [0.5f32, -1.25, 2.0, 0.75];
    let source: Vec<f32> = (0..16).map(|k| k as f32 * 0.5 - 3.0).collect();
    let want: Vec<f32> = (0..8)
        .map(|g| (0..4).fold(0.0f32, |acc, i| acc + weights[i] * source[g + i]))
        .collect();
    let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let kslot = |index| ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    };
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), 8);
    spec.insert(SpecKey::LocalSize(Axis::Y), 1);
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(SpecKey::FlatLocalSize, 8);
    let mut variant = LoweringInputs::new();
    variant.storage_view_slots.insert(sref(0));
    let mut outs = Vec::new();
    for (what, inputs, view) in [
        ("base", LoweringInputs::new(), "uniform"),
        ("variant", variant, "storage"),
    ] {
        let (_, m) = typed_module_with(&bc, ENTRY, p, &inputs).expect("module");
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(kslot(0)),
                    address: None,
                    bytes: bytes(&weights),
                },
                Buffer {
                    slot: Some(kslot(1)),
                    address: None,
                    bytes: bytes(&source),
                },
                Buffer {
                    slot: Some(kslot(2)),
                    address: None,
                    bytes: vec![0xEE; 32],
                },
            ],
            push: vec![0; 128],
            spec: spec.clone(),
        };
        let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let got: Vec<f32> = out.buffers[2]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        assert_eq!(got, want, "{what}: the weights through a {view} view");
        outs.push(out.buffers[2].clone());
    }
    assert_eq!(outs[0], outs[1], "the two views' bytes");
}
