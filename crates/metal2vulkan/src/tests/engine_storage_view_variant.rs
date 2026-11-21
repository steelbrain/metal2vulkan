//! The variant a plan names for Profile F's per-stage storage-buffer fallback
//! (`VariantKey::storage_view_slots`) is a fixed point: the plan of the pipeline compiled under a
//! plan's key, for the same shape, states the same key (`BindingPlan::variant`'s contract).
//!
//! A synthetic vertex function (four read-only `device` arrays and a `[[stage_in]]` attribute)
//! fits Profile F's four storage buffers per stage (`runtime-floor` without device addresses, a
//! device below the runtime's floor) until a shape's unaligned vertex buffer asks
//! for the pulled-layout variant, whose pulled source is a fifth storage buffer; the owned kernel
//! `descriptor_limit_uniform_views` needs the fallback on its own. Every shape binds each array
//! small or large (a length class above `maxUniformBufferRange`).

use std::collections::BTreeSet;

use crate::bind::{BindState, BoundBuffer, StageTable};
use crate::budget::Budget;
use crate::device::{Device, DevicePolicy};
use crate::error::{Error, Feature};
use crate::library::Library;
use crate::metal::VertexFormat;
use crate::pipeline::descriptor::{
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::interface::{BufferForm, DescriptorView, SlotInterface};
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError, Scratch};
use crate::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, RenderPipeline,
    RenderPipelineDescriptor, RuntimeFact, StageSource, UniformViewChoice,
};

/// Profile F: `runtime-floor` without device addresses, below the runtime's floor.
fn profile_f() -> Device {
    crate::pipeline::testing::device_f(DevicePolicy::RESIDENT_TABLES).expect("Profile F")
}

fn assembled(result: Result<Vec<u8>, AssembleError>, name: &str) -> Option<Vec<u8>> {
    match result {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// `text` assembled through a scratch file this test owns and removes.
fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-storage-view-variant-{name}"));
    let dir = scratch.path();
    let ll = dir.join(format!("{name}.ll"));
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    assembled(r, name)
}

const VERTEX: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float> }> @probe(<4 x float> %0, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %3, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %4, i32 noundef %5) local_unnamed_addr #0 {
  %z = zext i32 %5 to i64
  %p1 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %z
  %v1 = load float, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %z
  %v2 = load float, ptr addrspace(1) %p2, align 4
  %p3 = getelementptr inbounds float, ptr addrspace(1) %3, i64 %z
  %v3 = load float, ptr addrspace(1) %p3, align 4
  %p4 = getelementptr inbounds float, ptr addrspace(1) %4, i64 %z
  %v4 = load float, ptr addrspace(1) %p4, align 4
  %s1 = fadd float %v1, %v2
  %s2 = fadd float %v3, %v4
  %s = fadd float %s1, %s2
  %x = extractelement <4 x float> %0, i64 0
  %xs = fadd float %x, %s
  %out = insertelement <4 x float> %0, float %xs, i64 0
  %r = insertvalue <{ <4 x float> }> undef, <4 x float> %out, 0
  ret <{ <4 x float> }> %r
}

attributes #0 = { nounwind }

!air.vertex = !{!9}
!llvm.module.flags = !{!3, !4, !5, !6, !7, !8}
!air.version = !{!73}
!air.language_version = !{!74}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @probe, !10, !16}
!10 = !{!11}
!11 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!16 = !{!17, !21, !22, !23, !24, !25}
!17 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!21 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!22 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!23 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
!24 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"d"}
!25 = !{i32 5, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!73 = !{i32 2, i32 8, i32 0}
!74 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

fn buf(length: u64, offset: u64, gpu: u64) -> BoundBuffer {
    BoundBuffer::Buffer {
        length,
        offset,
        gpu_address: gpu,
        device_address: None,
        attribute_stride: None,
    }
}

fn slot(stage: Stage, index: u32) -> SlotRef {
    SlotRef {
        stage,
        class: SlotClass::Buffer,
        index,
    }
}

/// A buffer length that fits a uniform range, and one whose length class does not.
const SMALL: u64 = 1024;
const LARGE: u64 = 65536;

/// The vertex pipeline: position from layout 0 (stride 16), arrays at buffers 1 to 4.
fn vertex_descriptor(dev: &Device, bc: &[u8]) -> (RenderPipelineDescriptor, RenderPipeline) {
    let lib = Library::from_air(bc).expect("library");
    let v = lib.function("probe").expect("probe");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
    let mut vd = VertexDescriptor::new();
    vd.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(16)));
    vd.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 0),
    );
    d.vertex_descriptor = Some(vd);
    let base = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("base compile");
    (d, base)
}

/// The vertex shape: the vertex buffer at bind offset `offset`, and array `k` (buffer `k + 1`)
/// of `lengths[k]` bytes.
fn vertex_state(offset: u64, lengths: [u64; 4]) -> BindState {
    let mut t = StageTable::default();
    t.buffers.insert(0, buf(16 * 64, offset, 0x10_0000_0000));
    for (k, len) in lengths.iter().enumerate() {
        let index = k as u32 + 1;
        t.buffers.insert(
            index,
            buf(*len, 0, 0x10_0000_0000 + u64::from(index) * 0x1_0000_0000),
        );
    }
    let mut state = BindState::default();
    state.stages.insert(Stage::Vertex, t);
    state
}

fn form(slots: &std::collections::BTreeMap<SlotRef, SlotInterface>, s: SlotRef) -> BufferForm {
    match slots.get(&s) {
        Some(SlotInterface::Buffer(b)) => b.form,
        other => panic!("{s:?}: {other:?}"),
    }
}

/// The verifier's reproducer: four storage buffers fit, so the base pipeline runs no fallback,
/// yet states its candidates (`Spare`); an unaligned vertex buffer asks for the pulled variant,
/// whose fifth storage buffer makes its fallback view an array, and a large array at buffer 4.
/// The plan names buffer 4 kept already, so the variant views another array and its plan states
/// the same key.
#[test]
fn a_newly_pulled_layout_keeps_the_large_candidates_storage_views() {
    let Some(bc) = assemble("vertex", VERTEX) else {
        return;
    };
    let dev = profile_f();
    let (d, base) = vertex_descriptor(&dev, &bc);
    for index in 1..=4 {
        let s = slot(Stage::Vertex, index);
        assert_eq!(
            form(&base.interface.slots, s),
            BufferForm::Descriptor(DescriptorView::Storage)
        );
        assert!(base.dependencies.iter().any(|dep| dep.fact
            == RuntimeFact::UniformViewRange {
                slot: s,
                choice: UniformViewChoice::Spare,
            }));
    }
    let state = vertex_state(50, [SMALL, SMALL, SMALL, LARGE]);
    let plan = base.plan(&dev, &state.shape()).expect("base plan");
    assert_eq!(plan.variant.pulled_vertex_buffers, BTreeSet::from([0]));
    assert_eq!(
        plan.variant.storage_view_slots,
        BTreeSet::from([slot(Stage::Vertex, 4)])
    );
    let var = dev
        .compile_render_variant(&d, &plan.variant, &Budget::standard())
        .expect("variant compile");
    assert_eq!(
        form(&var.interface.slots, slot(Stage::Vertex, 4)),
        BufferForm::Descriptor(DescriptorView::Storage)
    );
    assert_eq!(
        form(&var.interface.slots, slot(Stage::Vertex, 3)),
        BufferForm::Descriptor(DescriptorView::UniformWords)
    );
    let vplan = var.plan(&dev, &state.shape()).expect("variant plan");
    assert_eq!(vplan.variant, plan.variant, "a fixed point");
}

/// For every shape of the vertex pipeline (the vertex buffer aligned or not, each array small or
/// large), the plan of the pipeline compiled under the base plan's key states the same key, or
/// that compile refuses the per-stage limit (the pulled source with every array large).
#[test]
fn every_render_variant_plan_states_its_own_key() {
    let Some(bc) = assemble("vertex_all", VERTEX) else {
        return;
    };
    let dev = profile_f();
    let (d, base) = vertex_descriptor(&dev, &bc);
    let (mut variants, mut refused) = (0, 0);
    for offset in [0, 50] {
        for bits in 0..16u32 {
            let lengths = [0, 1, 2, 3].map(|k| if bits >> k & 1 == 1 { LARGE } else { SMALL });
            let state = vertex_state(offset, lengths);
            let plan = base.plan(&dev, &state.shape()).expect("base plan");
            if plan.variant.is_empty() {
                continue;
            }
            variants += 1;
            match dev.compile_render_variant(&d, &plan.variant, &Budget::standard()) {
                Ok(var) => {
                    let vplan = var.plan(&dev, &state.shape()).expect("variant plan");
                    assert_eq!(vplan.variant, plan.variant, "{offset} {lengths:?}");
                }
                Err(Error::Unsupported(u)) if u.feature == Feature::DescriptorCount => {
                    assert_eq!(lengths, [LARGE; 4], "{offset}");
                    refused += 1;
                }
                Err(e) => panic!("{offset} {lengths:?}: {e}"),
            }
        }
    }
    assert_eq!((variants, refused), (16, 1), "every unaligned shape pulls");
}

/// For every shape of the owned kernel (each read-only array small or large), the plan of the
/// kernel compiled under the base plan's key states the same key, or that compile refuses the
/// per-stage limit (every candidate large).
#[test]
fn every_kernel_variant_plan_states_its_own_key() {
    const ENTRY: &str = "descriptor_limit_uniform_views";
    let Some(bc) = assembled(
        llvm_assemble(&engine_dir().join(format!("{ENTRY}.ll"))),
        ENTRY,
    ) else {
        return;
    };
    let dev = profile_f();
    let compile = |inputs: &LoweringInputs| {
        let lib = Library::from_air(&bc).expect("library");
        let f = lib.function(ENTRY).expect("function");
        let compiled =
            dev.compile_function(&FunctionDescriptor::new(f), inputs, &Budget::standard())?;
        dev.compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
    };
    let base = compile(&LoweringInputs::new()).expect("base");
    let (mut variants, mut refused) = (0, 0);
    for bits in 0..16u32 {
        let mut t = StageTable::default();
        for (k, index) in [0u32, 1, 2, 4].into_iter().enumerate() {
            let len = if bits >> k & 1 == 1 { LARGE } else { SMALL };
            t.buffers.insert(
                index,
                buf(len, 0, 0x10_0000_0000 + u64::from(index) * 0x1_0000_0000),
            );
        }
        t.buffers.insert(3, buf(256, 0, 0x13_0000_0000));
        t.buffers.insert(5, buf(SMALL, 0, 0x15_0000_0000));
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, t);
        let plan = base.plan(&dev, &state.shape()).expect("base plan");
        if plan.variant.is_empty() {
            continue;
        }
        variants += 1;
        match compile(&plan.variant.lowering_inputs(Stage::Kernel)) {
            Ok(var) => {
                let vplan = var.plan(&dev, &state.shape()).expect("variant plan");
                assert_eq!(vplan.variant, plan.variant, "{bits:04b}");
            }
            Err(Error::Unsupported(u)) if u.feature == Feature::DescriptorCount => refused += 1,
            Err(e) => panic!("{bits:04b}: {e}"),
        }
    }
    // A large `constant` array asks (8 shapes); of those, the ones leaving fewer than two small
    // `device` arrays to view refuse.
    assert!(
        variants > 0 && refused > 0,
        "{variants} variants, {refused} refused"
    );
}
