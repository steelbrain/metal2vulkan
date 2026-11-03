//! The graphics and texture binder end to end. No device is opened; nothing runs on a GPU.
//!
//! Pipelines go through the public compile API, `RenderPipeline::plan` /
//! `ComputePipeline::plan`, and `BindingPlan::instantiate_draw` / `instantiate_dispatch`:
//!
//! - A vertex + fragment pipeline (`engine/stage_io`) plans one write per layout binding, one
//!   vertex-buffer bind per fixed-function binding, and draws map to the Vulkan draw of the same
//!   arguments; an unbound vertex buffer and a missing per-bind stride are typed refusals. A
//!   pulled layout's stride fills its push field (static from the pipeline, per bind from the
//!   bind), and a pulled source buffer that is also a buffer argument is one binding. A draw's
//!   vertex-buffer extent is what its vertex range fetches, its index and indirect buffers are
//!   uses, and a bind offset or per-bind stride Vulkan cannot fetch is refused.
//! - Sampling kernels (`engine/sampling`) write combined image samplers: the bound texture's
//!   view and the runtime sampler slot, the neutral of the declared type when unbound, and a
//!   neutral with an `IncompatibleBinding` diagnostic for a texture of another type, the variant
//!   for one of another numeric class; `SamplerState` follows the bound sampler's word; anisotropy
//!   and LOD clamps are checked at plan time.
//! - Texture kernels (`engine/texture_ops`): a read cube is viewed as the 2D array it is
//!   declared as (`ImageViewShape`); texel buffers get their view format and a range of whole
//!   texels, checked exactly against `maxTexelBufferElements`.
//! - A storage texture (`engine/storage_texture_write`) bound as bgra8 asks for the storage
//!   variant exactly where bgra8 lacks storage, and the rgba8 plan stays on the common path.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::collections::BTreeSet;
use std::path::Path;

use crate::bind::{
    sampler_desc, AccessClass, BindState, BindingPlan, BoundBuffer, BoundSampler, BoundTexture,
    Dispatch, DrawCall, DrawCommands, FieldLocation, ImageSource, IndexBufferRef, IndexOverrun,
    IndirectRef, NeutralResource, PlanDiagnostic, SamplerDescriptor, SamplerSource, StageTable,
    TextureFacts, UseTarget, WriteSource,
};
use crate::budget::Budget;
use crate::device::profile::{Containment, Robustness};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, DescriptorError, Error, Feature};
use crate::library::Library;
use crate::metal::sampler::{SamplerUse, SamplerWord};
use crate::metal::{
    self, PixelFormat, SamplerMinMagFilter, SamplerMipFilter, SamplerReductionMode, TextureType,
    VertexFormat,
};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use crate::pipeline::interface::{AccessSet, BufferForm, DescriptorView, SlotInterface, Window};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{PushField, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::VertexStride;
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, RenderPipeline,
    RenderPipelineDescriptor, StageSource,
};
use crate::vk;

fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// Every committed profile as a device, by stem.
fn devices() -> Vec<(String, Device)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).expect("profiles") {
        let path = e.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        out.push((
            stem,
            Device::new(desc, DevicePolicy::default()).expect("device"),
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn device(stem: &str) -> Device {
    devices()
        .into_iter()
        .find(|(s, _)| s == stem)
        .map(|(_, d)| d)
        .expect("committed profile")
}

fn slot(stage: Stage, class: SlotClass, index: u32) -> SlotRef {
    SlotRef {
        stage,
        class,
        index,
    }
}

fn buffer(length: u64, offset: u64, stride: Option<u32>) -> BoundBuffer {
    BoundBuffer::Buffer {
        length,
        offset,
        gpu_address: 0x1_0000,
        device_address: None,
        attribute_stride: stride,
    }
}

// ---- Render pipelines --------------------------------------------------------------------------

/// `stage_io_vertex`'s attributes: one interleaved per-vertex buffer at index 0.
fn vertex_descriptor() -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    d.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(24)));
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        d.attributes
            .insert(index, VertexAttributeDescriptor::new(format, offset, 0));
    }
    d
}

/// `stage_io`'s vertex + fragment pipeline descriptor.
fn render_descriptor(bc: &[u8]) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let v = lib.function("stage_io_vertex").expect("vertex");
    let f = lib.function("stage_io_fragment").expect("fragment");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(f)));
    d.vertex_descriptor = Some(vertex_descriptor());
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    d.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    d.depth_format = Some(PixelFormat::Depth32Float);
    d
}

fn render_pipeline(dev: &Device, bc: &[u8]) -> Result<RenderPipeline, Error> {
    dev.compile_render_pipeline(&render_descriptor(bc), &Budget::standard())
}

/// A draw's own uses: each resource and its range, after checking that each is read in its
/// fixed-function stage.
fn uses(cmds: &crate::bind::DrawCommands) -> Vec<(UseTarget, Option<(u64, u64)>)> {
    cmds.uses
        .iter()
        .map(|u| {
            let (class, stages) = match u.resource {
                UseTarget::IndexBuffer => {
                    (AccessClass::Index, vk::PipelineStageFlags2::INDEX_INPUT)
                }
                _ => (
                    AccessClass::Indirect,
                    vk::PipelineStageFlags2::DRAW_INDIRECT,
                ),
            };
            assert_eq!(
                (u.access, u.class, u.stages),
                (AccessSet::READ, class, stages)
            );
            (u.resource, u.range.map(|r| (r.offset, r.len)))
        })
        .collect()
}

/// The vertex buffer at 0 and the constant `scale` at 1.
fn render_state(scale_offset: u64) -> BindState {
    let mut v = StageTable::default();
    v.buffers.insert(0, buffer(24 * 64, 48, None));
    v.buffers.insert(1, buffer(256, scale_offset, None));
    let mut s = BindState::default();
    s.stages.insert(Stage::Vertex, v);
    s
}

/// `Device::compile_render_pipeline` compiles a `Function` fragment stage with no output at a color
/// it writes that the pipeline has no attachment for, whose write Metal discards: the fixture's
/// fragment, writing colors 0 and 2, states color 0 alone among its color outputs in a pipeline of
/// attachment 0 alone, and both where it has both. Under alpha to coverage its color 0 stays
/// without attachment 0, the coverage reading its alpha.
#[test]
fn a_render_pipeline_writes_no_color_it_has_no_attachment_for() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3", "amd-780m-radv"] {
        let dev = device(stem);
        let outputs = |edit: &dyn Fn(&mut RenderPipelineDescriptor)| {
            let mut d = render_descriptor(&bc);
            edit(&mut d);
            dev.compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .graphics
                .color_outputs
        };
        assert_eq!(outputs(&|_| {}), 0b101, "{stem}");
        assert_eq!(outputs(&|d| d.color_attachments[2] = None), 0b1, "{stem}");
        let covered = |d: &mut RenderPipelineDescriptor| {
            d.color_attachments[0] = None;
            d.alpha_to_coverage = true;
        };
        assert_eq!(outputs(&covered), 0b101, "{stem}");
    }
}

/// A vertex + fragment pipeline plans every binding, binds its vertex buffer, and maps each Metal
/// draw to the Vulkan draw of the same arguments.
#[test]
fn a_render_pipeline_plans_and_draws() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let mut planned = 0;
    let mut narrowed = 0;
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna3-radv"] {
        let dev = device(stem);
        let p = render_pipeline(&dev, &bc).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let state = render_state(16);
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        let bindings: usize = p.layout.sets.iter().map(|s| s.bindings.len()).sum();
        assert_eq!(plan.descriptors.len(), bindings, "{stem}");
        assert!(plan.variant.is_empty(), "{stem}");
        for d in &plan.descriptors {
            let b = &p.layout.sets[0].bindings[d.binding as usize];
            if let BindingSource::Buffer(s) = b.source {
                assert_eq!(d.source, WriteSource::Buffer { slot: s }, "{stem}");
            }
        }
        assert_eq!(plan.vertex_buffers.len(), p.vertex_input.bindings.len());
        assert_eq!(plan.vertex_buffers[0].buffer_index, 0);

        let draw = DrawCall::Primitives {
            vertex_start: 3,
            vertex_count: 6,
            instance_count: 2,
            base_instance: 5,
        };
        let cmds = plan.instantiate_draw(&state, &draw).expect("draw");
        assert_eq!(
            cmds.draw,
            vk::DrawCommand::Draw {
                vertex_count: 6,
                instance_count: 2,
                first_vertex: 3,
                first_instance: 5
            }
        );
        assert_eq!(cmds.buffer_offsets, vec![(0, 48)], "{stem}");
        assert!(!cmds.indirect_needs_arguments);
        assert!(cmds.uses.is_empty(), "{stem}");
        // The vertex buffer's extent is the bytes vertices 3..=8 fetch, unless a pulled attribute
        // reads the same buffer through its whole storage view.
        let vb = slot(Stage::Vertex, SlotClass::Buffer, 0);
        let extent = |cmds: &crate::bind::DrawCommands| {
            cmds.extents
                .iter()
                .find(|e| e.slot == vb)
                .map(|e| e.window)
                .expect("vertex extent")
        };
        let reads = p.vertex_input.attributes.iter().map(|a| {
            let size = a.format.block().expect("block").size;
            (i64::from(a.offset), i64::from(a.offset + size))
        });
        let lo = reads.clone().map(|r| r.0).min().expect("attributes");
        let hi = reads.map(|r| r.1).max().expect("attributes");
        if p.vertex_input.pulled.iter().any(|a| a.buffer_index == 0) {
            assert_eq!(extent(&cmds), Window::Whole, "{stem}");
        } else {
            narrowed += 1;
            assert_eq!(
                extent(&cmds),
                Window::Bounded {
                    start: 3 * 24 + lo,
                    len: (5 * 24 + hi - lo) as u64
                },
                "{stem}"
            );
        }
        // A bind offset the attributes' alignment does not divide (a `float3` attribute needs 4
        // bytes) is the plan's pulled variant, and a draw of this plan with it is internal.
        let unaligned = with_vertex_buffer(&state, buffer(24 * 64, 50, None));
        assert_eq!(
            p.plan(&dev, &unaligned.shape())
                .expect("plan")
                .variant
                .pulled_vertex_buffers,
            BTreeSet::from([0]),
            "{stem}"
        );
        assert!(
            matches!(
                plan.instantiate_draw(&unaligned, &draw),
                Err(BindError::Internal(_))
            ),
            "{stem}"
        );
        // Every push range of the layout is written once, over its whole range.
        for r in &p.layout.push_constants {
            let w = cmds
                .push_constants
                .iter()
                .find(|w| w.stages == r.stages)
                .unwrap_or_else(|| panic!("{stem}: no push write for {r:?}"));
            assert_eq!((w.offset, w.bytes.len() as u32), (r.offset, r.size));
        }
        // The constant buffer's range covers its window from the aligned bind offset.
        for range in &cmds.descriptor_ranges {
            assert!(range.range.range >= 16, "{stem}");
        }
        let indexed = DrawCall::Indexed {
            index_type: vk::IndexType::UINT16,
            index_buffer: IndexBufferRef::new(9, 4096),
            index_count: 12,
            index_offset: 64,
            instance_count: 1,
            base_vertex: -2,
            base_instance: 0,
        };
        let cmds = plan.instantiate_draw(&state, &indexed).expect("indexed");
        assert_eq!(extent(&cmds), Window::Whole, "{stem}: GPU-side vertices");
        assert_eq!(
            uses(&cmds),
            vec![(UseTarget::IndexBuffer, Some((0, 24)))],
            "{stem}"
        );
        assert_eq!(
            cmds.draw,
            vk::DrawCommand::DrawIndexed {
                index_type: vk::IndexType::UINT16,
                index_buffer_offset: 64,
                index_count: 12,
                instance_count: 1,
                first_index: 0,
                vertex_offset: -2,
                first_instance: 0
            }
        );
        let indirect = DrawCall::Indirect {
            indirect: IndirectRef::new(4, 32),
            indexed: None,
        };
        let cmds = plan.instantiate_draw(&state, &indirect).expect("indirect");
        assert_eq!(
            cmds.draw,
            vk::DrawCommand::DrawIndirect {
                offset: 32,
                draw_count: 1,
                stride: 16
            }
        );
        assert_eq!(extent(&cmds), Window::Whole, "{stem}");
        assert_eq!(
            uses(&cmds),
            vec![(UseTarget::Indirect, Some((0, 16)))],
            "{stem}"
        );
        planned += 1;

        // An unbound fixed-function vertex buffer is refused; an unbound buffer argument gets the
        // zero buffer.
        let mut unbound = state.clone();
        unbound
            .stages
            .get_mut(&Stage::Vertex)
            .expect("vertex")
            .buffers
            .remove(&0);
        assert_eq!(
            p.plan(&dev, &unbound.shape()),
            Err(BindError::UnboundVertexBuffer {
                slot: slot(Stage::Vertex, SlotClass::Buffer, 0)
            })
        );
        let mut neutral = state.clone();
        neutral
            .stages
            .get_mut(&Stage::Vertex)
            .expect("vertex")
            .buffers
            .remove(&1);
        let plan = p.plan(&dev, &neutral.shape()).expect("plan");
        assert!(plan.descriptors.iter().any(|d| matches!(
            d.source,
            WriteSource::Neutral(NeutralResource::ZeroBuffer { .. })
        )));
        // A shape the state does not have is an internal error at instantiation.
        assert!(matches!(
            plan.instantiate_draw(&state, &draw),
            Err(BindError::Internal(_))
        ));
    }
    assert_eq!(planned, 3);
    assert!(
        narrowed > 0,
        "no profile fetches the vertex buffer fixed function only"
    );
}

/// A device without `drawIndirectFirstInstance` (as the Vulkan 1.2 floors are; here lavapipe's
/// description without it) takes an indirect draw's arguments on the CPU, since Metal's may carry a
/// base instance an indirect command there must not; one with it records the indirect command. A
/// direct draw is unaffected.
#[test]
fn an_indirect_draw_without_draw_indirect_first_instance_reads_its_arguments() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let edited = text.replace(
        "\"drawIndirectFirstInstance\": true",
        "\"drawIndirectFirstInstance\": false",
    );
    assert_ne!(edited, text, "lavapipe states the feature");
    let without = DeviceDescription::from_khronos_json(&edited, None).expect("description");
    let without = Device::new(without, DevicePolicy::default()).expect("device");
    for (dev, missing) in [(without, true), (device("lavapipe"), false)] {
        let stem = if missing { "without" } else { "lavapipe" };
        let p = render_pipeline(&dev, &bc).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let state = render_state(16);
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        let indirect = DrawCall::Indirect {
            indirect: IndirectRef::new(4, 32),
            indexed: None,
        };
        let cmds = plan.instantiate_draw(&state, &indirect).expect("indirect");
        assert_eq!(cmds.indirect_needs_arguments, missing, "{stem}");
        let direct = DrawCall::Primitives {
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 2,
        };
        let cmds = plan.instantiate_draw(&state, &direct).expect("direct");
        assert!(!cmds.indirect_needs_arguments, "{stem}");
    }
}

/// Every committed profile as a device, by stem, with whether the device enables
/// `robustBufferAccess2`, read from what it enables rather than from its name.
fn devices_by_robustness() -> Vec<(String, Device, bool)> {
    use crate::device::requirements::VkFeature;
    devices()
        .into_iter()
        .map(|(stem, dev)| {
            let robust = dev
                .profile()
                .fields()
                .enabled
                .features
                .contains(&VkFeature::RobustBufferAccess2);
            (stem, dev, robust)
        })
        .collect()
}

/// A direct indexed draw whose indices run past its index buffer's length is drawn unchanged where
/// the device enables `robustBufferAccess2`, which bounds the index fetch, and is clamped to the
/// indices inside the buffer where it does not, since Vulkan then requires every index inside the
/// bound range (`VUID-vkCmdDrawIndexed-robustBufferAccess2-08798`); `index_overrun` records which.
/// A draw to the buffer's last byte is unchanged, and one with no index inside the buffer draws
/// none on either. The feature decides, not the profile: lavapipe without it clamps. Checked on
/// every committed profile that compiles the pipeline.
#[test]
fn an_indexed_draw_past_its_index_buffer_is_bounded() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let edited = text.replace(
        "\"robustBufferAccess2\": true",
        "\"robustBufferAccess2\": false",
    );
    assert_ne!(edited, text, "lavapipe states the feature");
    let without = DeviceDescription::from_khronos_json(&edited, None).expect("description");
    let without = Device::new(without, DevicePolicy::default()).expect("device");
    let mut devices = devices_by_robustness();
    devices.push((
        "lavapipe without robustBufferAccess2".into(),
        without,
        false,
    ));
    let draw = |length, index_offset, index_count| DrawCall::Indexed {
        index_type: vk::IndexType::UINT16,
        index_buffer: IndexBufferRef::new(9, length),
        index_count,
        index_offset,
        instance_count: 3,
        base_vertex: 7,
        base_instance: 0,
    };
    let drawn = |cmds: &DrawCommands| match cmds.draw {
        vk::DrawCommand::DrawIndexed {
            index_buffer_offset,
            index_count,
            vertex_offset,
            ..
        } => (index_buffer_offset, index_count, vertex_offset),
        other => panic!("an indexed draw: {other:?}"),
    };
    let mut seen = [false; 2];
    for (stem, dev, robust) in devices {
        // `vk12-floor` and `roadmap-2022` refuse the pipeline's depth attachment.
        let Ok(p) = render_pipeline(&dev, &bc) else {
            continue;
        };
        seen[usize::from(robust)] = true;
        let state = render_state(16);
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        // 12 halves from byte 64 of 88 bytes: to the buffer's last byte.
        let cmds = plan
            .instantiate_draw(&state, &draw(88, 64, 12))
            .expect("inside");
        assert_eq!(cmds.index_overrun, None, "{stem}");
        assert_eq!(drawn(&cmds), (64, 12, 7), "{stem}");
        assert_eq!(uses(&cmds), vec![(UseTarget::IndexBuffer, Some((0, 24)))]);
        // 86 bytes: Metal's fetch reads to byte 88, the length rounded up to 4 (`fetch_end`).
        let cmds = plan
            .instantiate_draw(&state, &draw(86, 64, 12))
            .expect("rounded");
        assert_eq!(cmds.index_overrun, None, "{stem}");
        assert_eq!(uses(&cmds), vec![(UseTarget::IndexBuffer, Some((0, 24)))]);
        // 3 more: past the end.
        let cmds = plan
            .instantiate_draw(&state, &draw(88, 64, 15))
            .expect("past");
        if robust {
            assert_eq!(
                cmds.index_overrun,
                Some(IndexOverrun::Robust { past_end: 3 }),
                "{stem}"
            );
            assert_eq!(drawn(&cmds), (64, 15, 7), "{stem}");
        } else {
            assert_eq!(
                cmds.index_overrun,
                Some(IndexOverrun::Clamped { past_end: 3 }),
                "{stem}"
            );
            assert_eq!(drawn(&cmds), (64, 12, 7), "{stem}");
        }
        assert_eq!(uses(&cmds), vec![(UseTarget::IndexBuffer, Some((0, 24)))]);
        // An index offset at the buffer's end: no index inside it.
        let cmds = plan
            .instantiate_draw(&state, &draw(88, 88, 4))
            .expect("none");
        assert_eq!(
            cmds.index_overrun,
            Some(IndexOverrun::Clamped { past_end: 4 }),
            "{stem}"
        );
        assert_eq!(drawn(&cmds), (0, 0, 7), "{stem}");
        assert_eq!(uses(&cmds), vec![(UseTarget::IndexBuffer, Some((0, 0)))]);
        // A draw of no index: at its offset inside the buffer, at offset 0 from its end or past.
        for (offset, at) in [(64, 64), (88, 0), (200, 0)] {
            let cmds = plan
                .instantiate_draw(&state, &draw(88, offset, 0))
                .expect("no index");
            assert_eq!(cmds.index_overrun, None, "{stem}");
            assert_eq!(drawn(&cmds), (at, 0, 7), "{stem} {offset}");
            assert_eq!(uses(&cmds), vec![(UseTarget::IndexBuffer, Some((0, 0)))]);
        }
    }
    assert_eq!(seen, [true, true], "both forms are checked");
}

/// An indexed indirect draw's counts are in GPU memory, so where the device lacks
/// `robustBufferAccess2` (`VUID-VkDrawIndexedIndirectCommand-robustBufferAccess2-08798`), or its
/// index offset is at or past the buffer's length, it asks for its arguments on the CPU, whose
/// direct draw the binder bounds. Where the device has the feature it stays GPU-side unless
/// something else asks for the read, as for the same draw without indices. Checked on every
/// committed profile that compiles the pipeline, and on lavapipe without the feature.
#[test]
fn an_indexed_indirect_draw_reads_its_arguments_where_nothing_bounds_its_indices() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let edited = text.replace(
        "\"robustBufferAccess2\": true",
        "\"robustBufferAccess2\": false",
    );
    assert_ne!(edited, text, "lavapipe states the feature");
    let without = DeviceDescription::from_khronos_json(&edited, None).expect("description");
    let without = Device::new(without, DevicePolicy::default()).expect("device");
    let mut devices = devices_by_robustness();
    devices.push((
        "lavapipe without robustBufferAccess2".into(),
        without,
        false,
    ));
    let indirect = |indexed| DrawCall::Indirect {
        indirect: IndirectRef::new(4, 32),
        indexed,
    };
    let at = |offset| Some((vk::IndexType::UINT16, IndexBufferRef::new(9, 256), offset));
    // Whether a robust profile kept an indexed indirect draw on the GPU, and whether a profile
    // without the feature read one on the CPU that it would otherwise have kept there.
    let mut seen = [false; 2];
    for (stem, dev, robust) in devices {
        let Ok(p) = render_pipeline(&dev, &bc) else {
            continue;
        };
        let state = render_state(16);
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        let read = |draw| {
            plan.instantiate_draw(&state, &draw)
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .indirect_needs_arguments
        };
        let otherwise = read(indirect(None));
        assert_eq!(read(indirect(at(64))), otherwise || !robust, "{stem}");
        assert!(read(indirect(at(256))), "{stem}: an offset at the end");
        if !otherwise {
            seen[usize::from(robust)] = true;
        }
    }
    assert_eq!(seen, [true, true], "both forms are checked");
}

/// A vertex buffer bound at an offset its attributes' alignment does not divide is pulled in the
/// plan's variant. `compile_render_variant` compiles it (another pipeline key, the layout's
/// attributes pulled and recorded in `variant_pulled`), and its plan for the same shape states the
/// same key; it keeps stating it for an aligned shape, since the pipeline pulls the layout for any
/// bind. The variant's draw binds no vertex buffer: the layout's buffer is one storage binding the
/// fetch reads from the bind offset at the static stride.
#[test]
fn an_unbindable_vertex_buffer_is_pulled_in_the_variant() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let draw = DrawCall::Primitives {
        vertex_start: 3,
        vertex_count: 6,
        instance_count: 2,
        base_instance: 5,
    };
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna3-radv"] {
        let dev = device(stem);
        let d = render_descriptor(&bc);
        let p = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let aligned = render_state(16);
        assert!(p.vertex_input.variant_pulled.is_empty(), "{stem}");
        assert!(
            p.plan(&dev, &aligned.shape())
                .expect("plan")
                .variant
                .is_empty(),
            "{stem}"
        );
        let unaligned = with_vertex_buffer(&aligned, buffer(24 * 64, 50, None));
        let plan = p.plan(&dev, &unaligned.shape()).expect("plan");
        assert_eq!(
            plan.variant.pulled_vertex_buffers,
            BTreeSet::from([0]),
            "{stem}"
        );
        let v = dev
            .compile_render_variant(&d, &plan.variant, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem} variant: {e}"));
        let keys = dev
            .render_variant_key(
                &d,
                &plan.variant,
                &Budget::standard(),
                &crate::pipeline::StageEntries::new(),
            )
            .expect("variant keys");
        assert_eq!(keys.pipeline, v.key, "{stem}");
        assert_ne!(v.key, p.key, "{stem}");
        assert_eq!(v.vertex_input.variant_pulled, BTreeSet::from([0]), "{stem}");
        assert!(
            v.vertex_input.bindings.iter().all(|b| b.buffer_index != 0),
            "{stem}: layout 0 is not fixed function"
        );
        let pulled: BTreeSet<u32> = v.vertex_input.pulled.iter().map(|a| a.attribute).collect();
        assert_eq!(pulled, BTreeSet::from([0, 1, 4, 7]), "{stem}");
        let vplan = v.plan(&dev, &unaligned.shape()).expect("variant plan");
        assert_eq!(vplan.variant, plan.variant, "{stem}: a fixed point");
        assert_eq!(
            v.plan(&dev, &aligned.shape()).expect("plan").variant,
            plan.variant,
            "{stem}: the variant keeps pulling"
        );
        assert!(vplan.vertex_buffers.is_empty(), "{stem}");
        let vb = slot(Stage::Vertex, SlotClass::Buffer, 0);
        let binding = v
            .layout
            .sets
            .iter()
            .flat_map(|s| &s.bindings)
            .find(|b| b.source == BindingSource::Buffer(vb))
            .unwrap_or_else(|| panic!("{stem}: no binding of the source buffer"));
        assert_eq!(binding.ty, vk::DescriptorType::STORAGE_BUFFER, "{stem}");
        let cmds = vplan.instantiate_draw(&unaligned, &draw).expect("draw");
        assert!(cmds.buffer_offsets.is_empty(), "{stem}");
        assert_eq!(
            vertex_push(&vplan, &cmds, PushField::VertexBufferStride(0)),
            24,
            "{stem}"
        );
        let range = cmds
            .descriptor_ranges
            .iter()
            .find(|r| (r.set, r.binding) == (0, binding.binding))
            .expect("source buffer range");
        assert!(range.range.offset <= 50, "{stem}: {range:?}");
        // The default pipeline's plan does not draw the unaligned state.
        assert!(
            matches!(
                plan.instantiate_draw(&unaligned, &draw),
                Err(BindError::Internal(_))
            ),
            "{stem}"
        );
    }
}

/// A pulled source buffer bound to an allocation too large for `maxStorageBufferRange` is a
/// device-address slot in the variant. Planning the default pipeline for an unaligned, large vertex
/// buffer names both the pulled layout and its device-address slot at once, so the variant's own
/// plan states the same key. That plan writes no descriptor for the source buffer: the vertex stage
/// reads its root table, whose entry `2 + 0` is the bound buffer (its host and guest addresses,
/// padded size, and bind offset), and a bind without a device address is refused. On a profile
/// without device addresses the variant only pulls, and the draw's descriptor range exceeds the
/// limit.
#[test]
fn a_render_plan_reads_pulled_source_buffers_by_device_address() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let draw = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let large = 1u64 << 40;
    let (mut addressed, mut viewed) = (0, 0);
    for stem in ["moltenvk-m3", "lavapipe", "lunarg-desktop-baseline-2023"] {
        let dev = device(stem);
        let d = render_descriptor(&bc);
        let p = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let host = 0x7_0000_0000;
        let state = with_vertex_buffer(
            &render_state(16),
            BoundBuffer::Buffer {
                length: large,
                offset: 50,
                gpu_address: 0x1_0000,
                device_address: Some(host),
                attribute_stride: None,
            },
        );
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        let vb = slot(Stage::Vertex, SlotClass::Buffer, 0);
        assert_eq!(plan.variant.pulled_vertex_buffers, BTreeSet::from([0]));
        let psb = dev.profile().fields().descriptors.buffer_device_address
            && dev.profile().fields().leaf.int64;
        assert_eq!(
            plan.variant.psb_slots.contains(&vb),
            psb,
            "{stem}: {:?}",
            plan.variant
        );
        let v = dev
            .compile_render_variant(&d, &plan.variant, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem} variant: {e}"));
        let vplan = v.plan(&dev, &state.shape()).expect("variant plan");
        assert_eq!(vplan.variant, plan.variant, "{stem}: a fixed point");
        let source = v
            .layout
            .sets
            .iter()
            .flat_map(|s| &s.bindings)
            .any(|b| b.source == BindingSource::Buffer(vb));
        if !psb {
            assert!(source, "{stem}: a storage view");
            assert!(
                matches!(
                    vplan.instantiate_draw(&state, &draw),
                    Err(BindError::DescriptorRangeExceeded { slot, .. }) if slot == vb
                ),
                "{stem}"
            );
            viewed += 1;
            continue;
        }
        assert!(!source, "{stem}: no descriptor for a device-address slot");
        assert!(
            matches!(
                v.interface.slots.get(&vb),
                Some(SlotInterface::Buffer(b)) if b.form == BufferForm::Psb
            ),
            "{stem}"
        );
        let cmds = vplan.instantiate_draw(&state, &draw).expect("draw");
        assert!(cmds.buffer_offsets.is_empty(), "{stem}");
        let (_, table) = cmds
            .root_tables
            .iter()
            .find(|(stages, _)| stages.contains(vk::ShaderStageFlags::VERTEX))
            .unwrap_or_else(|| panic!("{stem}: no vertex root table"));
        let word = |entry: usize, field: usize| {
            let at = entry * 32 + field * 8;
            u64::from_le_bytes(table.bytes[at..at + 8].try_into().expect("eight bytes"))
        };
        assert_eq!(
            [word(2, 0), word(2, 1), word(2, 2), word(2, 3)],
            [host, 0x1_0000, large, 50],
            "{stem}: entry 2 + 0 is the source buffer"
        );
        let mut no_address = state.clone();
        no_address
            .stages
            .get_mut(&Stage::Vertex)
            .expect("vertex")
            .buffers
            .insert(
                0,
                BoundBuffer::Buffer {
                    length: large,
                    offset: 50,
                    gpu_address: 0x1_0000,
                    device_address: None,
                    attribute_stride: None,
                },
            );
        assert_eq!(
            v.plan(&dev, &no_address.shape()),
            Err(BindError::MissingDeviceAddress { slot: vb }),
            "{stem}"
        );
        addressed += 1;
    }
    assert_eq!((addressed, viewed), (2, 1));
}

/// A per-bind stride (`MTLBufferLayoutStrideDynamic`) becomes the vertex buffer's dynamic stride;
/// a buffer bound without one reads at stride 0, as plain Metal's encoder does.
#[test]
fn a_per_bind_stride_comes_from_the_bind() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let lib = Library::from_air(&bc).expect("library");
    let v = lib.function("stage_io_vertex").expect("vertex");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
    let mut vd = vertex_descriptor();
    vd.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(None));
    d.vertex_descriptor = Some(vd);
    d.rasterization_enabled = false;
    let p = match dev.compile_render_pipeline(&d, &Budget::standard()) {
        Ok(p) => p,
        Err(e) => panic!("per-bind stride pipeline: {e}"),
    };
    let state = render_state(0);
    // Stride 0: the draw's dynamic stride, or the pulled variant where fixed function cannot
    // fetch an element's 24 bytes at stride 0.
    let unstrided = p.plan(&dev, &state.shape()).expect("plan without a stride");
    if unstrided.variant.pulled_vertex_buffers.is_empty() {
        let cmds = unstrided
            .instantiate_draw(
                &state,
                &DrawCall::Primitives {
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0,
                },
            )
            .expect("draw without a stride");
        assert_eq!(cmds.vertex_strides, vec![(0, 0)]);
    } else {
        assert_eq!(unstrided.variant.pulled_vertex_buffers, BTreeSet::from([0]));
    }
    let mut strided = state.clone();
    strided
        .stages
        .get_mut(&Stage::Vertex)
        .expect("vertex")
        .buffers
        .insert(0, buffer(24 * 64, 0, Some(32)));
    let plan = p.plan(&dev, &strided.shape()).expect("plan");
    assert!(plan.vertex_buffers[0].per_bind_stride);
    let cmds = plan
        .instantiate_draw(
            &strided,
            &DrawCall::Primitives {
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        )
        .expect("draw");
    assert_eq!(cmds.vertex_strides, vec![(0, 32)]);
    // A per-bind stride must be a multiple of every attribute's alignment (`float3`: 4), as
    // dynamic state cover an element's 24 bytes, and stay within `maxVertexInputBindingStride`.
    let draw = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let with_stride = |stride| {
        let mut s = strided.clone();
        s.stages
            .get_mut(&Stage::Vertex)
            .expect("vertex")
            .buffers
            .insert(0, buffer(24 * 64, 0, Some(stride)));
        plan.instantiate_draw(&s, &draw)
    };
    // Another stride Vulkan's rules refuse is the plan's pulled variant; this plan's draw of it is
    // internal.
    let pulled = |stride| {
        let s = with_vertex_buffer(&strided, buffer(24 * 64, 0, Some(stride)));
        p.plan(&dev, &s.shape())
            .expect("plan")
            .variant
            .pulled_vertex_buffers
            == BTreeSet::from([0])
    };
    assert!(with_stride(24).is_ok());
    assert!(!pulled(24));
    assert!(matches!(with_stride(26), Err(BindError::Internal(_))));
    assert!(pulled(26));
    if p.graphics
        .dynamic_states
        .contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE)
    {
        assert!(pulled(20));
    }
    let max = dev
        .profile()
        .fields()
        .limits
        .max_vertex_input_binding_stride;
    assert!(pulled(max.next_multiple_of(4) + 4));
}

/// `state` with vertex-stage buffer 0 replaced by `b`.
/// A rebind of buffer slots writes over a draw's commands the commands a whole instantiation of
/// the new state writes: for each committed profile, each state of the vertex buffer and the
/// constant buffer at other offsets and over other allocations that keeps the plan's shape, the
/// commands of the first state rebound at the slots that changed are the new state's. Where the
/// rebind writes nothing (`Ok(false)`), the plan reads a slot some other way.
#[test]
fn a_rebind_of_buffer_slots_writes_what_an_instantiation_writes() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let (vertex, scale) = (
        slot(Stage::Vertex, SlotClass::Buffer, 0),
        slot(Stage::Vertex, SlotClass::Buffer, 1),
    );
    let draw = DrawCall::Primitives {
        vertex_start: 3,
        vertex_count: 6,
        instance_count: 2,
        base_instance: 5,
    };
    // The same allocation's length and offset at another address.
    let elsewhere = |b: BoundBuffer| match b {
        BoundBuffer::Buffer {
            length,
            offset,
            device_address,
            attribute_stride,
            ..
        } => BoundBuffer::Buffer {
            length,
            offset,
            gpu_address: 0x7_0000,
            device_address,
            attribute_stride,
        },
        other => other,
    };
    let (mut rebound, mut pushed, mut whole) = (0, 0, 0);
    for (stem, dev) in devices() {
        let Ok(p) = render_pipeline(&dev, &bc) else {
            continue;
        };
        // From a constant at an aligned offset, and at one a residual reaches.
        for first in [render_state(16), render_state(4)] {
            let Ok(plan) = p.plan(&dev, &first.shape()) else {
                continue;
            };
            let scales = [0, 4, 16, 20, 36, 48, 100, 240, 256];
            let vertices = [0, 24, 48, 72, 96, 2400];
            let cases = scales
                .iter()
                .flat_map(|&s| vertices.iter().map(move |&v| (s, v)));
            for ((s, v), moved) in cases.flat_map(|c| [(c, false), (c, true)]) {
                let mut state = render_state(s);
                let table = state.stages.get_mut(&Stage::Vertex).expect("vertex");
                table.buffers.insert(0, buffer(24 * 64 + v, v, None));
                if moved {
                    let b = table.buffers.get(&1).copied().expect("scale");
                    table.buffers.insert(1, elsewhere(b));
                }
                if state.shape() != first.shape() {
                    continue;
                }
                let before = plan.instantiate_draw(&first, &draw).expect("first draw");
                let expected = plan.instantiate_draw(&state, &draw);
                let mut cmds = before.clone();
                match plan.rebind_draw_slots_into(&state, &[vertex, scale], &mut cmds) {
                    Ok(true) => {
                        rebound += 1;
                        pushed += usize::from(cmds.push_constants != before.push_constants);
                        assert_eq!(Ok(&cmds), expected.as_ref(), "{stem} {s} {v} {moved}");
                    }
                    Ok(false) => whole += 1,
                    Err(e) => assert_eq!(Err(e), expected, "{stem} {s} {v} {moved}"),
                }
            }
        }
    }
    assert!(
        rebound > 0,
        "no rebind wrote commands ({whole} instantiated whole)"
    );
    assert!(
        pushed > 0,
        "no rebind wrote a push field ({rebound} rebound)"
    );
}

fn with_vertex_buffer(state: &BindState, b: BoundBuffer) -> BindState {
    let mut s = state.clone();
    s.stages
        .get_mut(&Stage::Vertex)
        .expect("vertex")
        .buffers
        .insert(0, b);
    s
}

/// The `u32` a draw writes for vertex push field `field`.
fn vertex_push(plan: &BindingPlan, cmds: &DrawCommands, field: PushField) -> u32 {
    stage_push(
        plan,
        cmds,
        (Stage::Vertex, vk::ShaderStageFlags::VERTEX),
        field,
    )
}

/// The `u32` a draw writes for push field `field` of `stage` (whose Vulkan stage is `flags`).
fn stage_push(
    plan: &BindingPlan,
    cmds: &DrawCommands,
    (stage, flags): (Stage, vk::ShaderStageFlags),
    field: PushField,
) -> u32 {
    let f = plan
        .push
        .fields
        .iter()
        .find(|f| f.stage == stage && f.field == field)
        .unwrap_or_else(|| panic!("{field:?} is not placed"));
    let FieldLocation::PushWindow { offset } = f.location else {
        panic!("{field:?} spilled");
    };
    let w = cmds
        .push_constants
        .iter()
        .find(|w| {
            w.stages.contains(flags)
                && w.offset <= offset
                && offset + 4 <= w.offset + w.bytes.len() as u32
        })
        .unwrap_or_else(|| panic!("no push write covers {field:?}"));
    let at = (offset - w.offset) as usize;
    u32::from_le_bytes(w.bytes[at..at + 4].try_into().expect("four bytes"))
}

/// Pulled attributes (signed formats into the `uint` attributes) from layout 1, the `scale`
/// argument's index, with a static stride, and from layout 2 with a per-bind stride. Buffer 1 is
/// one binding (the argument's slot, widened to a `Whole` storage view); the static stride fills
/// its push field without a per-bind stride, and the per-bind one is the bind's `attributeStride`,
/// 0 without one (as for fixed function).
#[test]
fn pulled_layouts_take_their_stride_and_share_argument_slots() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let lib = Library::from_air(&bc).expect("library");
    let v = lib.function("stage_io_vertex").expect("vertex");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
    let mut vd = vertex_descriptor();
    vd.layouts
        .insert(1, VertexBufferLayoutDescriptor::new(Some(4)));
    vd.layouts
        .insert(2, VertexBufferLayoutDescriptor::new(None));
    vd.attributes.insert(
        4,
        VertexAttributeDescriptor::new(VertexFormat::Short2, 0, 2),
    );
    vd.attributes
        .insert(7, VertexAttributeDescriptor::new(VertexFormat::Char4, 0, 1));
    d.vertex_descriptor = Some(vd);
    d.rasterization_enabled = false;
    let p = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("pulled pipeline: {e}"));
    let strides: Vec<(u32, VertexStride)> = p
        .vertex_input
        .pulled
        .iter()
        .map(|a| (a.buffer_index, a.stride))
        .collect();
    assert_eq!(
        strides,
        [(2, VertexStride::PerBind), (1, VertexStride::Static(4))]
    );
    let scale = slot(Stage::Vertex, SlotClass::Buffer, 1);
    assert!(
        matches!(
            p.interface.slots.get(&scale),
            Some(SlotInterface::Buffer(b)) if b.access == AccessSet::READ
                && b.window == Window::Whole
                && b.form == BufferForm::Descriptor(DescriptorView::Storage)
        ),
        "{:?}",
        p.interface.slots.get(&scale)
    );
    let bindings = p
        .layout
        .sets
        .iter()
        .flat_map(|s| &s.bindings)
        .filter(|b| b.source == BindingSource::Buffer(scale))
        .count();
    assert_eq!(bindings, 1, "the argument and the fetch share one binding");

    let mut state = render_state(0);
    let vertex = state.stages.get_mut(&Stage::Vertex).expect("vertex");
    vertex.buffers.insert(2, buffer(512, 8, None));
    let unstrided = p.plan(&dev, &state.shape()).expect("plan without a stride");
    let cmds = unstrided
        .instantiate_draw(
            &state,
            &DrawCall::Primitives {
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        )
        .expect("draw without a stride");
    assert_eq!(
        vertex_push(&unstrided, &cmds, PushField::VertexBufferStride(2)),
        0
    );
    state
        .stages
        .get_mut(&Stage::Vertex)
        .expect("vertex")
        .buffers
        .insert(2, buffer(512, 8, Some(12)));
    let plan = p.plan(&dev, &state.shape()).expect("plan");
    let writes: Vec<&WriteSource> = plan
        .descriptors
        .iter()
        .map(|w| &w.source)
        .filter(|w| **w == WriteSource::Buffer { slot: scale })
        .collect();
    assert_eq!(writes.len(), 1);
    let cmds = plan
        .instantiate_draw(
            &state,
            &DrawCall::Primitives {
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        )
        .expect("draw");
    assert_eq!(
        vertex_push(&plan, &cmds, PushField::VertexBufferStride(1)),
        4
    );
    assert_eq!(
        vertex_push(&plan, &cmds, PushField::VertexBufferStride(2)),
        12
    );
}

// ---- Textures and samplers ---------------------------------------------------------------------

fn kernel(dev: &Device, bc: &[u8], entry: &str) -> Result<ComputePipeline, Error> {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    dev.compile_compute_pipeline(&d, &Budget::standard())
}

fn texture(ty: TextureType, pf: PixelFormat) -> BoundTexture {
    let mut facts = TextureFacts::new(ty, pf);
    facts.usage = metal::TextureUsage::SHADER_READ | metal::TextureUsage::SHADER_WRITE;
    BoundTexture::Texture {
        facts,
        resource_id: None,
    }
}

fn sampler(normalized: bool, reduction: SamplerReductionMode) -> BoundSampler {
    let mut desc = SamplerDescriptor::new();
    desc.min_filter = SamplerMinMagFilter::Linear;
    desc.mag_filter = SamplerMinMagFilter::Linear;
    desc.mip_filter = SamplerMipFilter::Nearest;
    desc.normalized_coordinates = normalized;
    desc.reduction_mode = reduction;
    BoundSampler::new(desc)
}

fn kernel_state(
    textures: &[(u32, BoundTexture)],
    samplers: &[(u32, BoundSampler)],
    out: bool,
) -> BindState {
    let mut t = StageTable::default();
    for (i, x) in textures {
        t.textures.insert(*i, *x);
    }
    for (i, x) in samplers {
        t.samplers.insert(*i, *x);
    }
    if out {
        t.buffers.insert(0, buffer(4096, 0, None));
    }
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    s
}

fn combined(plan: &crate::bind::BindingPlan) -> Vec<(ImageSource, SamplerSource)> {
    plan.descriptors
        .iter()
        .filter_map(|d| match d.source {
            WriteSource::CombinedImageSampler { image, sampler } => Some((image, sampler)),
            _ => None,
        })
        .collect()
}

/// An `ABGR4Unorm` texture sampled with an opaque-black border reads through an
/// `R4G4B4A4`/`B4G4R4A4` view, whose built-in opaque-black border an implementation may swap
/// unless it reports `rgba4OpaqueBlackSwizzled`: the plan refuses the pair where the profile
/// lacks it (moltenvk-m3 has no `VK_KHR_maintenance10`) and plans it where it has it; a
/// transparent-black border, and an `RGBA8Unorm` texture, plan everywhere.
#[test]
fn opaque_black_borders_read_exactly_through_their_views() {
    use crate::device::requirements::VkProperty;
    use crate::error::{Capability, Needs};
    let Some(bc) = fixture("sampling") else {
        return;
    };
    let border = |color: metal::SamplerBorderColor| {
        let mut b = sampler(true, SamplerReductionMode::WeightedAverage);
        b.desc.s_address_mode = metal::SamplerAddressMode::ClampToBorderColor;
        b.desc.border_color = color;
        b
    };
    let (mut refused, mut planned) = (0, 0);
    for (stem, dev) in devices() {
        let Ok(p) = kernel(&dev, &bc, "sample_level") else {
            continue;
        };
        let state = |pf, s| kernel_state(&[(0, texture(TextureType::Type2D, pf))], &[(0, s)], true);
        let plan = |pf, s| p.plan(&dev, &state(pf, s).shape());
        let black = border(metal::SamplerBorderColor::OpaqueBlack);
        let clear = border(metal::SamplerBorderColor::TransparentBlack);
        assert!(plan(PixelFormat::RGBA8Unorm, black).is_ok(), "{stem}");
        let abgr4 = PixelFormat::ABGR4Unorm;
        // A profile that cannot sample the format at all refuses it for the format.
        match plan(abgr4, clear) {
            Ok(_) => {}
            Err(BindError::Unsupported(u)) if u.feature == Feature::Format => continue,
            Err(e) => panic!("{stem}: {e}"),
        }
        let has = dev
            .profile()
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::Rgba4OpaqueBlackSwizzled);
        match plan(abgr4, black) {
            Ok(_) if has => planned += 1,
            Err(BindError::Unsupported(u)) if !has => {
                assert_eq!(u.feature, Feature::SamplerBorderColor, "{stem}");
                let Needs::AnyOf(alternatives) = &u.needs else {
                    panic!("{stem}: {u:?}");
                };
                assert!(
                    alternatives
                        .iter()
                        .all(|a| a
                            .contains(&Capability::Property(VkProperty::Rgba4OpaqueBlackSwizzled))),
                    "{stem}: {u:?}"
                );
                refused += 1;
            }
            other => panic!("{stem} (property {has}): {other:?}"),
        }
    }
    assert!(
        refused > 0 && planned > 0,
        "refused {refused}, planned {planned}"
    );
}

/// Metal reads an opaque-black border of an `A8Unorm` texture, and of a texture whose swizzle reads
/// no alpha channel of its own, as its transparent-black border, so the plan makes such a pair's
/// runtime sampler with transparent black, which every profile reads exactly. A texture read with
/// its own alpha keeps opaque black, which off Metal's sampler, where the swizzle moves that alpha,
/// the device reads through the view with both border-swizzle features, or through a sampler made
/// with the view's mapping with `borderColorSwizzle` alone; without it the pair is refused.
#[test]
fn opaque_black_borders_follow_metal_s_texel_through_the_swizzle() {
    use crate::device::requirements::{VkFeature, VkProperty};
    use crate::metal::sampler::OpaqueBlackBorder as B;
    use metal::TextureSwizzle as T;
    let Some(bc) = fixture("sampling") else {
        return;
    };
    let mut black = sampler(true, SamplerReductionMode::WeightedAverage);
    black.desc.s_address_mode = metal::SamplerAddressMode::ClampToBorderColor;
    black.desc.border_color = metal::SamplerBorderColor::OpaqueBlack;
    let own = metal::TextureSwizzleChannels::DEFAULT;
    let argb = metal::TextureSwizzleChannels::new(T::Alpha, T::Red, T::Green, T::Blue);
    let rrr1 = metal::TextureSwizzleChannels::new(T::Red, T::Red, T::Red, T::One);
    let (mut refused, mut kept, mut mapped) = (0, 0, 0);
    for (stem, dev) in devices() {
        let Ok(p) = kernel(&dev, &bc, "sample_level") else {
            continue;
        };
        let plan = |pf, swizzle| {
            let mut t = texture(TextureType::Type2D, pf);
            if let BoundTexture::Texture { facts, .. } = &mut t {
                facts.swizzle = swizzle;
            }
            p.plan(&dev, &kernel_state(&[(0, t)], &[(0, black)], true).shape())
        };
        let border = |plan: &crate::bind::BindingPlan| match combined(plan).as_slice() {
            [(_, SamplerSource::Slot { opaque_black, .. })] => *opaque_black,
            other => panic!("{stem}: {other:?}"),
        };
        for (pf, swizzle) in [
            (PixelFormat::RGBA8Unorm, rrr1),
            (PixelFormat::R8Unorm, own),
            (PixelFormat::R8Unorm, argb),
            (PixelFormat::A8Unorm, own),
            (PixelFormat::A8Unorm, argb),
        ] {
            match plan(pf, swizzle) {
                Ok(plan) => assert_eq!(border(&plan), B::Transparent, "{stem} {pf:?} {swizzle:?}"),
                // A profile that cannot sample the format refuses it for the format.
                Err(BindError::Unsupported(u)) if u.feature == Feature::Format => {}
                Err(e) => panic!("{stem} {pf:?} {swizzle:?}: {e}"),
            }
        }
        let rgba = plan(PixelFormat::RGBA8Unorm, own).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(border(&rgba), B::Stated, "{stem}");
        let f = dev.profile().fields();
        let has = |feature| f.enabled.features.contains(&feature);
        let exact = f.enabled.properties.contains(&VkProperty::MetalSampler)
            || (has(VkFeature::BorderColorSwizzle) && has(VkFeature::BorderColorSwizzleFromImage));
        match plan(PixelFormat::RGBA8Unorm, argb) {
            Ok(plan) if exact => {
                assert_eq!(border(&plan), B::Stated, "{stem}");
                kept += 1;
            }
            // `borderColorSwizzle` alone: a sampler made for the view's mapping.
            Ok(plan) if has(VkFeature::BorderColorSwizzle) => {
                assert_eq!(
                    border(&plan),
                    B::Mapped(argb.compose(vk::ComponentMapping::IDENTITY)),
                    "{stem}"
                );
                mapped += 1;
            }
            Err(BindError::Unsupported(u)) if !has(VkFeature::BorderColorSwizzle) => {
                assert_eq!(u.feature, Feature::SamplerBorderColor, "{stem}");
                refused += 1;
            }
            other => panic!("{stem} (exact {exact}): {other:?}"),
        }
    }
    assert!(
        kept > 0 && mapped > 0,
        "kept {kept}, mapped {mapped}, refused {refused}"
    );
}

/// A runtime-sampled 2D texture: the bound texture's view and the sampler slot; the declared
/// type's neutrals when unbound; a neutral and a diagnostic for a texture of another type or
/// numeric type; `SamplerState` from the sampler's coordinate mode; a min/max sampler needs
/// `samplerFilterMinmax`. The plan is total over every committed profile where the kernel
/// compiles.
#[test]
fn sampled_textures_and_runtime_samplers() {
    let Some(bc) = fixture("sampling") else {
        return;
    };
    let tex = slot(Stage::Kernel, SlotClass::Texture, 0);
    let smp = slot(Stage::Kernel, SlotClass::Sampler, 0);
    let mut planned = 0;
    for (stem, dev) in devices() {
        let Ok(p) = kernel(&dev, &bc, "sample_level") else {
            continue;
        };
        let bound = kernel_state(
            &[(0, texture(TextureType::Type2D, PixelFormat::RGBA8Unorm))],
            &[(0, sampler(true, SamplerReductionMode::WeightedAverage))],
            true,
        );
        // Vulkan 1.2 requires `R8G8B8A8_UNORM` sampled images, so every profile's format
        // tables map the bound format.
        let plan = p
            .plan(&dev, &bound.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let c = combined(&plan);
        assert_eq!(c.len(), 1, "{stem}");
        match c[0] {
            (ImageSource::Texture { slot: s, view }, SamplerSource::Slot { slot: ss, .. }) => {
                assert_eq!((s, ss), (tex, smp), "{stem}");
                assert_eq!(view.view_type, vk::ImageViewType::TYPE_2D, "{stem}");
                assert_eq!(view.format, vk::Format::R8G8B8A8_UNORM, "{stem}");
                assert_eq!(view.aspect, vk::ImageAspectFlags::COLOR, "{stem}");
            }
            other => panic!("{stem}: {other:?}"),
        }
        assert!(plan.diagnostics.is_empty(), "{stem}");
        // The pixel-coordinate path is selected by specialization when the kernel declares it:
        // the bound sampler's word masked by the fields the kernel reads.
        let state = p
            .stage
            .specialization
            .constants
            .iter()
            .find_map(|c| match c.key {
                SpecKey::SamplerState { slot, fields } if slot == smp => Some((c.id, fields)),
                _ => None,
            });
        if let Some((id, fields)) = state {
            assert_ne!(fields & SamplerWord::PIXEL_COORDINATES, 0, "{stem}");
            let pixel_sampler = sampler(false, SamplerReductionMode::WeightedAverage);
            let pixel = kernel_state(
                &[(0, texture(TextureType::Type2D, PixelFormat::RGBA8Unorm))],
                &[(0, pixel_sampler)],
                true,
            );
            let value = |plan: &crate::bind::BindingPlan| {
                let e = plan
                    .specialization
                    .entries
                    .iter()
                    .find(|e| e.constant_id == id)
                    .expect("SamplerState entry");
                let at = e.offset as usize;
                let bytes = &plan.specialization.data[at..at + 4];
                u32::from_le_bytes(bytes.try_into().expect("a u32"))
            };
            let normalized = sampler(true, SamplerReductionMode::WeightedAverage);
            let keyed = |word: u32| word & fields;
            assert_eq!(
                value(&plan),
                keyed(SamplerWord::of(&normalized.desc).0),
                "{stem}"
            );
            assert_eq!(
                value(&p.plan(&dev, &pixel.shape()).expect("pixel")),
                keyed(SamplerWord::of(&pixel_sampler.desc).0),
                "{stem}"
            );
        }
        // Unbound: the neutral of the declared type and the neutral sampler.
        let empty = kernel_state(&[], &[], true);
        let plan = p.plan(&dev, &empty.shape()).expect("unbound");
        match combined(&plan)[0] {
            (
                ImageSource::Neutral(n @ NeutralResource::Image { view_type, .. }),
                SamplerSource::Neutral(NeutralResource::Sampler { compare: false }),
            ) => {
                assert_eq!(view_type, vk::ImageViewType::TYPE_2D, "{stem}");
                assert!(dev.neutral_kinds().contains(&n), "{stem}: {n:?}");
            }
            other => panic!("{stem}: {other:?}"),
        }
        // Another texture type: a neutral and a diagnostic.
        let s = kernel_state(
            &[(
                0,
                texture(TextureType::Type2DArray, PixelFormat::RGBA8Unorm),
            )],
            &[(0, sampler(true, SamplerReductionMode::WeightedAverage))],
            true,
        );
        let plan = p.plan(&dev, &s.shape()).expect("incompatible");
        assert!(
            matches!(combined(&plan)[0].0, ImageSource::Neutral(_)),
            "{stem}"
        );
        assert!(
            matches!(
                plan.diagnostics[..],
                [PlanDiagnostic::IncompatibleBinding { slot, .. }] if slot == tex
            ),
            "{stem}"
        );
        // Another numeric class: the variant that samples it in its own class.
        let s = kernel_state(
            &[(0, texture(TextureType::Type2D, PixelFormat::RGBA8Uint))],
            &[(0, sampler(true, SamplerReductionMode::WeightedAverage))],
            true,
        );
        let plan = p.plan(&dev, &s.shape()).expect("cross-class");
        assert!(
            plan.diagnostics.is_empty(),
            "{stem}: {:?}",
            plan.diagnostics
        );
        assert_eq!(
            plan.variant.cross_class.get(&tex),
            Some(&crate::pipeline::sym::SampledType::Uint),
            "{stem}"
        );
        // A min/max reduction reads the weighted average on every device.
        let minmax = kernel_state(
            &[(0, texture(TextureType::Type2D, PixelFormat::RGBA8Unorm))],
            &[(0, sampler(true, SamplerReductionMode::Minimum))],
            true,
        );
        let r = p.plan(&dev, &minmax.shape());
        assert!(r.is_ok(), "{stem}: {r:?}");
        // LOD clamps are checked at plan time from the shape, with the verdict `sampler_desc`
        // gives the bound descriptor; anisotropy plans on every device, its Vulkan sampler
        // isotropic without `samplerAnisotropy`.
        let with = |f: &dyn Fn(&mut SamplerDescriptor)| {
            let mut s = sampler(true, SamplerReductionMode::WeightedAverage);
            f(&mut s.desc);
            let state = kernel_state(
                &[(0, texture(TextureType::Type2D, PixelFormat::RGBA8Unorm))],
                &[(0, s)],
                true,
            );
            (p.plan(&dev, &state.shape()), s.desc)
        };
        let (r, desc) = with(&|d| d.max_anisotropy = 8);
        assert!(r.is_ok(), "{stem}: {r:?}");
        let made = sampler_desc(
            &dev,
            &desc,
            SamplerUse::default(),
            crate::metal::sampler::OpaqueBlackBorder::Stated,
            None,
        )
        .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(
            made.max_anisotropy.is_some(),
            dev.profile().fields().images.sampler_anisotropy,
            "{stem}"
        );
        let (r, _) = with(&|d| {
            d.lod_min_clamp = 4.0;
            d.lod_max_clamp = 1.0;
        });
        assert_eq!(
            r,
            Err(BindError::InvalidDescriptor(
                DescriptorError::SamplerLodClamp
            )),
            "{stem}"
        );
        // The dispatch instantiates without texture-specific data.
        let d = Dispatch::Threadgroups {
            groups: [1, 1, 1],
            threads_per_group: [32, 1, 1],
        };
        let plan = p.plan(&dev, &bound.shape()).expect("plan");
        plan.instantiate_dispatch(&bound, &d).expect("dispatch");
        planned += 1;
    }
    assert!(planned >= 3, "sample_level planned on {planned} profiles");
}

/// A texture of a format the table holds no representation of (`ASTC_4x2_HDR`, whose footprint no
/// Vulkan block has) is refused with the format's refusal only at a slot its
/// function uses: at a sampled slot the plan is `Unsupported(Format)`, and bound at an index the
/// function does not declare, or at an argument the function declares and never reads
/// (`reflection/textures`'s `unused_tex`, `air.arg_unused`), the plan is the one without it.
#[test]
fn a_texture_without_a_representation_is_refused_only_where_its_function_reads_it() {
    let (Some(sampling), Some(textures)) = (
        fixture("sampling"),
        match llvm_assemble(
            &engine_dir()
                .with_file_name("reflection")
                .join("textures.ll"),
        ) {
            Ok(b) => Some(b),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected textures.ll: {why}"),
        },
    ) else {
        return;
    };
    let unheld = || texture(TextureType::Type2D, PixelFormat::RGB8Unorm);
    let rgba = || texture(TextureType::Type2D, PixelFormat::RGBA8Unorm);
    let linear = || sampler(true, SamplerReductionMode::WeightedAverage);
    let refused = |r: &Result<BindingPlan, BindError>| matches!(r, Err(BindError::Unsupported(u)) if u.feature == Feature::Format);
    let (mut sampled, mut declared) = (0, 0);
    for (stem, dev) in devices() {
        if let Ok(p) = kernel(&dev, &sampling, "sample_level") {
            let r = p.plan(
                &dev,
                &kernel_state(&[(0, unheld())], &[(0, linear())], true).shape(),
            );
            assert!(refused(&r), "{stem}: a sampled unheld texture: {r:?}");
            let plain = p
                .plan(
                    &dev,
                    &kernel_state(&[(0, rgba())], &[(0, linear())], true).shape(),
                )
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let beside = p
                .plan(
                    &dev,
                    &kernel_state(&[(0, rgba()), (5, unheld())], &[(0, linear())], true).shape(),
                )
                .unwrap_or_else(|e| {
                    panic!("{stem}: an unheld texture at an undeclared index: {e}")
                });
            assert_eq!(beside.descriptors, plain.descriptors, "{stem}");
            assert!(beside.diagnostics.is_empty(), "{stem}");
            sampled += 1;
        }
        if let Ok(p) = kernel(&dev, &textures, "reflection_textures") {
            let unused = p
                .plan(
                    &dev,
                    &kernel_state(&[(16, unheld())], &[(0, linear())], true).shape(),
                )
                .unwrap_or_else(|e| {
                    panic!("{stem}: an unheld texture at the unread argument: {e}")
                });
            let none = p
                .plan(&dev, &kernel_state(&[], &[(0, linear())], true).shape())
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(unused.descriptors, none.descriptors, "{stem}");
            assert!(unused.diagnostics.is_empty(), "{stem}");
            let read = p.plan(
                &dev,
                &kernel_state(&[(3, unheld())], &[(0, linear())], true).shape(),
            );
            assert!(
                refused(&read),
                "{stem}: an unheld texture at a sampled argument: {read:?}"
            );
            declared += 1;
        }
    }
    assert!(
        sampled >= 3 && declared >= 1,
        "sample_level planned on {sampled} profiles, reflection_textures on {declared}"
    );
}

/// A swizzled texture buffer: at the slot `texel_buffers` reads, the plan binds it and specializes
/// the slot's `TextureSwizzle` to its swizzle's word (the identity's for an unswizzled one); at the
/// slot it only writes, which reads no texel to swizzle, it binds with no such constant (Metal
/// makes no swizzled texture a function writes, so only a misuse binds one there); and at the slot
/// `unk_size_queries` only queries it binds too.
#[test]
fn a_swizzled_texel_buffer_specializes_the_slot_that_reads_it() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels as C};
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    let bgr1 = C::new(S::Blue, S::Green, S::Red, S::One);
    let mut planned = 0;
    let mut queried = 0;
    for (stem, dev) in devices() {
        let Ok(p) = kernel(&dev, &bc, "texel_buffers") else {
            continue;
        };
        let tb = |offset, swizzle| BoundTexture::Buffer {
            pixel_format: PixelFormat::RGBA32Float,
            offset,
            length: 1024,
            bytes_per_row: 1024,
            resource_id: None,
            swizzle,
        };
        let read = slot(Stage::Kernel, SlotClass::Texture, 0);
        let id = p
            .stage
            .specialization
            .constants
            .iter()
            .find_map(|c| (c.key == SpecKey::TextureSwizzle(read)).then_some(c.id))
            .unwrap_or_else(|| panic!("{stem}: no TextureSwizzle constant"));
        let word = |plan: &BindingPlan| {
            let e = plan
                .specialization
                .entries
                .iter()
                .find(|e| e.constant_id == id)
                .expect("TextureSwizzle entry");
            let at = e.offset as usize;
            u32::from_le_bytes(
                plan.specialization.data[at..at + 4]
                    .try_into()
                    .expect("a u32"),
            )
        };
        for swizzle in [C::DEFAULT, bgr1] {
            let s = kernel_state(&[(0, tb(0, swizzle)), (1, tb(256, C::DEFAULT))], &[], false);
            let plan = p
                .plan(&dev, &s.shape())
                .unwrap_or_else(|e| panic!("{stem}: {swizzle:?} {e}"));
            assert_eq!(word(&plan), swizzle.word(), "{stem}");
        }
        let written = kernel_state(&[(0, tb(0, C::DEFAULT)), (1, tb(256, bgr1))], &[], false);
        let plan = p
            .plan(&dev, &written.shape())
            .unwrap_or_else(|e| panic!("{stem}: written {e}"));
        assert_eq!(word(&plan), C::DEFAULT.word(), "{stem}");
        let unread = slot(Stage::Kernel, SlotClass::Texture, 1);
        assert!(
            !p.stage
                .specialization
                .constants
                .iter()
                .any(|c| c.key == SpecKey::TextureSwizzle(unread)),
            "{stem}"
        );
        if let Some(queries) = fixture("unk_size_queries") {
            if let Ok(q) = kernel(&dev, &queries, "unk_size_queries") {
                let s = kernel_state(&[(1, tb(0, bgr1))], &[], false);
                q.plan(&dev, &s.shape())
                    .unwrap_or_else(|e| panic!("{stem}: queried {e}"));
                queried += 1;
            }
        }
        planned += 1;
    }
    assert!(
        planned > 0 && queried > 0,
        "{planned} planned, {queried} queried"
    );
}

/// A read cube is viewed as the 2D array it is declared as (`ImageViewShape`); a bound texture
/// buffer gets its view format and its range; an unaligned texture-buffer offset is refused.
#[test]
fn cube_views_and_texel_buffers() {
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    let mut cubes = 0;
    let mut texels = 0;
    for (stem, dev) in devices() {
        if let Ok(p) = kernel(&dev, &bc, "cube_faces") {
            let s = kernel_state(
                &[
                    (0, texture(TextureType::Cube, PixelFormat::RGBA8Unorm)),
                    (1, texture(TextureType::CubeArray, PixelFormat::RGBA8Unorm)),
                ],
                &[],
                true,
            );
            // Vulkan 1.2 requires `R8G8B8A8_UNORM` sampled images.
            let plan = p
                .plan(&dev, &s.shape())
                .unwrap_or_else(|e| panic!("{stem}: cube {e}"));
            let views: Vec<vk::ImageViewType> = plan
                .descriptors
                .iter()
                .filter_map(|d| match d.source {
                    WriteSource::Texture { view, .. } => Some(view.view_type),
                    _ => None,
                })
                .collect();
            assert_eq!(
                views,
                vec![vk::ImageViewType::TYPE_2D_ARRAY; 2],
                "{stem}: {plan:?}"
            );
            cubes += 1;
        }
        if let Ok(p) = kernel(&dev, &bc, "texel_buffers") {
            let tb = |offset| BoundTexture::Buffer {
                pixel_format: PixelFormat::RGBA32Float,
                offset,
                length: 1024,
                bytes_per_row: 1024,
                resource_id: None,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
            };
            let s = kernel_state(&[(0, tb(0)), (1, tb(256))], &[], false);
            // Vulkan 1.2 requires `R32G32B32A32_SFLOAT` uniform texel buffers.
            let plan = p
                .plan(&dev, &s.shape())
                .unwrap_or_else(|e| panic!("{stem}: texel {e}"));
            let formats: Vec<vk::Format> = plan
                .descriptors
                .iter()
                .filter_map(|d| match d.source {
                    WriteSource::TexelBuffer { format, .. } => Some(format),
                    _ => None,
                })
                .collect();
            assert_eq!(formats, vec![vk::Format::R32G32B32A32_SFLOAT; 2], "{stem}");
            let d = Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [1, 1, 1],
            };
            let cmds = plan.instantiate_dispatch(&s, &d).expect("dispatch");
            let ranges: Vec<(u64, u64)> = cmds
                .descriptor_ranges
                .iter()
                .map(|r| (r.range.offset, r.range.range))
                .collect();
            assert_eq!(ranges, vec![(0, 1024), (256, 1024)], "{stem}");
            // The view spans whole texel blocks (16 bytes of `RGBA32Float`), and its element
            // count is checked exactly against `maxTexelBufferElements`.
            let sized = |length| BoundTexture::Buffer {
                pixel_format: PixelFormat::RGBA32Float,
                offset: 0,
                length,
                bytes_per_row: length,
                resource_id: None,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
            };
            let ragged = kernel_state(&[(0, sized(1030)), (1, tb(256))], &[], false);
            let cmds = plan.instantiate_dispatch(&ragged, &d).expect("ragged");
            assert_eq!(cmds.descriptor_ranges[0].range.range, 1024, "{stem}");
            let limit = u64::from(dev.profile().fields().limits.max_texel_buffer_elements);
            let full = kernel_state(&[(0, sized(limit * 16)), (1, tb(256))], &[], false);
            assert!(plan.instantiate_dispatch(&full, &d).is_ok(), "{stem}");
            let over = kernel_state(&[(0, sized((limit + 1) * 16)), (1, tb(256))], &[], false);
            assert_eq!(
                plan.instantiate_dispatch(&over, &d),
                Err(BindError::TexelBufferTooLarge {
                    slot: slot(Stage::Kernel, SlotClass::Texture, 0),
                    elements: limit + 1,
                    limit
                }),
                "{stem}"
            );
            let a = dev
                .profile()
                .fields()
                .limits
                .min_texel_buffer_offset_alignment;
            if a > 1 {
                let unaligned = kernel_state(&[(0, tb(a / 2)), (1, tb(256))], &[], false);
                assert!(
                    matches!(
                        plan.instantiate_dispatch(&unaligned, &d),
                        Err(BindError::Unsupported(_))
                    ),
                    "{stem}"
                );
            }
            texels += 1;
        }
    }
    assert!(
        cubes >= 3 && texels >= 3,
        "cubes {cubes}, texel buffers {texels}"
    );
}

/// A written storage texture asks for the storage variant unless the formatless default
/// declaration serves its choice; the view is the alias format where the choice aliases (bgra8
/// without storage), and the bound format otherwise.
#[test]
fn storage_textures_choose_their_variant() {
    let Some(bc) = fixture("storage_texture_write") else {
        return;
    };
    let dst = slot(Stage::Kernel, SlotClass::Texture, 0);
    let mut variants = 0;
    for (stem, dev) in devices() {
        let Ok(p) = kernel(&dev, &bc, "storage_texture_write") else {
            continue;
        };
        for (pf, format) in [
            (PixelFormat::RGBA8Unorm, vk::Format::R8G8B8A8_UNORM),
            (PixelFormat::BGRA8Unorm, vk::Format::B8G8R8A8_UNORM),
        ] {
            let s = kernel_state(&[(0, texture(TextureType::Type2D, pf))], &[], false);
            let plan = match p.plan(&dev, &s.shape()) {
                Ok(plan) => plan,
                Err(BindError::Unsupported(_)) => continue,
                Err(e) => panic!("{stem}: {e}"),
            };
            let view = plan
                .descriptors
                .iter()
                .find_map(|d| match d.source {
                    WriteSource::Texture { slot, view } if slot == dst => Some(view),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{stem}: no storage write"));
            assert_eq!(view.layout, vk::ImageLayout::GENERAL);
            match plan.variant.storage_images.get(&dst) {
                None => assert_eq!(view.format, format, "{stem}"),
                Some(choice) => {
                    variants += 1;
                    let expected = if choice.alias.is_some() {
                        choice.vk_format
                    } else {
                        format
                    };
                    assert_eq!(view.format, expected, "{stem}: {choice:?}");
                }
            }
        }
    }
    eprintln!("storage variants planned: {variants}");
}

/// A written texture buffer is declared without a format only where the bound format's buffer
/// features state `STORAGE_WRITE_WITHOUT_FORMAT` (`VUID-vkCmdDispatch-OpTypeImage-07029`; nothing
/// implies the bit for a buffer). Elsewhere the plan names the declared format as the slot's
/// storage variant, bound or not (an unbound slot reads and writes its `R32` neutral), and the
/// variant plans to itself.
#[test]
fn a_written_texel_buffer_follows_its_buffer_features() {
    use crate::device::requirements::FormatUsage;
    use crate::pipeline::StorageImageClass;
    let Some(bc) = fixture("ob1_texel_buffer") else {
        return;
    };
    let entry = "ob1_texel_buffer";
    let dst = slot(Stage::Kernel, SlotClass::Texture, 1);
    let r32 = BoundTexture::Buffer {
        pixel_format: PixelFormat::R32Uint,
        offset: 0,
        length: 16,
        bytes_per_row: 16,
        resource_id: None,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
    };
    let state = |textures: &[(u32, BoundTexture)]| {
        let mut s = kernel_state(textures, &[], true);
        if let Some(t) = s.stages.get_mut(&Stage::Kernel) {
            t.buffers.insert(1, buffer(4096, 0, None));
        }
        s
    };
    let mut declared = BTreeSet::new();
    let mut formatless = BTreeSet::new();
    for (stem, dev) in devices() {
        let Ok(p) = kernel(&dev, &bc, entry) else {
            continue;
        };
        let profile = dev.profile();
        let without = profile.fields().images.storage_write_without_format
            && profile.supports_format(
                vk::Format::R32_UINT,
                FormatUsage::StorageTexelBufferWriteWithoutFormat,
            );
        for (what, s) in [
            ("bound", state(&[(0, r32), (1, r32)])),
            ("unbound", state(&[(0, r32)])),
        ] {
            let plan = p
                .plan(&dev, &s.shape())
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            let Some(choice) = plan.variant.storage_images.get(&dst) else {
                assert!(without, "{stem} {what}: no variant");
                formatless.insert(stem.clone());
                continue;
            };
            assert!(!without, "{stem} {what}: {choice:?}");
            assert_eq!(choice.vk_format, vk::Format::R32_UINT, "{stem} {what}");
            assert_eq!(choice.class, StorageImageClass::Identity, "{stem} {what}");
            assert_eq!(choice.alias, None, "{stem} {what}");
            declared.insert(stem.clone());
            let lib = Library::from_air(&bc).expect("library");
            let f = lib.function(entry).expect("entry");
            let compiled = dev
                .compile_function(
                    &FunctionDescriptor::new(f),
                    &plan.variant.lowering_inputs(Stage::Kernel),
                    &Budget::standard(),
                )
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            let variant = dev
                .compile_compute_pipeline(
                    &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
                    &Budget::standard(),
                )
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            let again = variant
                .plan(&dev, &s.shape())
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            assert_eq!(again.variant, plan.variant, "{stem} {what}: a fixed point");
        }
    }
    // SwiftShader has `shaderStorageImageWriteWithoutFormat`, but its `R32_UINT` buffer features
    // lack the bit; lavapipe's state it.
    assert!(declared.contains("swiftshader"), "{declared:?}");
    assert!(formatless.contains("lavapipe"), "{formatless:?}");
}

/// The value `plan` specializes constant `id` to (a `bool`'s low byte).
fn spec_value(plan: &BindingPlan, id: u32) -> u8 {
    let e = plan
        .specialization
        .entries
        .iter()
        .find(|e| e.constant_id == id)
        .expect("specialized");
    plan.specialization.data[e.offset as usize]
}

/// A read the shader checks whose alpha the function observes takes its out-of-bounds alpha from
/// `OutOfBoundsAlpha`, which the plan sets from the bound pixel format: 1 for a format without
/// alpha, 0 for one with it, and 0 for an unbound slot or one bound to a placeholder format (the
/// slot reads its neutral, which reads zero). A compute texel buffer declares it where
/// `robustBufferAccess2` is absent, a fragment texture where `robustImageAccess2` is.
#[test]
fn out_of_bounds_alpha_follows_the_bound_format() {
    let (Some(ops), Some(io), Some(tail)) = (
        fixture("texture_ops"),
        fixture("stage_io"),
        fixture("sampling_tail"),
    ) else {
        return;
    };
    let src = slot(Stage::Kernel, SlotClass::Texture, 0);
    let frag = slot(Stage::Fragment, SlotClass::Texture, 0);
    let tb = |pixel_format, offset| BoundTexture::Buffer {
        pixel_format,
        offset,
        length: 1024,
        bytes_per_row: 1024,
        resource_id: None,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
    };
    let (mut kernels, mut fragments) = (0, 0);
    for (stem, dev) in devices() {
        let f = dev.profile().fields();
        let Ok(p) = kernel(&dev, &ops, "texel_buffers") else {
            continue;
        };
        let c = p
            .stage
            .specialization
            .constants
            .iter()
            .find(|c| c.key == SpecKey::OutOfBoundsAlpha(src));
        let checked = !matches!(
            f.containment,
            Containment::On {
                robust: Robustness::Rba2
            }
        );
        assert_eq!(c.is_some(), checked, "{stem}");
        if let Some(c) = c {
            // Vulkan 1.2 requires `R32_SFLOAT` and `R32G32B32A32_SFLOAT` uniform texel buffers.
            for (pf, want) in [(PixelFormat::RGBA32Float, 0), (PixelFormat::R32Float, 1)] {
                let s = kernel_state(
                    &[(0, tb(pf, 0)), (1, tb(PixelFormat::RGBA32Float, 256))],
                    &[],
                    false,
                );
                let plan = p
                    .plan(&dev, &s.shape())
                    .unwrap_or_else(|e| panic!("{stem}: {pf:?}: {e}"));
                assert_eq!(spec_value(&plan, c.id), want, "{stem}: {pf:?}");
            }
            let unbound = kernel_state(&[(1, tb(PixelFormat::RGBA32Float, 256))], &[], false);
            let plan = p.plan(&dev, &unbound.shape()).expect("unbound");
            assert_eq!(spec_value(&plan, c.id), 0, "{stem}: unbound");
            let placeholder = kernel_state(
                &[
                    (0, tb(PixelFormat::Invalid, 0)),
                    (1, tb(PixelFormat::RGBA32Float, 256)),
                ],
                &[],
                false,
            );
            let plan = p.plan(&dev, &placeholder.shape()).expect("placeholder");
            assert_eq!(spec_value(&plan, c.id), 0, "{stem}: placeholder");
            kernels += 1;
        }
        let vertex = Library::from_air(&io).expect("library");
        let fragment = Library::from_air(&tail).expect("library");
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            vertex.function("stage_io_vertex_position").expect("vertex"),
        )));
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            fragment
                .function("fragment_read_across_blocks")
                .expect("fragment"),
        )));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let Ok(r) = dev.compile_render_pipeline(&d, &Budget::standard()) else {
            continue;
        };
        let c = r
            .stages
            .iter()
            .flat_map(|m| m.specialization.constants.iter())
            .find(|c| c.key == SpecKey::OutOfBoundsAlpha(frag))
            .copied();
        assert_eq!(c.is_some(), !f.images.robust_image_access2, "{stem}");
        let Some(c) = c else {
            continue;
        };
        // Vulkan 1.2 requires `R8_UNORM` and `R8G8B8A8_UNORM` sampled images.
        for (pf, want) in [(PixelFormat::RGBA8Unorm, 0), (PixelFormat::R8Unorm, 1)] {
            let mut t = StageTable::default();
            t.textures.insert(0, texture(TextureType::Type2D, pf));
            t.buffers.insert(0, buffer(256, 0, None));
            let mut s = BindState::default();
            s.stages.insert(Stage::Fragment, t);
            let plan = r
                .plan(&dev, &s.shape())
                .unwrap_or_else(|e| panic!("{stem}: {pf:?}: {e}"));
            assert_eq!(spec_value(&plan, c.id), want, "{stem}: fragment {pf:?}");
        }
        fragments += 1;
    }
    assert!(
        kernels > 0 && fragments > 0,
        "kernels {kernels}, fragments {fragments}"
    );
}

/// `a2_sparse_status` observes the residency of a sample and a gather of texture 0 through a
/// nearest, clamp-to-edge constexpr sampler, and of a read of texture 1. Each slot states
/// `RuntimeFact::SparseTexture`, its `SparseTexture` specialization, which the plan sets from the
/// bound texture's `sparse` (`MTLTexture.isSparse`). Slot 1's read has the tile-map arm on every
/// device, and so has slot 0 where the device has no `shaderResourceResidency`, its sample and
/// gather through Metal's footprint: a `ResourceKind::TileMap` binding the plan writes as the
/// slot's `WriteSource::TileMap` (read by the kernel) for a sparse texture and as the zero buffer
/// otherwise. Where the device has residency, slot 0's sample and gather are the device's sparse
/// accesses (the sparse arm), with no tile map.
#[test]
fn a_sparse_texture_is_answered_from_its_tile_map_where_the_device_has_no_residency() {
    use crate::device::requirements::VkFeature;
    use crate::pipeline::facts::RuntimeFact;
    let Some(bc) = fixture("a2_sparse_status") else {
        return;
    };
    let bound = |sparse: bool| {
        let mut t = texture(TextureType::Type2D, PixelFormat::RGBA32Float);
        if let BoundTexture::Texture { facts, .. } = &mut t {
            facts.sparse = sparse;
        }
        t
    };
    let state = |s0: bool, s1: bool| kernel_state(&[(0, bound(s0)), (1, bound(s1))], &[], true);
    let (mut with, mut without) = (0, 0);
    for (stem, dev) in devices() {
        let p = kernel(&dev, &bc, "a2_sparse_status").unwrap_or_else(|e| panic!("{stem}: {e}"));
        let residency = dev
            .profile()
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderResourceResidency);
        let slots = [0, 1].map(|i| slot(Stage::Kernel, SlotClass::Texture, i));
        for (s, sampled) in slots.into_iter().zip([true, false]) {
            let fact = RuntimeFact::SparseTexture {
                slot: s,
                sparse_arm: sampled && residency,
                tile_map: !(sampled && residency),
            };
            assert!(
                p.dependencies.iter().any(|d| d.fact == fact),
                "{stem}: {fact:?}"
            );
        }
        let ids = slots.map(|s| {
            p.stage
                .specialization
                .constants
                .iter()
                .find(|c| c.key == SpecKey::SparseTexture(s))
                .map(|c| c.id)
        });
        let [Some(a), Some(b)] = ids else {
            panic!("{stem}: no SparseTexture constants: {ids:?}");
        };
        let tile_maps = |plan: &BindingPlan| {
            plan.descriptors
                .iter()
                .filter_map(|d| match &d.source {
                    WriteSource::TileMap { slot } => Some(*slot),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let plain = p
            .plan(&dev, &state(false, false).shape())
            .unwrap_or_else(|e| panic!("{stem}: not sparse: {e}"));
        assert_eq!(tile_maps(&plain), [], "{stem}");
        assert_eq!(
            (spec_value(&plain, a), spec_value(&plain, b)),
            (0, 0),
            "{stem}"
        );
        let plan = p
            .plan(&dev, &state(true, false).shape())
            .unwrap_or_else(|e| panic!("{stem}: sparse slot 0: {e}"));
        assert_eq!(
            (spec_value(&plan, a), spec_value(&plan, b)),
            (1, 0),
            "{stem}"
        );
        // Each sparse slot's tile map bound and read.
        let mapped: &[SlotRef] = match residency {
            true => &slots[1..],
            false => &slots,
        };
        assert_eq!(tile_maps(&plan), mapped[..mapped.len() - 1], "{stem}");
        let both = p
            .plan(&dev, &state(true, true).shape())
            .unwrap_or_else(|e| panic!("{stem}: sparse slots: {e}"));
        assert_eq!(tile_maps(&both), mapped, "{stem}");
        for &s in mapped {
            assert!(
                both.uses
                    .iter()
                    .any(|u| u.resource == UseTarget::TileMap(s)),
                "{stem}: no tile-map use of {s:?}"
            );
        }
        if residency {
            with += 1;
        } else {
            without += 1;
        }
    }
    assert!(with > 0 && without > 0, "with {with}, without {without}");
}

/// `OpImageSparseSampleExplicitLod`, `OpImageSparseFetch`, and `OpImageSparseGather`.
const SPARSE_SAMPLE_EXPLICIT_LOD: u32 = 306;
const SPARSE_FETCH: u32 = 313;
const SPARSE_GATHER: u32 = 314;

/// The opcodes of a SPIR-V module.
fn opcodes(words: &[u32]) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        out.insert(w & 0xFFFF);
        i += usize::try_from(w >> 16).unwrap_or(1).max(1);
    }
    out
}

/// `sps2_residency_probes` observes the residency of nearest and linear samples and a gather
/// through clamp-to-zero constexpr samplers at level 0, every one inside Metal's measured footprint
/// rule. Where the device has no `shaderResourceResidency` the module answers them from the texels
/// Metal counts, through the tile map of both slots, which bind a sparse texture: no sparse
/// access, and no note. Where it has, each is the device's sparse access, its residency the
/// device's, recorded as `PrecisionNote::SparseResidency`, and the sampled slot has no tile map;
/// the slot read alone keeps its own.
#[test]
fn a_sample_with_metals_footprint_is_answered_where_the_device_has_no_residency() {
    use crate::device::requirements::VkFeature;
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::facts::RuntimeFact;
    let Some(bc) = fixture("sps2_residency_probes") else {
        return;
    };
    let bound = |sparse: bool| {
        let mut t = texture(TextureType::Type2D, PixelFormat::RGBA8Unorm);
        if let BoundTexture::Texture { facts, .. } = &mut t {
            facts.sparse = sparse;
        }
        t
    };
    let sparse = kernel_state(&[(0, bound(true)), (1, bound(true))], &[], true);
    let mut seen = [0; 2];
    for (stem, dev) in devices() {
        let p =
            kernel(&dev, &bc, "sps2_residency_probes").unwrap_or_else(|e| panic!("{stem}: {e}"));
        let residency = dev
            .profile()
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderResourceResidency);
        // Slot 1 is read alone, from its tile map on every device.
        for (i, sampled) in [(0, true), (1, false)] {
            let fact = RuntimeFact::SparseTexture {
                slot: slot(Stage::Kernel, SlotClass::Texture, i),
                sparse_arm: sampled && residency,
                tile_map: !(sampled && residency),
            };
            assert!(
                p.dependencies.iter().any(|d| d.fact == fact),
                "{stem}: {fact:?}"
            );
        }
        let ops = opcodes(&p.stage.spirv);
        assert_eq!(
            ops.contains(&SPARSE_SAMPLE_EXPLICIT_LOD),
            residency,
            "{stem}"
        );
        assert_eq!(ops.contains(&SPARSE_GATHER), residency, "{stem}");
        // The sampled slot's read through its sparse arm: one texel's residency, exact.
        assert_eq!(ops.contains(&SPARSE_FETCH), residency, "{stem}");
        assert_eq!(
            p.stage
                .precision_notes
                .contains(&PrecisionNote::SparseResidency),
            residency,
            "{stem}"
        );
        p.plan(&dev, &sparse.shape())
            .unwrap_or_else(|e| panic!("{stem}: sparse: {e}"));
        seen[usize::from(residency)] += 1;
    }
    assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
}

/// Where the profile's sparse textures are sparse images, `sps2_residency_probes`' sampled slot
/// specializes on `ConstantLanes`, which the plan sets from the bound texture: a sparse `R32Float`
/// texture's missing green and blue (0) and alpha (1), through its swizzle; 0 for a texture that is
/// not sparse, and for one whose format holds every channel. Elsewhere the slot declares no such
/// constant.
#[test]
fn a_sparse_texture_s_constant_lanes_are_its_format_s_through_its_swizzle() {
    use crate::device::requirements::VkFeature;
    use metal::{TextureSwizzle as T, TextureSwizzleChannels as S};
    let Some(bc) = fixture("sps2_residency_probes") else {
        return;
    };
    let bound = |pf: PixelFormat, sparse: bool, swizzle: S| {
        let mut t = texture(TextureType::Type2D, pf);
        if let BoundTexture::Texture { facts, .. } = &mut t {
            facts.sparse = sparse;
            facts.swizzle = swizzle;
        }
        t
    };
    let cases = [
        (
            PixelFormat::R32Float,
            true,
            S::DEFAULT,
            0b1110 | 0b1000 << 4,
        ),
        (PixelFormat::R32Float, false, S::DEFAULT, 0),
        (PixelFormat::RGBA8Unorm, true, S::DEFAULT, 0),
        (
            PixelFormat::R32Float,
            true,
            S::new(T::Alpha, T::Red, T::One, T::Zero),
            0b1101 | 0b0101 << 4,
        ),
    ];
    let mut seen = [0; 2];
    for (stem, dev) in devices() {
        let p =
            kernel(&dev, &bc, "sps2_residency_probes").unwrap_or_else(|e| panic!("{stem}: {e}"));
        let residency = dev
            .profile()
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderResourceResidency);
        seen[usize::from(residency)] += 1;
        let s0 = slot(Stage::Kernel, SlotClass::Texture, 0);
        let id = p
            .stage
            .specialization
            .constants
            .iter()
            .find(|c| c.key == SpecKey::ConstantLanes(s0))
            .map(|c| c.id);
        assert_eq!(id.is_some(), residency, "{stem}");
        let Some(id) = id else {
            continue;
        };
        for (pf, sparse, swizzle, want) in cases {
            let state = kernel_state(
                &[
                    (0, bound(pf, sparse, swizzle)),
                    (1, bound(PixelFormat::R32Float, sparse, S::DEFAULT)),
                ],
                &[],
                true,
            );
            let plan = p
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{stem}: {pf:?} {swizzle:?}: {e}"));
            assert_eq!(
                spec_value(&plan, id),
                want,
                "{stem}: {pf:?} sparse {sparse} {swizzle:?}"
            );
        }
    }
    assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
}

/// Metal drops every write a function makes to a sparse texture. `spse_write_probes` writes the
/// texture whose residency it reads, so its slot states `SparseTextureWrites` and a residency arm
/// (the tile map on every device), and binds a texture that is sparse and one that is not on every
/// device.
#[test]
fn a_written_sparse_texture_binds() {
    use crate::pipeline::facts::RuntimeFact;
    let Some(bc) = fixture("spse_write_probes") else {
        return;
    };
    let at = slot(Stage::Kernel, SlotClass::Texture, 0);
    let bound = |sparse: bool| {
        let mut t = texture(TextureType::Type2D, PixelFormat::R32Float);
        if let BoundTexture::Texture { facts, .. } = &mut t {
            facts.sparse = sparse;
        }
        kernel_state(&[(0, t)], &[], true)
    };
    let mut seen = 0;
    for (stem, dev) in devices() {
        let p = match kernel(&dev, &bc, "spse_write_probes") {
            Ok(p) => p,
            // A storage texture without a format the profile writes.
            Err(Error::NeedsInput(_)) => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        for fact in [
            RuntimeFact::SparseTextureWrites {
                slot: at,
                atomic: false,
            },
            RuntimeFact::SparseTexture {
                slot: at,
                sparse_arm: false,
                tile_map: true,
            },
        ] {
            assert!(
                p.dependencies.iter().any(|d| d.fact == fact),
                "{stem}: {fact:?}"
            );
        }
        for sparse in [false, true] {
            p.plan(&dev, &bound(sparse).shape())
                .unwrap_or_else(|e| panic!("{stem}: sparse {sparse}: {e}"));
        }
        seen += 1;
    }
    assert!(seen > 0);
}

/// A 4:2:2 texture (`GBGR422`, `BGRG422`) at a storage slot its function only writes
/// (`storage_texture_write`) binds the slot's neutral with an `IncompatibleBinding` note: plain
/// Metal completes the writes and stores nothing. At a slot that samples it (`sampling`'s
/// `sample_level`) the plan samples the decoded twin's view of Metal's texels, `R8G8B8A8_UNORM`
/// (`FormatEmulation::Decoded`).
#[test]
fn a_four_two_two_texture_a_function_only_writes_binds_the_neutral() {
    let (Some(write), Some(sampling)) = (fixture("storage_texture_write"), fixture("sampling"))
    else {
        return;
    };
    let linear = || sampler(true, SamplerReductionMode::WeightedAverage);
    let twin_view = |plan: &BindingPlan| {
        plan.descriptors.iter().any(|d| {
            matches!(
                &d.source,
                WriteSource::CombinedImageSampler {
                    image: ImageSource::Texture { view, .. },
                    ..
                } if view.format == vk::Format::R8G8B8A8_UNORM
            )
        })
    };
    let (mut written, mut sampled) = (0, 0);
    for (stem, dev) in devices() {
        for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
            let pairs = || texture(TextureType::Type2D, pf);
            if let Ok(p) = kernel(&dev, &write, "storage_texture_write") {
                let plan = p
                    .plan(&dev, &kernel_state(&[(0, pairs())], &[], false).shape())
                    .unwrap_or_else(|e| panic!("{stem} {pf:?}: {e}"));
                assert!(
                    plan.descriptors.iter().any(|d| matches!(
                        d.source,
                        WriteSource::Neutral(NeutralResource::Image { .. })
                    )),
                    "{stem} {pf:?}: {:?}",
                    plan.descriptors
                );
                assert!(
                    !plan
                        .descriptors
                        .iter()
                        .any(|d| matches!(d.source, WriteSource::Texture { .. })),
                    "{stem} {pf:?}"
                );
                assert!(
                    matches!(
                        plan.diagnostics[..],
                        [PlanDiagnostic::IncompatibleBinding { .. }]
                    ),
                    "{stem} {pf:?}: {:?}",
                    plan.diagnostics
                );
                written += 1;
            }
            if let Ok(p) = kernel(&dev, &sampling, "sample_level") {
                let plan = p
                    .plan(
                        &dev,
                        &kernel_state(&[(0, pairs())], &[(0, linear())], true).shape(),
                    )
                    .unwrap_or_else(|e| panic!("{stem} {pf:?}: a sampled 4:2:2 texture: {e}"));
                assert!(twin_view(&plan), "{stem} {pf:?}: {:?}", plan.descriptors);
                sampled += 1;
            }
        }
    }
    assert!(
        written >= 6 && sampled >= 6,
        "{written} written, {sampled} sampled"
    );
}
