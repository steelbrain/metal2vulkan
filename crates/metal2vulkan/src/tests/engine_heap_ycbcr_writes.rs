//! An argument buffer's `write()` to a two-plane YCbCr texture (`bind::heap`,
//! `memory::handles::FLAG_YCBCR_WRITES`).
//!
//! Metal stores such a write as it stores a slot's (an Apple M3 Max under macOS 27.0.1): a 4:4:4
//! format whose writes it stores into both planes (`metal::format::ycbcr_writes_stored`) through
//! the texture's write shadow, whose texels the heap's float storage element holds and a write
//! through the handle marks, and nothing of a format whose writes it drops
//! (`metal::format::writes_dropped`), whose storage element is the class's neutral. The handle
//! carries the entry's flag only under the stage's `SpecKey::HeapYCbCrWrites`, which a plan gives
//! from what the command names: a command that names no such texture keeps its pipeline's key.
//! A stage that reads the storage class (a `read_write` member) refuses such a texture it reaches
//! there, as a slot that reads one is refused. No GPU is used.

use std::collections::BTreeMap;
use std::path::Path;

use crate::bind::{
    handle_entry_flags, heap_element, BindState, BindingPlan, Dispatch, HeapElement, ResourceClass,
    TextureFacts, UsedResource, WriteSource,
};
use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Error, Feature, Needs, Unsupported};
use crate::library::Library;
use crate::memory::handles::FLAG_YCBCR_WRITES;
use crate::metal::{PixelFormat, TextureType, TextureUsage};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{
    Axis, HandleClass, ImageAccess, ImageHandleClass, ResourceKind, ResourceSlot, SpecKey, Stage,
    SyntheticBinding,
};
use crate::pipeline::testing::typed_module;
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, StageSource,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Resources};
use crate::vk;

use super::engine_binder::assemble_text;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Kernel `heap_ycbcr_write` writes buffer 1's `float4` at its index through the
/// `texture2d<float, write>` argument buffer 0 holds, at `(gid, 0)`; kernel `heap_ycbcr_rw` reads
/// the `read_write` texture argument buffer 0 holds at `(gid, 0)` into buffer 1 and writes buffer
/// 1's `float4` there.
const KERNELS: &str = r#"%struct.T = type { ptr addrspace(1) }
define void @heap_ycbcr_write(ptr addrspace(2) %r, ptr addrspace(1) %src, i32 %gid) {
  %pt = getelementptr inbounds %struct.T, ptr addrspace(2) %r, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %src, i32 %gid
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %c, <4 x float> %v, i32 0, i32 2)
  ret void
}
define void @heap_ycbcr_rw(ptr addrspace(2) %r, ptr addrspace(1) %src, i32 %gid) {
  %pt = getelementptr inbounds %struct.T, ptr addrspace(2) %r, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %src, i32 %gid
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %smp = call ptr addrspace(2) @air.get_read_sampler()
  %x = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 3)
  %xv = extractvalue { <4 x float>, i8 } %x, 0
  store <4 x float> %xv, ptr addrspace(1) %p, align 16
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %c, <4 x float> %v, i32 0, i32 3)
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.kernel = !{!0, !20}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_ycbcr_write, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"T", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"src"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{ptr @heap_ycbcr_rw, !1, !21}
!21 = !{!22, !27, !8}
!22 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !23, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"T", !"air.arg_name", !"r"}
!23 = !{i32 0, i32 8, i32 0, !"texture2d<float, read_write>", !"t", !"air.indirect_argument", !24}
!24 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"t"}
!27 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"src"}
"#;

/// The resource IDs of the members: a 4:4:4 YCbCr texture's, and a plain one's.
const YCBCR_ID: u64 = 0x0000_0001_0000_0010;
const PLAIN_ID: u64 = 0x0000_0001_0000_0020;

/// The committed profiles each kernel is tried on; one without a storage heap refuses it.
const STEMS: [&str; 3] = ["lavapipe", "moltenvk-m3", "amd-780m-radv"];

/// The device of committed profile `stem`.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// [`KERNELS`], assembled; `None` where `llvm-as` is missing (the test then skips visibly).
fn kernels() -> Option<Vec<u8>> {
    assemble_text("heap_ycbcr_writes", &format!("{HEADER}{KERNELS}"))
}

/// Kernel `entry` of `bc` on `dev`, `None` where the device has no heap for it.
fn pipeline(dev: &Device, bc: &[u8], entry: &str) -> Option<ComputePipeline> {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    match dev.compile_compute_pipeline(&d, &Budget::standard()) {
        Ok(p) => Some(p),
        Err(Error::Unsupported(u)) if u.feature == Feature::ArgumentBufferTextures => None,
        Err(e) => panic!("{entry}: {e}"),
    }
}

/// A 2D texture of `pf` and `usage`.
fn facts(pf: PixelFormat, usage: TextureUsage) -> TextureFacts {
    let mut f = TextureFacts::new(TextureType::Type2D, pf);
    f.usage = usage;
    f
}

/// A command making the textures of `members` resident, each with its resource ID.
fn state(members: &[(TextureFacts, u64)]) -> BindState {
    BindState {
        used_resources: (members.iter().zip(1u32..))
            .map(|(&(f, id), token)| {
                let mut u = UsedResource::new(ResourceClass::Texture(f), token);
                u.resource_id = Some(id);
                u
            })
            .collect(),
        ..Default::default()
    }
}

/// The value `plan` gives `pipe`'s constant of `key`.
fn given(pipe: &ComputePipeline, plan: &BindingPlan, key: SpecKey) -> Option<u64> {
    let constant = (pipe.stage.specialization.constants.iter()).find(|c| c.key == key)?;
    let entry = (plan.specialization.entries.iter()).find(|e| e.constant_id == constant.id)?;
    let at = entry.offset as usize;
    let bytes = &plan.specialization.data[at..at + entry.size as usize];
    let mut word = [0u8; 8];
    word[..bytes.len()].copy_from_slice(bytes);
    Some(u64::from_le_bytes(word))
}

/// The heap's float 2D storage class.
fn storage_class() -> ImageHandleClass {
    ImageHandleClass {
        dim: crate::pipeline::sym::ImageDim::D2,
        arrayed: false,
        multisampled: false,
        sampled_type: crate::pipeline::sym::SampledType::Float,
        depth_compare: false,
        access: ImageAccess::Storage,
        format: crate::pipeline::sym::StorageFormatClass::Unknown,
    }
}

const WRITE: TextureUsage = TextureUsage::SHADER_WRITE;

/// A heap write marks a 4:4:4 YCbCr member's write shadow, and only under its constant: end to
/// end on the interpreter, the binder's plan and table bytes for a command making a
/// `YCBCR8_444_2P` texture and an `RGBA32Float` one resident (the YCbCr one's element its write
/// shadow's `R32G32B32A32_SFLOAT` view, its entry flagged `FLAG_YCBCR_WRITES`, the plan giving
/// `SpecKey::HeapYCbCrWrites` 1), and the emitted search and write. Four texels of alpha 0.25
/// written through the YCbCr texture's handle store their red, green and blue and alpha 1, the
/// mark, where the constant is 1, and their alpha where it is 0; through the plain texture's,
/// their alpha under either value.
#[test]
fn a_heap_write_marks_a_ycbcr_member_s_write_shadow_under_its_constant() {
    let Some(bc) = kernels() else {
        return;
    };
    let entry = "heap_ycbcr_write";
    let key = SpecKey::HeapYCbCrWrites(Stage::Kernel);
    let fl = |v: f32| v.to_bits();
    let texels: [[f32; 4]; 4] = [
        [0.2, 0.6, 0.9, 0.25],
        [1.0, 0.0, 0.0, 0.25],
        [0.5, 0.5, 0.5, 0.25],
        [0.0, 0.0, 1.0, 0.25],
    ];
    let mut ran = 0;
    for stem in STEMS {
        let dev = device(stem);
        let Some(pipe) = pipeline(&dev, &bc, entry) else {
            continue;
        };
        let constant = (pipe.stage.specialization.constants.iter())
            .find(|c| c.key == key)
            .unwrap_or_else(|| panic!("{stem}: no {key:?} constant"));
        assert_eq!(constant.default, 1, "{stem}");
        let ycbcr = facts(PixelFormat::YCBCR84442P, WRITE);
        let plain = facts(PixelFormat::RGBA32Float, WRITE);
        let state = state(&[(ycbcr, YCBCR_ID), (plain, PLAIN_ID)]);
        let bplan = pipe.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(given(&pipe, &bplan, key), Some(1), "{stem}");
        let views: Vec<(u32, vk::Format)> = (bplan.descriptors.iter())
            .filter_map(|d| match &d.source {
                WriteSource::HeapEntry { element, view, .. } => Some((*element, view.format)),
                _ => None,
            })
            .collect();
        assert_eq!(
            views,
            vec![
                (1, vk::Format::R32G32B32A32_SFLOAT),
                (2, vk::Format::R32G32B32A32_SFLOAT)
            ],
            "{stem}: the write shadow's float view, and the plain texture's own"
        );
        let dispatch = Dispatch::Threadgroups {
            groups: [4, 1, 1],
            threads_per_group: [1, 1, 1],
        };
        let cmds = bplan
            .instantiate_dispatch(&state, &dispatch)
            .expect("dispatch");
        let (set, binding) = (bplan.descriptors.iter())
            .find_map(|d| {
                matches!(d.source, WriteSource::HandleTable { .. }).then_some((d.set, d.binding))
            })
            .expect("a handle table");
        let table = (cmds.descriptor_bytes.iter())
            .find(|b| (b.set, b.binding) == (set, binding))
            .expect("table bytes")
            .bytes
            .clone();
        // The entries past the null one, by ID: the YCbCr texture's flagged, the plain one's not.
        let flags: Vec<(u64, u32)> = table
            .chunks(16)
            .map(|e| {
                let word = |w: usize| u32::from_le_bytes(e[4 * w..4 * w + 4].try_into().unwrap());
                (u64::from(word(0)) | u64::from(word(1)) << 32, word(3))
            })
            .filter(|(id, _)| *id != 0)
            .collect();
        assert_eq!(
            flags
                .iter()
                .map(|(id, f)| (*id, f & FLAG_YCBCR_WRITES))
                .collect::<Vec<_>>(),
            vec![(YCBCR_ID, FLAG_YCBCR_WRITES), (PLAIN_ID, 0)],
            "{stem}"
        );

        let (plan, m) = typed_module(&bc, entry, dev.profile()).expect("module");
        assert!(plan.families.access.heap_ycbcr_writes, "{stem}");
        let class = (plan.families.texture.heap.values())
            .find_map(|h| h.class)
            .expect("a heap class");
        assert_eq!(class, storage_class(), "{stem}");
        let slot = ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Heap(HandleClass::Image(class)),
            index: 0,
        };
        let images: Vec<Image> = [(0, 1), (1, 4), (2, 4)]
            .into_iter()
            .map(|(e, width)| Image::new(slot, [width, 1, 1], 1, 1, 1).at_element(e))
            .collect();
        let src: Vec<u8> = texels
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let buffer = |kind, index, bytes| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind,
                index,
            }),
            address: None,
            bytes,
        };
        for (id, element) in [(YCBCR_ID, 1usize), (PLAIN_ID, 2)] {
            for value in [0u64, 1] {
                let mut spec = BTreeMap::new();
                for axis in [Axis::X, Axis::Y, Axis::Z] {
                    spec.insert(SpecKey::LocalSize(axis), 1);
                }
                spec.insert(SpecKey::FlatLocalSize, 1);
                spec.insert(key, value);
                let res = Resources {
                    buffers: vec![
                        buffer(ResourceKind::Buffer, 0, id.to_le_bytes().to_vec()),
                        buffer(ResourceKind::Buffer, 1, src.clone()),
                        buffer(
                            ResourceKind::Synthetic(SyntheticBinding::HandleTable),
                            0,
                            table.clone(),
                        ),
                    ],
                    push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
                    spec,
                };
                let cfg = Config {
                    workgroups: [4, 1, 1],
                    ..Config::default()
                };
                let (_, out) = run_with_images(&m, &cfg, res, images.clone())
                    .unwrap_or_else(|e| panic!("{stem}: {id:#x} under {value}: {e:?}"));
                let marked = id == YCBCR_ID && value == 1;
                let want: Vec<[u32; 4]> = texels
                    .iter()
                    .map(|&[r, g, b, a]| [fl(r), fl(g), fl(b), fl(if marked { 1.0 } else { a })])
                    .collect();
                assert_eq!(
                    out[element].texels, want,
                    "{stem}: {id:#x} under {value}: the stored texels"
                );
                for other in (0..3).filter(|&e| e != element) {
                    assert_eq!(
                        out[other].texels, images[other].texels,
                        "{stem}: {id:#x} under {value}: element {other} untouched"
                    );
                }
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no committed profile has a storage heap");
}

/// The plan gives `SpecKey::HeapYCbCrWrites` 1 exactly where a texture the command makes resident
/// is a 4:4:4 YCbCr one whose writes Metal stores and whose usage writes (`YCBCR8_444_2P`,
/// `RGB10_444_2P` beside an `RGBA8Unorm` one), and 0 where it makes none, plain ones
/// (`RGBA8Unorm`, `RGBA16Float`), one whose writes Metal drops (`YCBCR8_420_2P`), or a 4:4:4 one
/// for reading alone resident: every command of the second kind specializes the pipeline as the
/// others of it do (one key, one variant), so a command that names no such texture makes no new
/// one. The dropped one's storage element is the class's neutral, its entry and the plain ones'
/// unflagged; the stored one's element is its write shadow's float view, its entry flagged.
#[test]
fn a_heap_write_s_ycbcr_constant_is_its_command_s() {
    let Some(bc) = kernels() else {
        return;
    };
    let key = SpecKey::HeapYCbCrWrites(Stage::Kernel);
    let read = TextureUsage::SHADER_READ;
    let cases: [(&[(PixelFormat, TextureUsage)], u64); 7] = [
        (&[], 0),
        (&[(PixelFormat::RGBA8Unorm, WRITE)], 0),
        (&[(PixelFormat::RGBA16Float, WRITE)], 0),
        (&[(PixelFormat::YCBCR84202P, WRITE)], 0),
        (&[(PixelFormat::YCBCR84442P, read)], 0),
        (&[(PixelFormat::YCBCR84442P, WRITE)], 1),
        (
            &[
                (PixelFormat::RGBA8Unorm, WRITE),
                (PixelFormat::RGB104442P, WRITE),
            ],
            1,
        ),
    ];
    let mut planned = 0;
    for stem in STEMS {
        let dev = device(stem);
        let Some(pipe) = pipeline(&dev, &bc, "heap_ycbcr_write") else {
            continue;
        };
        let mut unnamed = Vec::new();
        for (members, value) in cases {
            let what = format!("{members:?} on {stem}");
            let members: Vec<(TextureFacts, u64)> = (members.iter().zip(1u64..))
                .map(|(&(pf, usage), i)| (facts(pf, usage), 0x0000_0001_0000_0000 | i << 4))
                .collect();
            let plan = pipe
                .plan(&dev, &state(&members).shape())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(given(&pipe, &plan, key), Some(value), "{what}");
            if value == 0 {
                unnamed.push((plan.variant.clone(), plan.specialization.clone()));
            }
            planned += 1;
        }
        assert!(
            unnamed.windows(2).all(|w| w[0] == w[1]),
            "{stem}: a command naming no stored YCbCr texture takes another key"
        );
        // The elements and flags the device-wide heap writes as each texture is made.
        let profile = dev.profile();
        let class = HandleClass::Image(storage_class());
        let layout = vk::ImageLayout::GENERAL;
        let dropped = facts(PixelFormat::YCBCR84202P, WRITE);
        assert_eq!(
            heap_element(profile, &class, &dropped, layout),
            Ok(HeapElement::Neutral),
            "{stem}"
        );
        let stored = facts(PixelFormat::YCBCR84442P, WRITE);
        match heap_element(profile, &class, &stored, layout) {
            Ok(HeapElement::View(v)) => {
                assert_eq!(v.format, vk::Format::R32G32B32A32_SFLOAT, "{stem}")
            }
            other => panic!("{stem}: the stored texture's element {other:?}"),
        }
        for (pf, flagged) in [
            (PixelFormat::YCBCR84442P, true),
            (PixelFormat::YCBCR104442PSRGB, true),
            (PixelFormat::RGB84442P, true),
            (PixelFormat::YCBCR84202P, false),
            (PixelFormat::RGBA8Unorm, false),
        ] {
            let flags = handle_entry_flags(profile, &facts(pf, WRITE));
            assert_eq!(flags & FLAG_YCBCR_WRITES != 0, flagged, "{pf:?}");
        }
    }
    assert!(planned > 0, "no committed profile has a storage heap");
}

/// A stage that reads the heap's storage class (a `read_write` texture's read) states
/// `RuntimeFact::HeapStorageReads`, and its plan refuses a texture of a dropped or shadowed write
/// it reaches there (`YCBCR8_420_2P`, `YCBCR8_444_2P`), `Unsupported { feature: Format, needs:
/// NotImplemented }`, as a slot that reads one is refused; it plans an `RGBA32Float` one.
#[test]
fn a_heap_storage_read_of_a_ycbcr_texture_is_refused_typed() {
    let Some(bc) = kernels() else {
        return;
    };
    let mut planned = 0;
    for stem in STEMS {
        let dev = device(stem);
        let Some(pipe) = pipeline(&dev, &bc, "heap_ycbcr_rw") else {
            continue;
        };
        assert!(
            pipe.dependencies.iter().any(|d| d.fact
                == RuntimeFact::HeapStorageReads {
                    stage: Stage::Kernel,
                    class: storage_class(),
                }),
            "{stem}: {:?}",
            pipe.dependencies
        );
        let usage = TextureUsage::SHADER_READ | WRITE;
        let plan = |pf| pipe.plan(&dev, &state(&[(facts(pf, usage), YCBCR_ID)]).shape());
        for pf in [PixelFormat::YCBCR84202P, PixelFormat::YCBCR84442P] {
            match plan(pf) {
                Err(BindError::Unsupported(Unsupported {
                    feature: Feature::Format,
                    needs: Needs::NotImplemented,
                    ..
                })) => {}
                other => panic!("{stem}: {pf:?}: {other:?}"),
            }
        }
        plan(PixelFormat::RGBA32Float).unwrap_or_else(|e| panic!("{stem}: RGBA32Float: {e}"));
        planned += 1;
    }
    assert!(planned > 0, "no committed profile has a storage heap");
}
