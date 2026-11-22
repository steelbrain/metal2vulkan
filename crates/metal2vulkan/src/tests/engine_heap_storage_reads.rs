//! A read of an argument buffer's `read_write` texture on a device that reads storage without a
//! format by the formats' own bits alone (`DeviceProfile::storage_reads_by_format_bits`: ANV before
//! Gfx12.5, the Iris Xe of `intel-irisxe-rplp-anv`), whose `shaderStorageImageReadWithoutFormat`
//! is false.
//!
//! Vulkan 1.3 and `VK_KHR_format_feature_flags2` enable the SPIR-V capability
//! `StorageImageReadWithoutFormat`, and a read of a view without a format is valid where the view's
//! format has `STORAGE_READ_WITHOUT_FORMAT_BIT` (`VUID-vkCmdDispatch-OpTypeImage-07028`). The heap's
//! storage class is declared without a format, so its read compiles there (`lower::texture::slot`,
//! `declare_heap`); the binder holds a member whose view lacks the bit for its writes, its entry
//! flagged `FLAG_STORAGE_UNREADABLE`, and a read through its handle reads the null element. Metal
//! defines the read of its read-write tier formats, so a function that reads the class where one
//! of their views lacks the bit is refused instead (`bind::heap::tier_read_gaps`). No GPU is used.

use std::collections::BTreeMap;
use std::path::Path;

use crate::bind::heap::{storage_unreadable, tier_read_gaps};
use crate::bind::{
    handle_entry, BindState, ResourceClass, TextureFacts, UsedResource, WriteSource,
};
use crate::budget::Budget;
use crate::device::metal::read_write_tier_formats;
use crate::device::requirements::{FormatUsage, SpirvCapability, VkFeature};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::Library;
use crate::memory::handles::{Lookup, FLAG_STORAGE_UNREADABLE};
use crate::metal::{PixelFormat, TextureType, TextureUsage};
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

/// Kernel `heap_rw_read` reads the `texture2d<float, read_write>` argument buffer 0 holds at
/// `(gid, 0)` into buffer 1's `float4` at its index.
const KERNEL: &str = r#"%struct.T = type { ptr addrspace(1) }
define void @heap_rw_read(ptr addrspace(2) %r, ptr addrspace(1) %dst, i32 %gid) {
  %pt = getelementptr inbounds %struct.T, ptr addrspace(2) %r, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %dst, i32 %gid
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %smp = call ptr addrspace(2) @air.get_read_sampler()
  %x = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 3)
  %xv = extractvalue { <4 x float>, i8 } %x, 0
  store <4 x float> %xv, ptr addrspace(1) %p, align 16
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_rw_read, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"T", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, read_write>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"t"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"dst"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

const ENTRY: &str = "heap_rw_read";

/// The committed profile of a device that reads storage without a format by the formats' bits
/// alone: the Iris Xe (Raptor Lake-P) under Mesa 26.2.6.
const IRIS: &str = "intel-irisxe-rplp-anv";

/// The resource IDs of the members: a readable texture's, and a flagged one's.
const READABLE_ID: u64 = 0x0000_0001_0000_0010;
const FLAGGED_ID: u64 = 0x0000_0001_0000_0020;

/// The description of committed profile `stem`.
fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

/// The device of committed profile `stem`.
fn device(stem: &str) -> Device {
    Device::from_queries(description(stem), DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// [`KERNEL`], assembled; `None` where `llvm-as` is missing (the test then skips visibly).
fn kernel() -> Option<Vec<u8>> {
    assemble_text("heap_storage_reads", &format!("{HEADER}{KERNEL}"))
}

/// [`ENTRY`] of `bc`, compiled on `dev`.
fn pipeline(dev: &Device, bc: &[u8]) -> Result<ComputePipeline, Error> {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(ENTRY).expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    dev.compile_compute_pipeline(&d, &Budget::standard())
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

/// A 2D `read_write` texture of `pf`.
fn read_write(pf: PixelFormat) -> TextureFacts {
    let mut f = TextureFacts::new(TextureType::Type2D, pf);
    f.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
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

/// The non-tier formats whose storage view the Iris Xe cannot read without a format (Mesa's isl
/// table: no typed reads of `R10G10B10A2`, `B10G10R10A2`, `R11G11B10_FLOAT` or `B8G8R8A8` below
/// Gfx12.5; `BGR10A2Unorm` through its `A2B10G10R10` twin as well).
const IRIS_UNREADABLE: [PixelFormat; 5] = [
    PixelFormat::RGB10A2Unorm,
    PixelFormat::RGB10A2Uint,
    PixelFormat::BGR10A2Unorm,
    PixelFormat::RG11B10Float,
    PixelFormat::BGRA8Unorm,
];

/// The heap's storage read compiles where the formats' own bits grant it: on the Iris Xe, without
/// `shaderStorageImageReadWithoutFormat`, the pipeline requires the SPIR-V capability (which
/// Vulkan 1.3 enables) and not the feature, and its handle-table search carries the unreadable
/// flag; on lavapipe, which has the feature, it carries none. The same device as a Vulkan 1.2
/// context without `VK_KHR_format_feature_flags2` sees, has neither route, and the read is refused
/// naming the feature, as before.
#[test]
fn a_heap_storage_read_compiles_where_the_formats_bits_grant_it() {
    let Some(bc) = kernel() else {
        return;
    };
    let iris = device(IRIS);
    assert!(iris.profile().storage_reads_by_format_bits());
    let pipe = pipeline(&iris, &bc).unwrap_or_else(|e| panic!("{IRIS}: {e}"));
    let capability = SpirvCapability::of(crate::spv::Capability::StorageImageReadWithoutFormat);
    assert!(pipe.requirements.spirv_capabilities.contains(&capability));
    assert!(!pipe
        .requirements
        .features
        .contains(&VkFeature::ShaderStorageImageReadWithoutFormat));
    let (plan, _) = typed_module(&bc, ENTRY, iris.profile()).expect("module");
    assert!(plan.families.access.heap_unreadable);
    assert!(matches!(
        plan.families.access.lookup(),
        Lookup::Flagged {
            unreadable: true,
            ..
        }
    ));

    let lavapipe = device("lavapipe");
    assert!(!lavapipe.profile().storage_reads_by_format_bits());
    pipeline(&lavapipe, &bc).expect("lavapipe");
    let (plan, _) = typed_module(&bc, ENTRY, lavapipe.profile()).expect("module");
    assert!(!plan.families.access.heap_unreadable);

    let old = crate::pipeline::testing::description_at_1_2(IRIS, false).expect("description");
    let old = Device::from_queries(old, DevicePolicy::RESIDENT_TABLES).expect("device");
    assert!(!old.profile().storage_reads_by_format_bits());
    match pipeline(&old, &bc) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::ArgumentBufferTextures);
            let Needs::AnyOf(ways) = &u.needs else {
                panic!("{:?}", u.needs);
            };
            assert!(
                ways.iter().any(|way| way.contains(&Capability::Feature(
                    VkFeature::ShaderStorageImageReadWithoutFormat
                ))),
                "{ways:?}"
            );
        }
        other => panic!("a Vulkan 1.2 context: {:?}", other.map(|_| ())),
    }
}

/// Over the CPU interpreter, the Iris Xe's module reads a member through its handle, and reads
/// the null element (zero) through a handle whose entry carries `FLAG_STORAGE_UNREADABLE`, though
/// the element the entry names holds texels: the guard, not the image, answers zero.
#[test]
fn a_flagged_member_reads_the_null_element() {
    let Some(bc) = kernel() else {
        return;
    };
    let iris = device(IRIS);
    let (plan, m) = typed_module(&bc, ENTRY, iris.profile()).expect("module");
    let class = (plan.families.texture.heap.values())
        .find_map(|h| h.class)
        .expect("a heap class");
    assert_eq!(class, storage_class());
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Heap(HandleClass::Image(class)),
        index: 0,
    };
    let fl = |v: f32| v.to_bits();
    let texels = |base: f32| -> Vec<[u32; 4]> {
        (0..4)
            .map(|i| {
                let v = base + i as f32;
                [fl(v), fl(v + 0.25), fl(v + 0.5), fl(1.0)]
            })
            .collect()
    };
    let mut images: Vec<Image> = [(0, 1), (1, 4), (2, 4)]
        .into_iter()
        .map(|(e, width)| {
            let mut img = Image::new(slot, [width, 1, 1], 1, 1, 1).at_element(e);
            img.robust = Some([0, 0, 0, fl(1.0)]);
            img
        })
        .collect();
    images[1].texels = texels(1.0);
    images[2].texels = texels(10.0);
    // The table: the null entry, the readable member at element 1, the flagged one at 2.
    let table: Vec<u8> = [
        handle_entry(0, 0, 0),
        handle_entry(READABLE_ID, 1, 0),
        handle_entry(FLAGGED_ID, 2, FLAG_STORAGE_UNREADABLE),
    ]
    .concat();
    let buffer = |kind, index, bytes| Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind,
            index,
        }),
        address: None,
        bytes,
    };
    for (id, want) in [(READABLE_ID, texels(1.0)), (FLAGGED_ID, vec![[0; 4]; 4])] {
        let mut spec = BTreeMap::new();
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            spec.insert(SpecKey::LocalSize(axis), 1);
        }
        spec.insert(SpecKey::FlatLocalSize, 1);
        let res = Resources {
            buffers: vec![
                buffer(ResourceKind::Buffer, 0, id.to_le_bytes().to_vec()),
                buffer(ResourceKind::Buffer, 1, vec![0xab; 64]),
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
        let (out, _) = run_with_images(&m, &cfg, res, images.clone())
            .unwrap_or_else(|e| panic!("{id:#x}: {e:?}"));
        let got: Vec<[u32; 4]> = out.buffers[1]
            .chunks(16)
            .map(|t| {
                let w = |i: usize| u32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
                [w(0), w(1), w(2), w(3)]
            })
            .collect();
        assert_eq!(got, want, "{id:#x}");
    }
}

/// On every committed profile that reads storage without a format by the formats' bits alone,
/// where a view of one of Metal's read-write tier formats on the profile cannot be read so
/// (flagged), a function reading its storage class is refused for it ([`tier_read_gaps`]): no tier
/// format reads zero. On the Iris Xe none is flagged; the non-tier formats its view cannot read
/// are flagged; on lavapipe, which has the feature, nothing is flagged.
#[test]
fn only_a_view_the_device_cannot_read_is_flagged() {
    let mut bits_alone = 0;
    for (stem, profile) in crate::spirv::testing::committed_profiles().expect("profiles") {
        let profile = &profile;
        if !profile.storage_reads_by_format_bits() {
            continue;
        }
        bits_alone += 1;
        let tiers = read_write_tier_formats(profile);
        for &pf in tiers.iter().flat_map(|tier| tier.iter()) {
            let facts = read_write(pf);
            let class = ImageHandleClass {
                sampled_type: crate::metal::format::sampled_type(pf).expect("type"),
                ..storage_class()
            };
            let flagged = storage_unreadable(profile, &facts.shape());
            let gaps = tier_read_gaps(profile, &class);
            assert!(!flagged || !gaps.is_empty(), "{stem}: {pf:?} reads zero");
            if stem == IRIS {
                assert!(!flagged && gaps.is_empty(), "{stem}: {pf:?}: {gaps:?}");
            }
        }
    }
    assert!(
        bits_alone > 0,
        "no committed profile reads by the bits alone"
    );

    let iris = device(IRIS);
    let lavapipe = device("lavapipe");
    for pf in IRIS_UNREADABLE {
        let shape = read_write(pf).shape();
        assert!(storage_unreadable(iris.profile(), &shape), "{pf:?}");
        assert!(!storage_unreadable(lavapipe.profile(), &shape), "{pf:?}");
    }
}

/// A plan on the Iris Xe holds an `RGBA16Float` member and an `RG11B10Float` one in the storage
/// class and flags only the second's table entry.
#[test]
fn a_plan_flags_its_unreadable_members() {
    let Some(bc) = kernel() else {
        return;
    };
    let iris = device(IRIS);
    let pipe = pipeline(&iris, &bc).expect("pipeline");
    let state = state(&[
        (read_write(PixelFormat::RGBA16Float), READABLE_ID),
        (read_write(PixelFormat::RG11B10Float), FLAGGED_ID),
    ]);
    let bplan = pipe.plan(&iris, &state.shape()).expect("plan");
    let views: Vec<(u32, vk::Format)> = (bplan.descriptors.iter())
        .filter_map(|d| match &d.source {
            WriteSource::HeapEntry { element, view, .. } => Some((*element, view.format)),
            _ => None,
        })
        .collect();
    assert_eq!(
        views,
        vec![
            (1, vk::Format::R16G16B16A16_SFLOAT),
            (2, vk::Format::B10G11R11_UFLOAT_PACK32)
        ]
    );
    let dispatch = crate::bind::Dispatch::Threadgroups {
        groups: [1, 1, 1],
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
    let table = &(cmds.descriptor_bytes.iter())
        .find(|b| (b.set, b.binding) == (set, binding))
        .expect("table bytes")
        .bytes;
    let flags: Vec<(u64, u32)> = table
        .chunks(16)
        .map(|e| {
            let word = |w: usize| u32::from_le_bytes(e[4 * w..4 * w + 4].try_into().unwrap());
            (
                u64::from(word(0)) | u64::from(word(1)) << 32,
                word(3) & FLAG_STORAGE_UNREADABLE,
            )
        })
        .filter(|(id, _)| *id != 0)
        .collect();
    assert_eq!(
        flags,
        vec![(READABLE_ID, 0), (FLAGGED_ID, FLAG_STORAGE_UNREADABLE)]
    );
}

/// A function that reads a storage class one of Metal's read-write tier formats is held in through
/// a view the device cannot read without a format is refused, naming the view's bit, and the member
/// is flagged unreadable: on the Iris Xe with `R16G16B16A16_SFLOAT`'s `STORAGE_READ_WITHOUT_FORMAT`
/// bit cleared (a synthetic profile), the kernel is `Unsupported` naming that bit alone (the
/// class's neutral, `R32_SFLOAT`, keeps its own), and an `RGBA16Float` member is flagged. The
/// committed profile compiles the kernel and leaves the member unflagged.
#[test]
fn a_tier_format_without_its_read_bit_is_refused() {
    let Some(bc) = kernel() else {
        return;
    };
    let shape = read_write(PixelFormat::RGBA16Float).shape();
    let iris = device(IRIS);
    assert!(!storage_unreadable(iris.profile(), &shape));
    pipeline(&iris, &bc).expect("the committed profile");
    let mut desc = description(IRIS);
    let format = vk::Format::R16G16B16A16_SFLOAT;
    let props = desc.formats.formats.get_mut(&format).expect("RGBA16F");
    props.optimal_tiling = props
        .optimal_tiling
        .difference(vk::FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT);
    let cleared = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
    assert!(cleared.profile().storage_reads_by_format_bits());
    let bit = Capability::Format {
        format,
        usage: FormatUsage::StorageImageReadWithoutFormat,
    };
    assert_eq!(
        tier_read_gaps(cleared.profile(), &storage_class()),
        vec![bit.clone()]
    );
    match pipeline(&cleared, &bc) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::ArgumentBufferTextures);
            assert_eq!(u.needs, Needs::AnyOf(vec![vec![bit]]));
        }
        other => panic!("the kernel: {:?}", other.map(|_| ())),
    }
    assert!(storage_unreadable(cleared.profile(), &shape));
}
