//! Length classes, the `BufferRangeLimit` variant, and the uses of used resources.

use super::*;
use crate::bind::state::{
    length_class, BindState, BoundBuffer, BufferRanges, ResourceClass, StageShape, StageTable,
    UsedResource,
};
use crate::device::profile::BufferAccess;
use crate::pipeline::facts::Mechanism;
use crate::pipeline::interface::BufferKind;
use crate::pipeline::sym::Stage;
use crate::spirv::testing::committed_profiles;

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// The class bounds the padded length from above and is the smallest such power of two.
#[test]
fn a_length_class_bounds_the_padded_length() {
    for length in [
        0u64,
        1,
        15,
        16,
        17,
        255,
        256,
        1 << 27,
        (1 << 27) + 1,
        u64::MAX,
    ] {
        let c = length_class(length);
        let padded = u128::from(length).next_multiple_of(16);
        assert!(padded < 1u128 << c, "{length}: class {c}");
        if padded > 0 {
            assert!(
                padded >= 1u128 << (c - 1),
                "{length}: class {c} is not the smallest"
            );
        }
    }
    assert_eq!(length_class(0), 0);
    // A class proves a range fits exactly when every padded length below `2^class` does.
    let limit = (1u64 << 27) - 1;
    assert!(class_fits(length_class((1 << 27) - 16), limit));
    assert!(!class_fits(length_class(1 << 27), limit));
    assert!(class_fits(
        length_class(u64::from(u32::MAX) - 15),
        u64::from(u32::MAX)
    ));
}

fn planned(index: u32, bound: BoundKind, form: BufferForm) -> PlannedBuffer {
    PlannedBuffer {
        slot: slot(index),
        window: Window::Whole,
        form,
        access: AccessSet::READ,
        binding: None,
        alignment: 16,
        max_range: 0,
        bound,
    }
}

/// On Profile A a large bound buffer under a `BufferRangeLimit` fact needs the variant, in either
/// form (a fixed point); a small one, an unbound slot, or a slot without the fact does not.
/// Profile F (below the runtime's floor, [`PROFILE_F`](crate::pipeline::testing::PROFILE_F))
/// honors the fact at instantiation and never asks for a variant.
#[test]
fn a_large_buffer_needs_the_device_address_variant_on_profile_a_only() {
    let deps = [0, 1, 2, 3].map(|n| RuntimeDependency {
        fact: RuntimeFact::BufferRangeLimit { slot: slot(n) },
        mechanism: Mechanism::Variant,
    });
    let storage = BufferForm::Descriptor;
    let mut seen = [false; 2];
    let mut profiles = committed_profiles().expect("profiles");
    let f = crate::pipeline::testing::device_f(crate::device::DevicePolicy::RESIDENT_TABLES)
        .expect("Profile F");
    profiles.push((
        crate::pipeline::testing::PROFILE_F.to_string(),
        f.profile().clone(),
    ));
    for (stem, p) in profiles {
        let max = u64::from(p.fields().limits.max_storage_buffer_range);
        let big = length_class(max.saturating_add(1));
        let small = length_class(max / 4);
        let buffers = [
            planned(
                0,
                BoundKind::Buffer {
                    device_address: true,
                    length_class: big,
                },
                storage,
            ),
            planned(
                1,
                BoundKind::Buffer {
                    device_address: true,
                    length_class: big,
                },
                BufferForm::Psb,
            ),
            planned(
                2,
                BoundKind::Buffer {
                    device_address: true,
                    length_class: small,
                },
                storage,
            ),
            planned(3, BoundKind::Unbound, storage),
            planned(
                4,
                BoundKind::Buffer {
                    device_address: true,
                    length_class: big,
                },
                storage,
            ),
        ];
        let got = psb_variant(&deps, &buffers, &p);
        match p.fields().buffer_access {
            BufferAccess::Addressed => {
                seen[0] = true;
                assert_eq!(got, [slot(0), slot(1)].into(), "{stem}");
            }
            BufferAccess::Floor => {
                seen[1] = true;
                assert!(got.is_empty(), "{stem}");
            }
        }
    }
    assert_eq!(seen, [true, true], "both profile kinds are checked");
}

/// Used resources reach a plan's `uses` exactly when a function can reach them, with the guest's
/// declared usage, and the shape unions a token listed twice.
#[test]
fn used_resources_are_listed_when_the_pipeline_reaches_them() {
    let mut state = BindState::default();
    let mut buffer = UsedResource::new(ResourceClass::Buffer, 1);
    buffer.usage = metal::ResourceUsage::READ;
    let mut again = buffer;
    again.usage = metal::ResourceUsage::WRITE;
    again.stages = metal::RenderStages::FRAGMENT;
    let mut heap = UsedResource::new(ResourceClass::Heap, 2);
    heap.stages = metal::RenderStages::VERTEX;
    let mut texture = UsedResource::new(
        ResourceClass::Texture(crate::bind::state::TextureFacts::new(
            metal::TextureType::Type2D,
            metal::PixelFormat::RGBA8Unorm,
        )),
        3,
    );
    texture.usage = metal::ResourceUsage::SAMPLE;
    state.used_resources = vec![buffer, heap, texture, again];
    let shape = state.shape();
    assert_eq!(
        shape.used[&1].usage,
        metal::ResourceUsage::READ | metal::ResourceUsage::WRITE
    );
    let compute = |_| vk::PipelineStageFlags2::COMPUTE_SHADER;
    // The kernel may write memory.
    let all = vk::PipelineStageFlags2::COMPUTE_SHADER;
    let tokens = |uses: &[ResourceUse]| -> Vec<(UseTarget, AccessSet, AccessClass)> {
        uses.iter()
            .map(|u| (u.resource, u.access, u.class))
            .collect()
    };
    assert!(used_uses(&shape.used, false, false, compute, all).is_empty());
    assert_eq!(
        tokens(&used_uses(&shape.used, true, false, compute, all)),
        vec![
            (
                UseTarget::Used(1),
                AccessSet::READ_WRITE,
                AccessClass::Storage
            ),
            (
                UseTarget::Used(2),
                AccessSet::READ_WRITE,
                AccessClass::Storage
            ),
        ]
    );
    assert_eq!(
        tokens(&used_uses(&shape.used, false, true, compute, all)),
        vec![
            (
                UseTarget::Used(2),
                AccessSet::READ_WRITE,
                AccessClass::Storage
            ),
            (UseTarget::Used(3), AccessSet::READ, AccessClass::Sampled),
        ]
    );
    // A render plan lists a resource only in the stages it is used in.
    let vertex = vk::PipelineStageFlags2::VERTEX_SHADER;
    let vertex_only = |s: metal::RenderStages| {
        if s.contains(metal::RenderStages::VERTEX) {
            vk::PipelineStageFlags2::VERTEX_SHADER
        } else {
            vk::PipelineStageFlags2::empty()
        }
    };
    assert_eq!(
        tokens(&used_uses(&shape.used, true, true, vertex_only, vertex)),
        vec![(
            UseTarget::Used(2),
            AccessSet::READ_WRITE,
            AccessClass::Storage
        )]
    );
}

/// A used resource the usage lets a function write is written only in the stages whose function
/// may write memory, and read in the others: a draw whose vertex function stores nothing reads a
/// buffer an earlier compute encoder made resident for writing, as its fragment function, which
/// stores, writes it. One only read keeps a single use; one no stage writes is read everywhere.
#[test]
fn used_resources_are_written_only_by_stages_that_write_memory() {
    let mut state = BindState::default();
    let mut written = UsedResource::new(ResourceClass::Buffer, 1);
    written.usage = metal::ResourceUsage::WRITE;
    written.stages = metal::RenderStages::VERTEX.union(metal::RenderStages::FRAGMENT);
    let mut read = UsedResource::new(ResourceClass::Buffer, 2);
    read.usage = metal::ResourceUsage::READ;
    read.stages = written.stages;
    state.used_resources = vec![written, read];
    let shape = state.shape();
    let vertex = vk::PipelineStageFlags2::VERTEX_SHADER;
    let fragment = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    let reach = |s: metal::RenderStages| {
        let mut f = vk::PipelineStageFlags2::empty();
        if s.contains(metal::RenderStages::VERTEX) {
            f |= vertex;
        }
        if s.contains(metal::RenderStages::FRAGMENT) {
            f |= fragment;
        }
        f
    };
    let listed = |writers| -> Vec<(UseTarget, AccessSet, vk::PipelineStageFlags2)> {
        used_uses(&shape.used, true, false, reach, writers)
            .iter()
            .map(|u| (u.resource, u.access, u.stages))
            .collect()
    };
    let write = AccessSet {
        read: false,
        write: true,
    };
    assert_eq!(
        listed(fragment),
        vec![
            (UseTarget::Used(1), write, fragment),
            (UseTarget::Used(1), AccessSet::READ, vertex),
            (UseTarget::Used(2), AccessSet::READ, vertex | fragment),
        ]
    );
    assert_eq!(
        listed(vk::PipelineStageFlags2::empty()),
        vec![
            (UseTarget::Used(1), AccessSet::READ, vertex | fragment),
            (UseTarget::Used(2), AccessSet::READ, vertex | fragment),
        ]
    );
    assert_eq!(
        listed(vertex | fragment),
        vec![
            (UseTarget::Used(1), write, vertex | fragment),
            (UseTarget::Used(2), AccessSet::READ, vertex | fragment),
        ]
    );
}

/// A buffer range limit tells a buffer's length class apart as far as a plan reads it
/// (`BufferRanges::length_class`), under a 4 GiB limit (an RX 7900 XTX's), limits of 128 MiB,
/// 64 KiB and 16 KiB, and none (the default, every class its own): the class a buffer's shape
/// states of each length answers `class_fits` against the limit as the exact class does (all a
/// plan reads of it: `range_variant`), still bounds the padded length from above, and is the exact
/// one under none; and instantiation's kind check (`instantiate::bound_buffer`) takes a binding of
/// each length for the kind a plan of another length stored exactly where the two classes are
/// one, which is only where their exact classes answer the limit alike.
#[test]
fn length_classes_are_told_apart_as_far_as_a_plan_reads_them() {
    let mut lengths = vec![0u64, 1, 15, 16, 17, u64::MAX];
    for bit in 4..64 {
        lengths.extend([(1u64 << bit) - 1, 1 << bit, (1 << bit) + 1]);
    }
    let limits = [
        BufferRanges::default(),
        BufferRanges { storage: u32::MAX },
        BufferRanges { storage: 1 << 27 },
        BufferRanges { storage: 1 << 16 },
        BufferRanges { storage: 1 << 14 },
    ];
    for ranges in limits {
        let fits = |class| class_fits(class, u64::from(ranges.storage));
        let states: Vec<BindState> = (lengths.iter())
            .map(|&length| {
                let mut table = StageTable {
                    buffer_ranges: ranges,
                    ..StageTable::default()
                };
                let bound = BoundBuffer::Buffer {
                    gpu_address: 0x1_0000,
                    length,
                    offset: 0,
                    device_address: Some(0x2_0000),
                    attribute_stride: None,
                };
                table.buffers.insert(0, bound);
                let mut state = BindState::default();
                state.stages.insert(Stage::Kernel, table);
                state
            })
            .collect();
        for (&a, state) in lengths.iter().zip(&states) {
            let (exact, class) = (length_class(a), ranges.length_class(a));
            let what = format!("{ranges:?}, {a} bytes");
            assert_eq!(
                fits(class),
                fits(exact),
                "{what}: class {class}, exact {exact}"
            );
            let padded = u128::from(a).next_multiple_of(16);
            assert!(padded < 1u128 << class, "{what}: class {class}");
            if ranges == BufferRanges::default() {
                assert_eq!(class, exact, "{what}");
            }
            let shape = state.shape();
            let expected = BoundKind::of(shape.stages[&Stage::Kernel].buffers.get(&0));
            for (&b, other) in lengths.iter().zip(&states) {
                let took = crate::bind::instantiate::bound_buffer(other, slot(0), expected).is_ok();
                assert_eq!(
                    took,
                    ranges.length_class(b) == class,
                    "{what}: a binding of {b} bytes for its kind"
                );
                if took {
                    assert_eq!(
                        fits(length_class(b)),
                        fits(exact),
                        "{what}: {b} bytes alike"
                    );
                }
            }
        }
    }
    // Under a 4 GiB limit every length up to 4 GiB is one class, past it each its own.
    let ranges = limits[1];
    assert_eq!(ranges.length_class(1), ranges.length_class((1 << 32) - 16));
    assert_ne!(ranges.length_class(1), ranges.length_class(1 << 32));
    assert_ne!(ranges.length_class(1 << 32), ranges.length_class(1 << 33));
}

/// A buffer bound by shape and state agree on the bound kind, including the length class.
#[test]
fn the_bound_kind_carries_the_length_class() {
    let b = BoundBuffer::Buffer {
        length: 100,
        offset: 4,
        gpu_address: 0,
        device_address: Some(8),
        attribute_stride: None,
    };
    assert_eq!(
        BoundKind::of(Some(&b.shape(BufferRanges::default()))),
        BoundKind::Buffer {
            device_address: true,
            length_class: 7
        }
    );
    assert_eq!(BoundKind::of(None), BoundKind::Unbound);
}

#[test]
fn slots_beyond_the_metal_tables_are_refused() {
    let mut shape = BindShape::default();
    let mut s = StageShape::default();
    s.buffers.insert(
        30,
        BufferShape::Bytes {
            attribute_stride: None,
        },
    );
    shape.stages.insert(Stage::Kernel, s.clone());
    assert_eq!(check_slots(&shape), Ok(()));
    s.buffers.insert(
        31,
        BufferShape::Bytes {
            attribute_stride: None,
        },
    );
    shape.stages.insert(Stage::Kernel, s);
    assert_eq!(
        check_slots(&shape),
        Err(BindError::SlotOutOfRange {
            slot: SlotRef {
                stage: Stage::Kernel,
                class: SlotClass::Buffer,
                index: 31
            }
        })
    );
    let mut s = StageShape::default();
    s.threadgroup_lengths.insert(31, 16);
    shape.stages.insert(Stage::Kernel, s);
    assert!(check_slots(&shape).is_err());
}

/// On every committed profile a storage view's descriptor base is aligned to
/// `max(minStorageBufferOffsetAlignment, 16)`, so whatever the bind offset and window, every byte
/// of the range sits at a descriptor offset congruent to its allocation offset modulo 16: a 16-byte
/// leaf whose Metal address is a multiple of 16 is one `uvec4` element of the view.
#[test]
fn descriptor_bases_are_sixteen_aligned_on_every_profile() {
    let bindings: BTreeMap<_, _> = [(BindingSource::Buffer(slot(0)), (0, 0))].into();
    let shape = BufferShape::Buffer {
        device_address: false,
        length_class: length_class(4096),
        attribute_stride: None,
        offset_alignment: 4,
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let limits = &p.fields().limits;
        let min = limits.min_storage_buffer_offset_alignment;
        let b = BufferSlot {
            access: AccessSet::READ,
            window: Window::Whole,
            form: BufferForm::Descriptor,
            kind: BufferKind::Buffer,
        };
        let pb = plan_buffer((slot(0), &b, Some(&shape)), false, limits, &bindings)
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(pb.alignment, min.max(16), "{stem}");
        for bind_offset in (0..64).step_by(4) {
            for window in [
                Window::Whole,
                Window::Bounded { start: 0, len: 64 },
                Window::Bounded { start: 4, len: 28 },
                Window::Bounded { start: 52, len: 16 },
                Window::Bounded { start: -8, len: 40 },
            ] {
                let (range, residual) = crate::bind::instantiate::descriptor_range(
                    bind_offset,
                    4096,
                    window,
                    pb.alignment,
                    u64::MAX,
                )
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
                assert_eq!(range.offset % 16, 0, "{stem} {bind_offset}");
                // Byte `m` from the bind offset is byte `residual + m` of the range.
                assert_eq!(
                    u64::from(residual) % 16,
                    bind_offset % 16,
                    "{stem} {bind_offset} {window:?}"
                );
            }
        }
    }
}

/// A slot whose function observes residency (`RuntimeFact::SparseTexture`) binds a texture that is
/// not sparse (or nothing) on every profile, and a sparse one where the module has the sparse arm
/// or the tile-map arm; without either a sparse texture is refused, naming
/// `shaderResourceResidency` where the profile lacks it and `NotImplemented` where it has it (an
/// access without a sparse form). A slot the function writes binds a sparse texture (its writes
/// dropped under the specialization), and one it updates atomically refuses it as
/// `NotRepresentable` on every profile.
#[test]
fn a_sparse_texture_needs_the_slots_sparse_arm() {
    use crate::bind::state::TextureFacts;
    let at = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 2,
    };
    let shape = |sparse: Option<bool>| {
        let mut s = BindShape::default();
        let mut stage = StageShape::default();
        if let Some(sparse) = sparse {
            let mut facts =
                TextureFacts::new(metal::TextureType::Type2D, metal::PixelFormat::RGBA8Unorm);
            facts.sparse = sparse;
            stage.textures.insert(2, facts.shape());
        }
        s.stages.insert(Stage::Kernel, stage);
        s
    };
    let arms = |sparse_arm: bool, tile_map: bool| {
        vec![RuntimeDependency {
            fact: RuntimeFact::SparseTexture {
                slot: at,
                sparse_arm,
                tile_map,
            },
            mechanism: if sparse_arm || tile_map {
                Mechanism::Specialization
            } else {
                Mechanism::PlanTimeCheck
            },
        }]
    };
    let deps = |sparse_arm: bool| arms(sparse_arm, false);
    let writes = |atomic: bool| {
        vec![RuntimeDependency {
            fact: RuntimeFact::SparseTextureWrites { slot: at, atomic },
            mechanism: if atomic {
                Mechanism::PlanTimeCheck
            } else {
                Mechanism::Specialization
            },
        }]
    };
    assert_eq!(sparse_texture(&shape(Some(true)), at), 1);
    assert_eq!(sparse_texture(&shape(Some(false)), at), 0);
    assert_eq!(sparse_texture(&shape(None), at), 0);
    let mut seen = [0; 2];
    for (stem, p) in committed_profiles().expect("profiles") {
        for bound in [None, Some(false)] {
            for arm in [false, true] {
                check_sparse_textures(&deps(arm), &p, &shape(bound))
                    .unwrap_or_else(|e| panic!("{stem}: {bound:?} {arm}: {e}"));
            }
        }
        check_sparse_textures(&deps(true), &p, &shape(Some(true)))
            .unwrap_or_else(|e| panic!("{stem}: sparse arm: {e}"));
        // A tile-map arm answers a sparse texture too.
        check_sparse_textures(&arms(false, true), &p, &shape(Some(true)))
            .unwrap_or_else(|e| panic!("{stem}: tile-map arm: {e}"));
        let residency = p
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderResourceResidency);
        let feature = Feature::SparseTexture { slot: at };
        let want = if residency {
            Unsupported::not_implemented(feature)
        } else {
            Unsupported::needs(
                feature,
                Capability::Feature(VkFeature::ShaderResourceResidency),
            )
        };
        match check_sparse_textures(&deps(false), &p, &shape(Some(true))) {
            Err(BindError::Unsupported(u)) => assert_eq!(u, want, "{stem}"),
            other => panic!("{stem}: {other:?}"),
        }
        // A written slot binds a sparse texture, its writes dropped; one updated atomically refuses
        // it, since Metal's answer there is not deterministic.
        for bound in [None, Some(false), Some(true)] {
            check_sparse_textures(&writes(false), &p, &shape(bound))
                .unwrap_or_else(|e| panic!("{stem}: written, {bound:?}: {e}"));
        }
        for bound in [None, Some(false)] {
            check_sparse_textures(&writes(true), &p, &shape(bound))
                .unwrap_or_else(|e| panic!("{stem}: atomic, {bound:?}: {e}"));
        }
        let mut atomic = deps(false);
        atomic.extend(writes(true));
        for deps in [writes(true), atomic] {
            match check_sparse_textures(&deps, &p, &shape(Some(true))) {
                Err(BindError::Unsupported(u)) => {
                    let feature = Feature::SparseTexture { slot: at };
                    assert_eq!(u, Unsupported::not_representable(feature), "{stem}")
                }
                other => panic!("{stem}: atomic: {other:?}"),
            }
        }
        seen[usize::from(residency)] += 1;
    }
    assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
}

/// A constexpr-sampled slot enters `VariantKey::address_maps` only where the bound texture's
/// address-mode map changes a mode the slot's constexpr samplers state, with the map restricted
/// to those modes: a draw of an ordinary texture, or of a mapped one whose map keeps those modes,
/// keeps its key, and textures whose maps agree on them share a variant. A slot no constexpr
/// sampler samples takes no map, whatever its texture's.
#[test]
fn only_a_map_that_moves_a_constexpr_mode_enters_the_variant() {
    use crate::bind::state::TextureFacts;
    use crate::metal::sampler::AddressModeMap;
    use crate::metal::SamplerAddressMode as M;
    let texture = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    };
    let identity = AddressModeMap::IDENTITY;
    let edges = identity.with(M::Repeat, M::ClampToEdge);
    // Every mode to the edge, as a two-plane 4:2:0 texture reads.
    let all_edges = [
        M::MirrorClampToEdge,
        M::Repeat,
        M::MirrorRepeat,
        M::ClampToZero,
        M::ClampToBorderColor,
    ]
    .into_iter()
    .fold(identity, |map, mode| map.with(mode, M::ClampToEdge));
    let shape = |maps: [AddressModeMap; 2]| {
        let mut s = BindShape::default();
        let mut stage = StageShape::default();
        for (index, map) in (0..).zip(maps) {
            let mut facts =
                TextureFacts::new(metal::TextureType::Type2D, metal::PixelFormat::RGBA8Unorm);
            facts.address_modes = map;
            stage.textures.insert(index, facts.shape());
        }
        s.stages.insert(Stage::Kernel, stage);
        s
    };
    // Slot 0 is constexpr-sampled, by a repeat sampler and by one of `other` modes; slot 1 is
    // not.
    let deps = |other: Option<M>| {
        let sampled = |mode| RuntimeDependency {
            fact: RuntimeFact::ConstexprSampled {
                slot: texture(0),
                modes: [mode; 3],
            },
            mechanism: Mechanism::Variant,
        };
        let mut deps = vec![sampled(M::Repeat)];
        deps.extend(other.map(sampled));
        deps
    };
    let repeat = deps(None);
    assert!(address_map_variant(&repeat, &shape([identity, identity])).is_empty());
    assert!(address_map_variant(&repeat, &shape([identity, edges])).is_empty());
    assert!(address_map_variant(&[], &shape([edges, edges])).is_empty());
    assert_eq!(
        address_map_variant(&repeat, &shape([edges, identity])),
        BTreeMap::from([(texture(0), edges)])
    );
    // The map restricted to the stated modes: one that sends every mode to the edge states the
    // same variant as one that sends repeat there alone.
    assert_eq!(
        address_map_variant(&repeat, &shape([all_edges, identity])),
        BTreeMap::from([(texture(0), edges)])
    );
    // A map that keeps every stated mode adds nothing: a clamp-to-edge sampler of a texture whose
    // every mode reads the edge samples it as before.
    let edge = vec![RuntimeDependency {
        fact: RuntimeFact::ConstexprSampled {
            slot: texture(0),
            modes: [M::ClampToEdge; 3],
        },
        mechanism: Mechanism::Variant,
    }];
    assert!(address_map_variant(&edge, &shape([all_edges, identity])).is_empty());
    // Two samplers of the slot: the map on the modes of both.
    assert_eq!(
        address_map_variant(&deps(Some(M::ClampToZero)), &shape([all_edges, identity])),
        BTreeMap::from([(texture(0), edges.with(M::ClampToZero, M::ClampToEdge))])
    );
    // A slot with no texture bound takes no map.
    assert!(address_map_variant(&repeat, &BindShape::default()).is_empty());
}

/// A slot that states the in-place form enters `VariantKey::ycbcr_in_place` only where the bound
/// texture's planes are read in place and each runtime sampler the slot samples through states a
/// filter the form reads: one filter, no anisotropy, cubic filter, comparison or other reduction.
#[test]
fn the_in_place_variant_takes_a_slot_its_texture_and_samplers_read() {
    use crate::bind::state::{BoundSampler, SamplerDescriptor, TextureFacts};
    use crate::bind::ycbcr_convert::{YCbCrCodes, YCbCrMatrix, YCbCrPlanes, YCbCrRange};
    use crate::bind::YCbCrInPlace;
    use metal::{CompareFunction, SamplerMinMagFilter as F, SamplerReductionMode};
    let texture = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    };
    let pf = metal::PixelFormat::YCBCR84202P;
    let codes = metal::format::ycbcr_format(pf)
        .and_then(YCbCrCodes::of)
        .expect("codes");
    let planes = YCbCrPlanes::new(
        [8, 4],
        [0, 16],
        [64, 16],
        (YCbCrRange::Video, YCbCrMatrix::Bt709),
        codes,
    );
    // Slot 0 and 1 bind textures whose planes are read in place, slot 2 one whose are not.
    let shape = |sampler: SamplerDescriptor| {
        let mut stage = StageShape::default();
        for (index, in_place) in [(0, true), (1, true), (2, false)] {
            let mut facts = TextureFacts::new(metal::TextureType::Type2D, pf);
            if in_place {
                facts.ycbcr_in_place = YCbCrInPlace::of(&planes);
            }
            stage.textures.insert(index, facts.shape());
        }
        stage.samplers.insert(3, BoundSampler::new(sampler).shape());
        let mut s = BindShape::default();
        s.stages.insert(Stage::Kernel, stage);
        s
    };
    let fact = |slot, samplers| RuntimeDependency {
        fact: RuntimeFact::YCbCrInPlace {
            slot,
            samplers,
            device: true,
        },
        mechanism: Mechanism::Variant,
    };
    // Slot 0 samples through constexpr samplers alone, slot 1 also through `[[sampler(3)]]`.
    let deps = [
        fact(texture(0), 0),
        fact(texture(1), 1 << 3),
        fact(texture(2), 0),
    ];
    let linear = SamplerDescriptor {
        min_filter: F::Linear,
        mag_filter: F::Linear,
        ..SamplerDescriptor::new()
    };
    let both = std::collections::BTreeSet::from([texture(0), texture(1)]);
    let s = shape(linear);
    let in_place = ycbcr_in_place_variant(&deps, &s);
    assert_eq!(in_place, both);
    assert_eq!(
        ycbcr_in_place_variant(&deps, &shape(SamplerDescriptor::new())),
        both
    );
    // A sampler the form does not read keeps slot 1 from it, not slot 0.
    let unread = [
        SamplerDescriptor {
            min_filter: F::Linear,
            ..SamplerDescriptor::new()
        },
        SamplerDescriptor {
            max_anisotropy: 4,
            ..linear
        },
        SamplerDescriptor {
            min_filter: F::Cubic,
            mag_filter: F::Cubic,
            ..SamplerDescriptor::new()
        },
        SamplerDescriptor {
            compare_function: CompareFunction::Less,
            ..linear
        },
        SamplerDescriptor {
            reduction_mode: SamplerReductionMode::Minimum,
            ..linear
        },
    ];
    for sampler in unread {
        assert_eq!(
            ycbcr_in_place_variant(&deps, &shape(sampler)),
            std::collections::BTreeSet::from([texture(0)]),
            "{sampler:?}"
        );
    }
    // A stage that states no form takes none, whatever its textures.
    assert!(ycbcr_in_place_variant(&[], &s).is_empty());
    // No texture's planes are held as an image the device samples: no slot samples one.
    assert!(ycbcr_planes_sampled_variant(&deps, &s, &in_place).is_empty());
}

/// An in-place slot enters `VariantKey::ycbcr_planes_sampled` only where the bound texture's
/// planes are held as an image the device's conversion samples, the function's every sample is one
/// that sampler takes (the fact's `device`), and each runtime sampler it samples through addresses
/// no border on the plane axes.
#[test]
fn the_planes_image_variant_takes_a_slot_the_device_s_sampler_reads() {
    use crate::bind::state::{BoundSampler, SamplerDescriptor, TextureFacts};
    use crate::bind::ycbcr_convert::{YCbCrCodes, YCbCrMatrix, YCbCrPlanes, YCbCrRange};
    use crate::bind::YCbCrInPlace;
    use metal::SamplerAddressMode as M;
    let texture = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    };
    let pf = metal::PixelFormat::YCBCR84202P;
    let codes = metal::format::ycbcr_format(pf)
        .and_then(YCbCrCodes::of)
        .expect("codes");
    let planes = YCbCrPlanes::new(
        [8, 4],
        [0, 16],
        [64, 16],
        (YCbCrRange::Full, YCbCrMatrix::Bt601),
        codes,
    );
    let words = YCbCrInPlace::of(&planes).expect("in place");
    // Slots 0 to 2 bind textures whose planes are held as an image, slot 3 one whose are not.
    let shape = |sampler: SamplerDescriptor| {
        let mut stage = StageShape::default();
        for index in 0..4 {
            let mut facts = TextureFacts::new(metal::TextureType::Type2D, pf);
            facts.ycbcr_in_place = Some(match index {
                3 => words,
                _ => words.with_planes_image(),
            });
            stage.textures.insert(index, facts.shape());
        }
        stage.samplers.insert(3, BoundSampler::new(sampler).shape());
        let mut s = BindShape::default();
        s.stages.insert(Stage::Kernel, stage);
        s
    };
    let fact = |slot, samplers, device| RuntimeDependency {
        fact: RuntimeFact::YCbCrInPlace {
            slot,
            samplers,
            device,
        },
        mechanism: Mechanism::Variant,
    };
    // Slot 0 samples through constexpr samplers the device takes, slot 1 also through
    // `[[sampler(3)]]`, slot 2 through a constexpr sampler it does not take.
    let deps = [
        fact(texture(0), 0, true),
        fact(texture(1), 1 << 3, true),
        fact(texture(2), 0, false),
        fact(texture(3), 0, true),
    ];
    let edge = SamplerDescriptor::new();
    let s = shape(edge);
    let in_place = ycbcr_in_place_variant(&deps, &s);
    assert_eq!(in_place.len(), 4);
    let sampler = words.sampler();
    assert_eq!(
        ycbcr_planes_sampled_variant(&deps, &s, &in_place),
        BTreeMap::from([(texture(0), sampler), (texture(1), sampler)])
    );
    // A runtime sampler that addresses a border on a plane axis keeps slot 1 from it; one on `r`
    // does not.
    for (mode, axis, taken) in [
        (M::ClampToZero, 0, false),
        (M::ClampToBorderColor, 1, false),
        (M::Repeat, 0, true),
        (M::MirrorClampToEdge, 1, true),
        (M::ClampToBorderColor, 2, true),
    ] {
        let mut sampler_state = edge;
        match axis {
            0 => sampler_state.s_address_mode = mode,
            1 => sampler_state.t_address_mode = mode,
            _ => sampler_state.r_address_mode = mode,
        }
        let s = shape(sampler_state);
        let in_place = ycbcr_in_place_variant(&deps, &s);
        let got = ycbcr_planes_sampled_variant(&deps, &s, &in_place);
        assert_eq!(
            got.contains_key(&texture(1)),
            taken,
            "{mode:?} on axis {axis}"
        );
        assert!(got.contains_key(&texture(0)), "{mode:?} on axis {axis}");
    }
    // A slot outside the in-place variant takes no image.
    let none = std::collections::BTreeSet::new();
    assert!(ycbcr_planes_sampled_variant(&deps, &s, &none).is_empty());
}
