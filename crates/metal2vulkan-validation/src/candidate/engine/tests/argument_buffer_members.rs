//! Argument-buffer textures and samplers in the engine executor: a texture is a used texture whose
//! resource id its owner holds at its field, a member of the plan's heap that the kernel's handle
//! table finds; a sampler is an argument sampler whose id its owner holds, an element of the plan's
//! sampler arrays that the kernel's sampler table finds. No device is opened.

use super::{b64, device, kernel_case, map_kernel};
use crate::candidate::engine::argument_buffers::{
    field_words, owner_bytes, resource_id, sampler_id,
};
use crate::candidate::engine::resources::argument_sampler_descriptor;
use crate::candidate::engine::{
    bind_state, compute_shape, require_case_families, sampler_ids, texture_tokens, variants,
    BufferAddress, ComputeOutput,
};
use crate::case::{
    ArgumentBufferBufferResource, ArgumentBufferSamplerResource, ArgumentBufferTextureResource,
    AuthoredCase, BufferResource, OutputSelection, ResourceRole, SamplerAddressMode, SamplerFilter,
    SamplerMipFilter, TextureFormat, TextureType,
};
use crate::check::admission::NotSupported;
use metal2vulkan::bind::{BufferRanges, HeapSource, NeutralResource, ResourceClass, WriteSource};
use metal2vulkan::device::DevicePolicy;
use metal2vulkan::library::Library;
use metal2vulkan::metal::{
    PixelFormat, RenderStages, ResourceUsage, SamplerMinMagFilter, TextureType as MetalType,
};
use metal2vulkan::pipeline::FunctionDescriptor;
use metal2vulkan::Budget;
use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
use std::collections::BTreeMap;

/// An argument-buffer texture of `format` and `dimensions` at `field_offset` of buffer 0, holding
/// `bytes` as `role`.
fn member(
    field_offset: u32,
    format: TextureFormat,
    dimensions: [u32; 3],
    role: ResourceRole,
    bytes: &[u8],
) -> ArgumentBufferTextureResource {
    let (bytes_b64, initial_bytes_b64) = match role {
        ResourceRole::Input => (Some(b64(bytes)), None),
        _ => (None, Some(b64(bytes))),
    };
    ArgumentBufferTextureResource {
        buffer_binding: 0,
        field_offset,
        role,
        texture_type: TextureType::D2,
        format,
        dimensions,
        sample_count: 1,
        bytes_b64,
        initial_bytes_b64,
    }
}

fn gap(case: &AuthoredCase, expected: &str) {
    match map_kernel(case) {
        Err(NotSupported::NotImplemented(what)) => assert!(what.contains(expected), "{what}"),
        other => panic!("{other:?}"),
    }
}

/// Every member of buffer 0's argument buffer maps into its owner: the owner (4 authored bytes)
/// grows to cover each 8-byte field; each texture is a used texture (its token after the
/// argument-buffer buffers', its resource id nonzero and its own), resident for reading and
/// writing like the oracle's `useResource`; the owner is created holding the buffer's address and
/// each texture's resource id at their fields; a texture's region is an output; and an owner that
/// is not bound, a key authored twice, a texture buffer, a field named twice, and overlapping
/// fields are refused.
#[test]
fn argument_buffer_members_are_written_into_their_owner() {
    let mut case = kernel_case();
    case.argument_buffer_buffers
        .push(ArgumentBufferBufferResource {
            buffer_binding: 0,
            field_offset: 8,
            role: ResourceRole::Input,
            bytes_b64: Some(b64(&[5; 12])),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        });
    case.argument_buffer_textures.extend([
        member(
            16,
            TextureFormat::Rgba8Unorm,
            [2, 1, 1],
            ResourceRole::Input,
            &[1, 2, 3, 4, 5, 6, 7, 8],
        ),
        member(
            24,
            TextureFormat::R32Uint,
            [1, 1, 1],
            ResourceRole::Output,
            &[0xab; 4],
        ),
    ]);
    case.output = OutputSelection::ArgumentBufferTexture {
        buffer_binding: 0,
        field_offset: 24,
        origin: [0, 0, 0],
        dimensions: [1, 1, 1],
    };
    assert_eq!(require_case_families(&case), Ok(()));
    let mapped = map_kernel(&case).unwrap();
    assert_eq!(mapped.buffers[&0].len(), 32);
    assert_eq!(&mapped.buffers[&0][..4], &[0xab; 4]);
    assert_eq!(
        mapped
            .argument_buffer_textures
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [(0, 16), (0, 24)]
    );
    let sampled = &mapped.argument_buffer_textures[&(0, 16)];
    assert_eq!(
        (
            sampled.texture_type,
            sampled.format,
            sampled.dimensions,
            &sampled.bytes[..]
        ),
        (
            TextureType::D2,
            TextureFormat::Rgba8Unorm,
            [2, 1, 1],
            &[1, 2, 3, 4, 5, 6, 7, 8][..]
        )
    );
    assert_eq!(
        mapped.argument_buffer_buffers,
        BTreeMap::from([((0, 8), vec![5; 12])])
    );
    assert_eq!(
        mapped.output,
        ComputeOutput::ArgumentBufferTexture {
            owner: 0,
            field_offset: 24,
            origin: [0, 0, 0],
            dimensions: [1, 1, 1],
        }
    );

    // The bind state: the buffer is token 0, the textures follow it with their resource ids.
    let tokens = texture_tokens(&mapped).unwrap();
    assert_eq!(tokens, BTreeMap::from([((0, 16), 1), ((0, 24), 2)]));
    let host = |guest| BufferAddress {
        guest,
        host: Some(guest),
    };
    let addresses = BTreeMap::from([(0, host(0x1000))]);
    let embedded = BTreeMap::from([((0, 8), host(0xdead_0000))]);
    let state = bind_state(&mapped, &addresses, &embedded, BufferRanges::default()).unwrap();
    let [buffer, first, second] = state.used_resources.as_slice() else {
        panic!("{:?}", state.used_resources);
    };
    assert_eq!((buffer.class, buffer.token), (ResourceClass::Buffer, 0));
    for (used, token, format) in [
        (first, 1, PixelFormat::RGBA8Unorm),
        (second, 2, PixelFormat::R32Uint),
    ] {
        let ResourceClass::Texture(facts) = used.class else {
            panic!("{used:?}");
        };
        assert_eq!(
            (facts.texture_type, facts.pixel_format),
            (MetalType::Type2D, format)
        );
        assert_eq!(used.token, token);
        assert_eq!(used.resource_id, Some(resource_id(token)));
        assert_eq!(used.usage, ResourceUsage::READ | ResourceUsage::WRITE);
        assert_eq!(used.stages, RenderStages::empty());
        assert_eq!((used.gpu_address, used.device_address), (None, None));
    }
    for addressed in [false, true] {
        let shape = compute_shape(&mapped, addressed, BufferRanges::default()).unwrap();
        assert_eq!(shape.used.len(), 3);
        if addressed {
            assert_eq!(state.shape(), shape);
        }
    }

    // The owner holds, at each field, the word Metal's argument encoder writes there.
    let fields = field_words(&embedded, &tokens, &BTreeMap::new()).unwrap();
    let owner = owner_bytes(&mapped.buffers, 0, &fields).unwrap();
    let word = |at: usize| u64::from_le_bytes(owner[at..at + 8].try_into().unwrap());
    assert_eq!(&owner[..8], &[0xab, 0xab, 0xab, 0xab, 0, 0, 0, 0]);
    assert_eq!(
        [word(8), word(16), word(24)],
        [0xdead_0000, resource_id(1), resource_id(2)]
    );
    // Ids are nonzero (a handle table never enters id 0) and distinct.
    let ids = (0..64)
        .map(resource_id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ids.len(), 64);
    assert!(!ids.contains(&0));
    // A field named by two members, overlapping fields, and a field past the owner are refused.
    assert!(field_words(&embedded, &BTreeMap::from([((0, 8), 1)]), &BTreeMap::new(),).is_err());
    let overlapping = BTreeMap::from([((0, 8), 1), ((0, 12), 2)]);
    assert!(owner_bytes(&mapped.buffers, 0, &overlapping).is_err());
    assert!(owner_bytes(&mapped.buffers, 0, &BTreeMap::from([((0, 32), 1)])).is_err());

    let mut orphan = case.clone();
    orphan.argument_buffer_textures[0].buffer_binding = 3;
    gap(&orphan, "without its owner buffer");
    let mut twice = case.clone();
    twice.argument_buffer_textures[1].field_offset = 16;
    gap(&twice, "bound twice");
    let mut texel = case.clone();
    texel.argument_buffer_textures[0].texture_type = TextureType::Buffer;
    gap(&texel, "texture buffer");
}

/// End to end without a device: `argument_buffer_read` of the owned `handles` fixture on
/// `moltenvk-m3`, bound as the executor binds it, plans its argument-buffer texture as a heap
/// member (`HeapEntry` of its token) with the kernel's handle table, whose bytes the dispatch
/// fills to their planned length, and the resource id the owner holds at the field is the one
/// the bind state names for that member. That the table resolves each used resource's id to its
/// member's element is the binder's contract, which the crate's own tests check by decoding the
/// table (`a_sampled_heap_writes_each_member_and_its_table`).
#[test]
fn an_argument_buffer_texture_is_the_heap_member_its_owner_names() {
    let mut case = kernel_case();
    case.entry = "argument_buffer_read".into();
    case.buffers = vec![
        BufferResource {
            binding: 0,
            role: ResourceRole::Input,
            bytes_b64: Some(b64(&[0; 8])),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        },
        BufferResource {
            binding: 1,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some(b64(&[0xab; 16])),
            rasterization_rate_map: None,
        },
    ];
    case.argument_buffer_textures.push(member(
        0,
        TextureFormat::Rgba32Float,
        [1, 1, 1],
        ResourceRole::Input,
        &[0x3f; 16],
    ));
    case.output = OutputSelection::Buffer {
        binding: 1,
        offset: 0,
        length: 16,
    };
    let mapped = map_kernel(&case).unwrap();
    let bitcode = match llvm_assemble(&engine_dir().join("handles.ll")) {
        Ok(bitcode) => bitcode,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected handles.ll: {why}"),
    };
    let device = device("moltenvk-m3", DevicePolicy::RESIDENT_TABLES);
    let library = Library::from_air(&bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(&mapped.entry).unwrap());
    let shape = compute_shape(&mapped, true, BufferRanges::default()).unwrap();
    let (pipeline, _) = variants::compile_compute_for_shape(
        &device,
        &function,
        &shape,
        Some(&mapped.dispatch),
        &Budget::standard(),
    )
    .unwrap();
    let addresses = mapped
        .buffers
        .keys()
        .map(|&index| {
            let mut address = BufferAddress::synthetic(index);
            address.host = Some(address.guest);
            (index, address)
        })
        .collect();
    let state = bind_state(
        &mapped,
        &addresses,
        &BTreeMap::new(),
        BufferRanges::default(),
    )
    .unwrap();
    assert_eq!(state.shape(), shape);
    let plan = pipeline.plan(&device, &shape).unwrap();
    assert!(plan.diagnostics.is_empty(), "{:?}", plan.diagnostics);
    let token = texture_tokens(&mapped).unwrap()[&(0, 0)];
    let elements = plan
        .descriptors
        .iter()
        .filter_map(|write| match write.source {
            WriteSource::HeapEntry {
                source, element, ..
            } => Some((source, element)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(elements, [(HeapSource::Used(token), 1)]);
    let table = plan
        .descriptors
        .iter()
        .find(|write| matches!(write.source, WriteSource::HandleTable { .. }))
        .expect("the kernel's handle table");
    let commands = plan.instantiate_dispatch(&state, &mapped.dispatch).unwrap();
    let bytes = &commands
        .descriptor_bytes
        .iter()
        .find(|bytes| (bytes.set, bytes.binding) == (table.set, table.binding))
        .expect("the handle table's bytes")
        .bytes;
    let WriteSource::HandleTable { bytes_len, .. } = table.source else {
        unreachable!()
    };
    assert_eq!(bytes.len(), bytes_len as usize);
    let fields = field_words(
        &BTreeMap::new(),
        &texture_tokens(&mapped).unwrap(),
        &BTreeMap::new(),
    )
    .unwrap();
    let owner = owner_bytes(&mapped.buffers, 0, &fields).unwrap();
    let id = u64::from_le_bytes(owner[..8].try_into().unwrap());
    assert_eq!(id, resource_id(token));
    assert!(state
        .used_resources
        .iter()
        .any(|used| used.token == token && used.resource_id == Some(id)));
}

/// The material `{ texture2d<float> t; sampler s; }` of `argument_sampler_sample` (the owned
/// `tail_argument_samplers` fixture): buffer 0 holds it, buffer 1 the four `float4` outputs.
fn sampler_case() -> AuthoredCase {
    let mut case = kernel_case();
    case.entry = "argument_sampler_sample".into();
    case.buffers = vec![
        BufferResource {
            binding: 0,
            role: ResourceRole::Input,
            bytes_b64: Some(b64(&[0; 16])),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        },
        BufferResource {
            binding: 1,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some(b64(&[0xab; 64])),
            rasterization_rate_map: None,
        },
    ];
    case.argument_buffer_textures.push(member(
        0,
        TextureFormat::Rgba32Float,
        [1, 1, 1],
        ResourceRole::Input,
        &[0x3f; 16],
    ));
    case.argument_buffer_samplers
        .push(ArgumentBufferSamplerResource {
            buffer_binding: 0,
            field_offset: 8,
            address_mode: SamplerAddressMode::ClampToEdge,
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            mip_filter: SamplerMipFilter::NotMipmapped,
            normalized_coordinates: true,
            compare_function: None,
        });
    case.output = OutputSelection::Buffer {
        binding: 1,
        offset: 0,
        length: 64,
    };
    case
}

/// An argument-buffer sampler maps to its authored descriptor by key, its owner grown to cover its
/// field; a key authored twice and an owner that is not bound are refused.
#[test]
fn an_argument_buffer_sampler_maps_by_its_field() {
    let case = sampler_case();
    let mapped = map_kernel(&case).unwrap();
    assert_eq!(
        mapped.argument_buffer_samplers.keys().collect::<Vec<_>>(),
        [&(0, 8)]
    );
    assert_eq!(mapped.buffers[&0].len(), 16);
    let mut twice = case.clone();
    twice
        .argument_buffer_samplers
        .push(twice.argument_buffer_samplers[0].clone());
    gap(&twice, "argument-buffer sampler 0+8 bound twice");
    let mut ownerless = case.clone();
    ownerless.argument_buffer_samplers[0].buffer_binding = 3;
    gap(
        &ownerless,
        "argument-buffer sampler 3+8 without its owner buffer",
    );
    // Its field cannot also hold another member.
    let mut shared = case;
    shared.argument_buffer_textures[0].field_offset = 8;
    let mapped = map_kernel(&shared).unwrap();
    assert!(field_words(
        &BTreeMap::new(),
        &texture_tokens(&mapped).unwrap(),
        &sampler_ids(&mapped).unwrap(),
    )
    .is_err());
}

/// End to end without a device: `argument_sampler_sample` of the owned `tail_argument_samplers`
/// fixture on `moltenvk-m3`, bound as the executor binds it. The bind state lists the sampler with
/// its authored descriptor and the id the owner holds at the field; the plan fills the sampler
/// array with it after the neutral element, and the dispatch fills each of the kernel's tables to
/// its planned length. That the sampler table resolves the listed id to its element is the binder's
/// contract, which the crate's own tests check by decoding the table
/// (`src/tests/engine_heap_binder.rs`).
#[test]
fn an_argument_buffer_sampler_is_the_sampler_table_entry_its_owner_names() {
    let mapped = map_kernel(&sampler_case()).unwrap();
    let bitcode = match llvm_assemble(&engine_dir().join("tail_argument_samplers.ll")) {
        Ok(bitcode) => bitcode,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected tail_argument_samplers.ll: {why}")
        }
    };
    let device = device("moltenvk-m3", DevicePolicy::RESIDENT_TABLES);
    let library = Library::from_air(&bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(&mapped.entry).unwrap());
    let shape = compute_shape(&mapped, true, BufferRanges::default()).unwrap();
    let (pipeline, _) = variants::compile_compute_for_shape(
        &device,
        &function,
        &shape,
        Some(&mapped.dispatch),
        &Budget::standard(),
    )
    .unwrap();
    let addresses = mapped
        .buffers
        .keys()
        .map(|&index| {
            let mut address = BufferAddress::synthetic(index);
            address.host = Some(address.guest);
            (index, address)
        })
        .collect();
    let state = bind_state(
        &mapped,
        &addresses,
        &BTreeMap::new(),
        BufferRanges::default(),
    )
    .unwrap();
    assert_eq!(state.shape(), shape);
    let id = sampler_ids(&mapped).unwrap()[&(0, 8)];
    assert_eq!(id, sampler_id(0).unwrap());
    assert_ne!(id, resource_id(texture_tokens(&mapped).unwrap()[&(0, 0)]));
    assert_eq!(state.argument_samplers.len(), 1);
    let listed = &state.argument_samplers[0];
    assert_eq!(listed.resource_id, id);
    assert_eq!(
        listed.desc,
        argument_sampler_descriptor(&sampler_case().argument_buffer_samplers[0])
    );
    assert_eq!(listed.desc.min_filter, SamplerMinMagFilter::Linear);
    let plan = pipeline.plan(&device, &shape).unwrap();
    assert!(plan.diagnostics.is_empty(), "{:?}", plan.diagnostics);
    // One sampled, non-comparing float use: one array of twice the class's 16 entries, the
    // neutral sampler first (every plan's sampler arrays and tables hold twice their class,
    // whatever filter the states listed have, for states made late).
    let mut elements = plan
        .descriptors
        .iter()
        .filter_map(|write| match write.source {
            WriteSource::ArgumentSampler {
                index,
                compare,
                integer,
            } => Some((write.array_element, index, compare, integer)),
            _ => None,
        })
        .collect::<Vec<_>>();
    elements.sort();
    assert_eq!(
        elements,
        (1..32)
            .map(|e| (e, e - 1, false, false))
            .collect::<Vec<_>>()
    );
    assert!(plan.descriptors.iter().any(|write| write.array_element == 0
        && write.source == WriteSource::Neutral(NeutralResource::Sampler { compare: false })));
    let commands = plan.instantiate_dispatch(&state, &mapped.dispatch).unwrap();
    let fields = field_words(
        &BTreeMap::new(),
        &texture_tokens(&mapped).unwrap(),
        &sampler_ids(&mapped).unwrap(),
    )
    .unwrap();
    let owner = owner_bytes(&mapped.buffers, 0, &fields).unwrap();
    assert_eq!(u64::from_le_bytes(owner[8..16].try_into().unwrap()), id);
    let tables = plan
        .descriptors
        .iter()
        .filter_map(|write| match write.source {
            WriteSource::HandleTable { bytes_len, .. } => {
                Some((write.set, write.binding, bytes_len))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!tables.is_empty(), "the kernel's sampler table");
    for (set, binding, bytes_len) in tables {
        let bytes = &commands
            .descriptor_bytes
            .iter()
            .find(|bytes| (bytes.set, bytes.binding) == (set, binding))
            .expect("a table's bytes")
            .bytes;
        assert_eq!(bytes.len(), bytes_len as usize);
    }
}
