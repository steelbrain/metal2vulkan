//! Argument-buffer textures and samplers in the engine executor's mapping: a texture's resource id
//! and a sampler's id are what its owner holds at its field. No device is opened.

use super::{b64, kernel_case, map_kernel};
use crate::candidate::engine::argument_buffers::owner_bytes;
use crate::candidate::engine::{require_case_families, ComputeOutput};
use crate::case::{
    ArgumentBufferBufferResource, ArgumentBufferSamplerResource, ArgumentBufferTextureResource,
    AuthoredCase, BufferResource, OutputSelection, ResourceRole, SamplerAddressMode, SamplerFilter,
    SamplerMipFilter, TextureFormat, TextureType,
};
use crate::check::admission::NotSupported;
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
/// grows to cover each 8-byte field; the owner is created holding the buffer's address and each
/// texture's resource id at their fields; a texture's region is an output; and an owner that is
/// not bound, a key authored twice, overlapping fields, and a field past the owner are refused. A
/// texture buffer maps, as a run through the runtime makes it.
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

    // The owner holds, at each field, the word Metal's argument encoder writes there.
    let fields = BTreeMap::from([((0, 8), 0xdead_0000), ((0, 16), 0x11), ((0, 24), 0x22)]);
    let owner = owner_bytes(&mapped.buffers, 0, &fields).unwrap();
    let word = |at: usize| u64::from_le_bytes(owner[at..at + 8].try_into().unwrap());
    assert_eq!(&owner[..8], &[0xab, 0xab, 0xab, 0xab, 0, 0, 0, 0]);
    assert_eq!([word(8), word(16), word(24)], [0xdead_0000, 0x11, 0x22]);
    // Overlapping fields and a field past the owner are refused.
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
    let texel = map_kernel(&texel).unwrap();
    assert_eq!(
        texel.argument_buffer_textures[&(0, 16)].texture_type,
        TextureType::Buffer
    );
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
}
