//! Argument-buffer textures, samplers, command buffers, and pipeline states in the engine
//! executor: a texture is a used texture whose resource id its owner holds at its field, a member
//! of the plan's heap that the kernel's handle table finds; a sampler is an argument sampler whose
//! id its owner holds, an element of the plan's sampler arrays that the kernel's sampler table
//! finds; a command buffer is its ID in its owner, and its record storage an argument-buffer buffer
//! at the address the product resolves that ID at; a pipeline state is its token in its owner,
//! which a translated pipeline reads only as an encoder's operand. No device is opened.

use super::{b64, device, kernel_case, map_kernel};
use crate::candidate::engine::argument_buffers::{
    field_words, owner_bytes, records_storage, resource_id, sampler_id,
};
use crate::candidate::engine::resources::argument_sampler_descriptor;
use crate::candidate::engine::{
    bind_state, compute_shape, product_refusal, records_ids, require_case_families, sampler_ids,
    synthetic_used_addresses, texture_tokens, variants, BufferAddress, ComputeOutput,
};
use crate::case::indirect_commands::{CommandBufferResource, PipelineResource};
use crate::case::{
    ArgumentBufferBufferResource, ArgumentBufferSamplerResource, ArgumentBufferTextureResource,
    AuthoredCase, BufferResource, OutputSelection, ResourceRole, SamplerAddressMode, SamplerFilter,
    SamplerMipFilter, TextureFormat, TextureType,
};
use crate::check::admission::{ExecutorError, NotSupported};
use crate::engine::{Outcome, UnsupportedReason};
use metal2vulkan::bind::indirect_commands::records_address;
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

/// A compute-dispatch command buffer at `field_offset` of buffer 0.
fn command_buffer(field_offset: u32) -> CommandBufferResource {
    serde_json::from_value(serde_json::json!({
        "buffer_binding": 0,
        "field_offset": field_offset,
        "capacity": 1,
        "descriptor": {
            "dispatch": "threadgroups",
            "kind": "compute",
            "max_buffer_bind_count": 0,
            "max_threadgroup_memory_bind_count": 0,
            "support_dynamic_attribute_stride": false
        },
        "execution_range": [0, 1],
        "inherit_buffers": true,
        "inherit_pipeline_state": true
    }))
    .unwrap()
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
/// writing like the oracle's `useResource`; the command buffer is its initialized record storage,
/// an argument-buffer buffer after the authored one; the owner is created holding the buffers'
/// addresses, each texture's resource id, and the pipeline state's authored token at their fields;
/// a texture's region is an output; and an owner that is not bound, a key authored twice, a texture
/// buffer, a field named twice, and overlapping fields are refused.
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
    case.command_buffers.push(command_buffer(32));
    case.pipeline_states.push(PipelineResource {
        buffer_binding: 0,
        field_offset: 40,
        token: 7,
        air_sha256: "22".repeat(32),
        case_id: "33".repeat(32),
    });
    case.output = OutputSelection::ArgumentBufferTexture {
        buffer_binding: 0,
        field_offset: 24,
        origin: [0, 0, 0],
        dimensions: [1, 1, 1],
    };
    assert_eq!(require_case_families(&case), Ok(()));
    let mapped = map_kernel(&case).unwrap();
    assert_eq!(mapped.buffers[&0].len(), 48);
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
    assert_eq!(mapped.objects, BTreeMap::from([((0, 40), 7)]));
    let command = command_buffer(32);
    assert_eq!(
        mapped.command_buffers,
        BTreeMap::from([((0, 32), command.clone())])
    );
    let mut storage_bytes = vec![0; command.record_storage_bytes().unwrap()];
    command.initialize_records(&mut storage_bytes).unwrap();
    assert_eq!(
        mapped.argument_buffer_buffers,
        BTreeMap::from([((0, 8), vec![5; 12]), ((0, 32), storage_bytes.clone())])
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

    // The bind state: the buffer is token 0, the record storage 1, the textures follow them
    // with their resource ids. The command buffer's ID is 1, its record storage at that ID's
    // records address with a device address of its own.
    let tokens = texture_tokens(&mapped).unwrap();
    assert_eq!(tokens, BTreeMap::from([((0, 16), 2), ((0, 24), 3)]));
    let records = records_ids(&mapped).unwrap();
    assert_eq!(records, BTreeMap::from([((0, 32), 1)]));
    assert_eq!(
        synthetic_used_addresses(&mapped).unwrap()[&(0, 32)],
        records_storage(1, None)
    );
    let host = |guest| BufferAddress {
        guest,
        host: Some(guest),
    };
    let addresses = BTreeMap::from([(0, host(0x1000))]);
    let embedded = BTreeMap::from([
        ((0, 8), host(0xdead_0000)),
        ((0, 32), records_storage(1, Some(0xbeef_0000))),
    ]);
    let state = bind_state(&mapped, &addresses, &embedded, BufferRanges::default()).unwrap();
    let [buffer, storage, first, second] = state.used_resources.as_slice() else {
        panic!("{:?}", state.used_resources);
    };
    assert_eq!((buffer.class, buffer.token), (ResourceClass::Buffer, 0));
    assert_eq!(
        (
            storage.class,
            storage.token,
            storage.gpu_address,
            storage.device_address,
            storage.length,
            storage.usage
        ),
        (
            ResourceClass::Buffer,
            1,
            Some(records_address(1)),
            Some(0xbeef_0000),
            storage_bytes.len() as u64,
            ResourceUsage::READ | ResourceUsage::WRITE
        )
    );
    for (used, token, format) in [
        (first, 2, PixelFormat::RGBA8Unorm),
        (second, 3, PixelFormat::R32Uint),
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
        assert_eq!(shape.used.len(), 4);
        if addressed {
            assert_eq!(state.shape(), shape);
        }
    }

    // The owner holds, at each field, the word Metal's argument encoder writes there: the
    // command buffer's ID, not its record storage's address.
    let fields = field_words(
        &embedded,
        &records,
        &tokens,
        &BTreeMap::new(),
        mapped.objects.iter().map(|(&key, &word)| (key, word)),
    )
    .unwrap();
    let owner = owner_bytes(&mapped.buffers, 0, &fields).unwrap();
    let word = |at: usize| u64::from_le_bytes(owner[at..at + 8].try_into().unwrap());
    assert_eq!(&owner[..8], &[0xab, 0xab, 0xab, 0xab, 0, 0, 0, 0]);
    assert_eq!(
        [word(8), word(16), word(24), word(32), word(40)],
        [0xdead_0000, resource_id(2), resource_id(3), 1, 7]
    );
    // A command buffer without its record storage, and record storage away from its ID's
    // address, are refused.
    let words = |embedded: &BTreeMap<(u32, u32), BufferAddress>| {
        field_words(
            embedded,
            &records,
            &tokens,
            &BTreeMap::new(),
            std::iter::empty(),
        )
    };
    let mut without = embedded.clone();
    without.remove(&(0, 32));
    assert!(words(&without).is_err());
    let mut away = embedded.clone();
    away.insert((0, 32), host(0xbeef_0000));
    assert!(words(&away).is_err());
    // Ids are nonzero (a handle table never enters id 0) and distinct.
    let ids = (0..64)
        .map(resource_id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ids.len(), 64);
    assert!(!ids.contains(&0));
    // A field named by two members, overlapping fields, and a field past the owner are refused.
    assert!(field_words(
        &embedded,
        &BTreeMap::new(),
        &BTreeMap::from([((0, 8), 1)]),
        &BTreeMap::new(),
        std::iter::empty()
    )
    .is_err());
    let overlapping = BTreeMap::from([((0, 8), 1), ((0, 12), 2)]);
    assert!(owner_bytes(&mapped.buffers, 0, &overlapping).is_err());
    assert!(owner_bytes(&mapped.buffers, 0, &BTreeMap::from([((0, 44), 1)])).is_err());

    let mut orphan = case.clone();
    orphan.argument_buffer_textures[0].buffer_binding = 3;
    gap(&orphan, "without its owner buffer");
    let mut twice = case.clone();
    twice.argument_buffer_textures[1].field_offset = 16;
    gap(&twice, "bound twice");
    let mut texel = case.clone();
    texel.argument_buffer_textures[0].texture_type = TextureType::Buffer;
    gap(&texel, "texture buffer");
    let mut command_orphan = case.clone();
    command_orphan.command_buffers[0].buffer_binding = 7;
    gap(&command_orphan, "without its owner buffer");
    let mut pipeline_orphan = case.clone();
    pipeline_orphan.pipeline_states[0].buffer_binding = 7;
    gap(&pipeline_orphan, "without its owner buffer");
    let mut shared = case.clone();
    shared.pipeline_states[0].field_offset = 32;
    gap(&shared, "pipeline state 0+32 bound twice");
    let mut over_buffer = case.clone();
    over_buffer.command_buffers[0].field_offset = 8;
    gap(&over_buffer, "command buffer 0+8 bound twice");
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
    let device = device("moltenvk-m3", DevicePolicy::default());
    let library = Library::from_air(&bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(&mapped.entry).unwrap());
    let shape = compute_shape(&mapped, true, BufferRanges::default()).unwrap();
    let (pipeline, _) = variants::compile_compute_for_shape(
        &device,
        &function,
        &shape,
        Some(&mapped.dispatch),
        None,
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
        &BTreeMap::new(),
        &texture_tokens(&mapped).unwrap(),
        &BTreeMap::new(),
        std::iter::empty(),
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

/// A kernel `name` whose argument buffer `%ab` (Metal buffer 0, constant space) is
/// `{compute_pipeline_state compute; uint scale; uint bias;}`, with a `ulong` output buffer `%out`
/// (Metal buffer 1); `body` precedes its `ret`.
fn pipeline_state_kernel(name: &str, body: &str) -> String {
    format!(
        r#"%ab = type {{ ptr addrspace(1), i32, i32 }}

define void @{name}(ptr addrspace(2) %ab, ptr addrspace(1) %out) {{
{body}
  ret void
}}

!air.kernel = !{{!0}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !8}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Pipelines", !"air.arg_name", !"ab"}}
!4 = !{{i32 0, i32 8, i32 0, !"compute_pipeline_state", !"compute", !"air.indirect_argument", !5, i32 8, i32 4, i32 0, !"uint", !"scale", !"air.indirect_argument", !6, i32 12, i32 4, i32 0, !"uint", !"bias", !"air.indirect_argument", !7}}
!5 = !{{i32 0, !"air.compute_pipeline_state", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"compute"}}
!6 = !{{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"scale"}}
!7 = !{{i32 2, !"air.indirect_constant", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"bias"}}
!8 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The executor's word for a pipeline state (the case's token) is never the id Metal's encoder
/// writes, so no case whose output depends on a pipeline-state member's bytes may be compared.
/// Through the executor's own mapping and compile, on each Vulkan profile the candidate runs: a
/// kernel that reads only the argument buffer's data members compiles, and its owner holds the
/// token at the pipeline-state field; a kernel whose output depends on the field is the product's
/// typed `ArgumentBufferObjects` refusal, which the executor reports as not supported, never as an
/// observation. That covers the field's word, and a null test of the pipeline state, loaded as the
/// handle or at an offset not known at translation time: Metal's resource id and the token are both
/// nonzero once set, so a null test would otherwise agree.
#[test]
fn a_pipeline_state_whose_bytes_a_kernel_reads_never_reaches_a_comparison() {
    let data = "  %ps = getelementptr inbounds %ab, ptr addrspace(2) %ab, i64 0, i32 1\n  %s = load i32, ptr addrspace(2) %ps, align 4\n  %pb = getelementptr inbounds %ab, ptr addrspace(2) %ab, i64 0, i32 2\n  %b = load i32, ptr addrspace(2) %pb, align 4\n  %v = add i32 %s, %b\n  %z = zext i32 %v to i64\n  store i64 %z, ptr addrspace(1) %out, align 8";
    let word = "  %w = load i64, ptr addrspace(2) %ab, align 8\n  store i64 %w, ptr addrspace(1) %out, align 8";
    let null = "  %c = icmp eq ptr addrspace(1) %h, null\n  %w = zext i1 %c to i64\n  store i64 %w, ptr addrspace(1) %out, align 8";
    let handle_null =
        format!("  %h = load ptr addrspace(1), ptr addrspace(2) %ab, align 8\n{null}");
    // `scale & 1` selects the field (0) or the data that follows it (8).
    let dynamic_null = format!("  %ps = getelementptr inbounds %ab, ptr addrspace(2) %ab, i64 0, i32 1\n  %s = load i32, ptr addrspace(2) %ps, align 4\n  %a = and i32 %s, 1\n  %i = zext i32 %a to i64\n  %pp = getelementptr inbounds ptr addrspace(1), ptr addrspace(2) %ab, i64 %i\n  %h = load ptr addrspace(1), ptr addrspace(2) %pp, align 8\n{null}");
    for (entry, body, compiles) in [
        ("pipeline_data", data, true),
        ("pipeline_word", word, false),
        ("pipeline_null", handle_null.as_str(), false),
        ("pipeline_dynamic_null", dynamic_null.as_str(), false),
    ] {
        let Some(bitcode) = super::assemble(&pipeline_state_kernel(entry, body)) else {
            return;
        };
        let mut case = kernel_case();
        case.entry = entry.into();
        case.buffers = vec![
            BufferResource {
                binding: 0,
                role: ResourceRole::Input,
                bytes_b64: Some(b64(&[0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0])),
                initial_bytes_b64: None,
                rasterization_rate_map: None,
            },
            BufferResource {
                binding: 1,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some(b64(&[0xab; 8])),
                rasterization_rate_map: None,
            },
        ];
        case.pipeline_states.push(PipelineResource {
            buffer_binding: 0,
            field_offset: 0,
            token: 5,
            air_sha256: "22".repeat(32),
            case_id: "33".repeat(32),
        });
        case.output = OutputSelection::Buffer {
            binding: 1,
            offset: 0,
            length: 8,
        };
        assert_eq!(require_case_families(&case), Ok(()));
        let mapped = map_kernel(&case).unwrap();
        let fields = field_words(
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            mapped.objects.iter().map(|(&key, &word)| (key, word)),
        )
        .unwrap();
        let owner = owner_bytes(&mapped.buffers, 0, &fields).unwrap();
        assert_eq!(u64::from_le_bytes(owner[..8].try_into().unwrap()), 5);
        assert_eq!(&owner[8..], &[3, 0, 0, 0, 4, 0, 0, 0]);
        let library = Library::from_air(&bitcode).unwrap();
        let function = FunctionDescriptor::new(library.function(entry).unwrap());
        for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
            let device = device(stem, DevicePolicy::default());
            for addressed in [false, true] {
                let shape = compute_shape(&mapped, addressed, BufferRanges::default()).unwrap();
                let compiled = variants::compile_compute_for_shape(
                    &device,
                    &function,
                    &shape,
                    Some(&mapped.dispatch),
                    None,
                    &Budget::standard(),
                );
                match (compiled, compiles) {
                    (Ok(_), true) => {}
                    (Err(error), false) => match product_refusal(&error) {
                        ExecutorError::NotSupported(NotSupported::Product {
                            outcome:
                                Outcome::Unsupported {
                                    reason: UnsupportedReason::NotImplemented,
                                    feature,
                                },
                            ..
                        }) => assert_eq!(feature, "ArgumentBufferObjects", "{stem} {entry}"),
                        other => panic!("{stem} {entry}: {other:?}"),
                    },
                    (Ok(_), false) => panic!("{stem} {entry} compiled"),
                    (Err(error), true) => panic!("{stem} {entry}: {error}"),
                }
            }
        }
    }
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
        &BTreeMap::new(),
        &texture_tokens(&mapped).unwrap(),
        &sampler_ids(&mapped).unwrap(),
        std::iter::empty(),
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
    let device = device("moltenvk-m3", DevicePolicy::default());
    let library = Library::from_air(&bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(&mapped.entry).unwrap());
    let shape = compute_shape(&mapped, true, BufferRanges::default()).unwrap();
    let (pipeline, _) = variants::compile_compute_for_shape(
        &device,
        &function,
        &shape,
        Some(&mapped.dispatch),
        None,
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
        &BTreeMap::new(),
        &texture_tokens(&mapped).unwrap(),
        &sampler_ids(&mapped).unwrap(),
        std::iter::empty(),
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
