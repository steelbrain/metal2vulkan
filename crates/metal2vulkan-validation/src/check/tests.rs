use super::*;
use crate::case::TextureType;

/// The store-wide sweep names exactly the installed cases the current checker rejects.
///
/// A case is installed under the semantics of its day and then left alone. `e84ea3fa` moved 41
/// authored texture arrays to a different Metal slot without anyone re-reading the store, so the
/// property under test is not "a bad manifest is rejected" -- `check_case` already did that --
/// but that a manifest ALREADY in the store is re-read at all.
#[test]
fn the_store_sweep_names_the_installed_cases_the_checker_now_rejects() {
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection, ResourceRole, Stage,
    };
    use crate::store::CorpusStore;
    use crate::ScratchDir;

    let air_ll =
        std::fs::read_to_string(metal2vulkan_devtools::fixtures_dir().join("kernel_copy_word.ll"))
            .expect("public fixture");
    let air_sha256 = crate::hash::sha256_bytes(air_ll.as_bytes());
    let buffer = |binding, role| BufferResource {
        binding,
        role,
        bytes_b64: (role == ResourceRole::Input).then(|| "AAAAAA==".to_string()),
        initial_bytes_b64: (role != ResourceRole::Input).then(|| "q6urqw==".to_string()),
        rasterization_rate_map: None,
    };
    let case = |name: &str, buffers: Vec<BufferResource>| {
        let mut case = AuthoredCase {
            air_sha256: air_sha256.clone(),
            case_id: String::new(),
            name: name.into(),
            entry: "copy_word".into(),
            stage: Stage::Kernel,
            buffers,
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            command_buffers: vec![],
            pipeline_states: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            argument_buffer_samplers: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            varying_values: Vec::new(),
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
                stage_in_region: None,
            }),
            draw: None,
            tessellation: None,
            mesh_draw: None,
            output: OutputSelection::Buffer {
                binding: 1,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            fuel: None,
            out_of_bounds: None,
            rationale: None,
            authored_by: None,
            render_pass_rate_map: None,
            raster_sample_count: None,
            public_source: None,
            no_effect: None,
            creation_bound: None,
        };
        case.case_id = case.computed_case_id().expect("case id");
        case
    };

    let scratch = ScratchDir::new("recheck-installed-cases").expect("scratch");
    let store = CorpusStore::new(scratch.path());
    let sound = case(
        "copies-the-word",
        vec![
            buffer(0, ResourceRole::Input),
            buffer(1, ResourceRole::Output),
        ],
    );
    // Installed straight into the store, exactly as a case authored under older semantics sits
    // there today: `put_case` does not check, only `corpus-case-check --install` does.
    let stale = case(
        "copies-the-word-from-nowhere",
        vec![buffer(1, ResourceRole::Output)],
    );
    store.put_case(sound.clone()).expect("install sound case");
    store.put_case(stale.clone()).expect("install stale case");

    let recheck = recheck_installed_cases(scratch.path(), &store).expect("sweep the store");
    let reported = recheck.stale;
    assert_eq!(recheck.total, 2);
    assert!(recheck.partial.is_empty());
    assert_eq!(
        reported
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        vec!["copies-the-word-from-nowhere"],
        "the sound case must not be reported and the stale one must be"
    );
    assert!(
        reported[0]
            .errors
            .iter()
            .any(|error| error.contains("buffer 0")),
        "the report carries the checker's own reason: {:?}",
        reported[0].errors
    );
}

/// The writability gate asks whether the shader can write the selected output, not what its
/// qualifier spells: the compiler annotates some written buffers `air.read` (17 corpus modules,
/// `particle_fill_grid` among them), and the checker takes the write from the IR's stores.
#[test]
fn an_output_declared_read_but_written_by_the_kernel_is_accepted() {
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection, ResourceRole, Stage,
    };

    let air_ll = std::fs::read_to_string(
        metal2vulkan_devtools::fixtures_dir().join("kernel_copy_word_declared_read.ll"),
    )
    .expect("public fixture");
    let interface =
        interface::tests::of_text(&air_ll, "copy_word_declared_read", &BTreeMap::new()).unwrap();
    assert!(
        interface.params.iter().all(|param| matches!(
            param.binding.role,
            metal2vulkan_air::abi::ParamRole::Buffer(ref buffer)
                if buffer.access == metal2vulkan_air::abi::Access::Read
        )),
        "the fixture must declare both buffers read-only"
    );

    let mut case = AuthoredCase {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        case_id: String::new(),
        name: "copies-the-word-through-a-buffer-declared-read".into(),
        entry: "copy_word_declared_read".into(),
        stage: Stage::Kernel,
        buffers: vec![
            BufferResource {
                binding: 0,
                role: ResourceRole::Input,
                bytes_b64: Some("KgAAAA==".to_string()),
                initial_bytes_b64: None,
                rasterization_rate_map: None,
            },
            BufferResource {
                binding: 1,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".to_string()),
                rasterization_rate_map: None,
            },
        ],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 1,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: None,
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    };
    case.case_id = case.computed_case_id().expect("case id");
    let none = conformance::ImageblockOutputs::default();
    assert!(conformance::selected_output_is_writable(
        &case, &interface, &none
    ));
    // The input buffer is neither declared writable nor written.
    case.output = OutputSelection::Buffer {
        binding: 0,
        offset: 0,
        length: 4,
    };
    assert!(!conformance::selected_output_is_writable(
        &case, &interface, &none
    ));
}

/// A sparse texture the function declares writable is admitted as an input: the executors make it
/// writable and read nothing of it back, so a write to an unmapped texel is observed by the
/// function's own reads, which Metal answers. It was refused while no case measured that write.
#[test]
fn a_sparse_texture_the_function_declares_writable_is_an_input() {
    use crate::case::sparse::{SparseMapping, SparseTextureResource};
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection, ResourceRole,
        Stage, TextureResource,
    };
    use base64::Engine as _;

    let air_ll = include_str!("../../fixtures/public/kernel_write_short_texture.ll");
    let entry = "kernel_write_short_texture";
    let interface = interface::tests::of_text(air_ll, entry, &BTreeMap::new()).unwrap();
    let texture = |sparse: Option<SparseTextureResource>| TextureResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba16Sint,
        dimensions: [128, 128, 1],
        sample_count: 1,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode(vec![0u8; 8 * 128 * 128])),
        initial_bytes_b64: None,
        sparse,
        mip_levels: 1,
    };
    let case = |sparse| AuthoredCase {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        case_id: String::new(),
        name: "writes-a-sparse-texture".into(),
        entry: entry.into(),
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("AAAAAA==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![texture(sparse)],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        render_pass_rate_map: None,
        raster_sample_count: None,
        rationale: None,
        authored_by: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    };
    let errors = |case: &AuthoredCase| {
        let mut errors = Vec::new();
        conformance::validate_textures(case, &interface, &mut errors);
        errors
    };
    let mapped = SparseTextureResource {
        mappings: vec![SparseMapping::Region {
            level: 0,
            slice: 0,
            origin: [0, 0, 0],
            size: [128, 128, 1],
        }],
    };
    // A write is observed only by the function's own reads: the texture stays an input, admitted
    // sparse or not.
    assert_eq!(errors(&case(Some(mapped))), Vec::<String>::new());
    assert_eq!(errors(&case(None)), Vec::<String>::new());
}

fn texture_use(
    dim: interface::ImageDimKey,
    array: bool,
    texel: interface::TexelClass,
) -> interface::TextureUse {
    interface::TextureUse {
        dim,
        array,
        ms: false,
        depth: false,
        texel: Some(texel),
    }
}

#[test]
fn a_texture_is_checked_against_the_shape_and_texels_its_operations_spell() {
    use interface::{ImageDimKey, TexelClass};
    let half = texture_use(ImageDimKey::D2, false, TexelClass::Float);
    let mut errors = Vec::new();
    // A single-channel half image backs a `texture2d<half>`: the class is float.
    conformance::validate_texture_uses(
        "half texture",
        TextureType::D2,
        TextureFormat::R16Float,
        &[half],
        &mut errors,
    );
    assert!(errors.is_empty(), "{errors:?}");
    conformance::validate_texture_uses(
        "half texture",
        TextureType::D2Array,
        TextureFormat::R16Float,
        &[half],
        &mut errors,
    );
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("requires texture type D2"), "{errors:?}");
    errors.clear();
    conformance::validate_texture_uses(
        "uint texture",
        TextureType::D2,
        TextureFormat::R16Float,
        &[texture_use(ImageDimKey::D2, false, TexelClass::Uint)],
        &mut errors,
    );
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("texels"), "{errors:?}");
}

#[test]
fn a_texture_may_match_a_function_constant_alternative() {
    use interface::{ImageDimKey, TexelClass};
    let mut errors = Vec::new();
    conformance::validate_alternatives(
        "conditional texture",
        TextureType::D2,
        TextureFormat::Rgba16Float,
        vec![
            vec![texture_use(ImageDimKey::D2, true, TexelClass::Float)],
            vec![texture_use(ImageDimKey::D2, false, TexelClass::Float)],
        ],
        &mut errors,
    );
    assert!(errors.is_empty(), "{errors:?}");
    conformance::validate_alternatives(
        "conditional texture",
        TextureType::D3,
        TextureFormat::Rgba16Float,
        vec![
            vec![texture_use(ImageDimKey::D2, true, TexelClass::Float)],
            vec![texture_use(ImageDimKey::D2, false, TexelClass::Float)],
        ],
        &mut errors,
    );
    assert_eq!(errors.len(), 1, "{errors:?}");
}

/// The public fixture whose map operand is its first parameter, and its entry.
const RATE_MAP_FIXTURE: (&str, &str) = (
    include_str!("../../fixtures/public/kernel_rate_map_physical_to_screen.ll"),
    "kernel_rate_map_physical_to_screen",
);

/// A uniform map of full quality over the fixture's 64x64 screen.
fn identity_rate_map() -> crate::case::RateMapLiteral {
    use crate::case::{RateMapLayerLiteral, RateMapLiteral, RateSample};
    RateMapLiteral {
        screen_size: [64, 64],
        layers: vec![RateMapLayerLiteral {
            horizontal: vec![RateSample(1.0)],
            vertical: vec![RateSample(1.0)],
        }],
    }
}

/// A case of a rate-map kernel of `air_ll`: buffer 0 its input, authored as `map` or as bytes,
/// and buffer 1 its output, authored as `output_map` (a misplaced map) or as bytes.
fn rate_map_case(
    air_ll: &str,
    map: Option<crate::case::RateMapLiteral>,
    output_map: Option<crate::case::RateMapLiteral>,
) -> AuthoredCase {
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection, ResourceRole, Stage,
    };
    use base64::Engine as _;

    let input = BufferResource {
        binding: 0,
        role: ResourceRole::Input,
        bytes_b64: map.is_none().then(|| "AACAPw==".to_string()),
        initial_bytes_b64: None,
        rasterization_rate_map: map,
    };
    let output = BufferResource {
        binding: 1,
        role: ResourceRole::Output,
        bytes_b64: None,
        initial_bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0xab; 64])),
        rasterization_rate_map: output_map,
    };
    let mut case = AuthoredCase {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        case_id: String::new(),
        name: "map-physical-coordinates".into(),
        entry: RATE_MAP_FIXTURE.1.into(),
        stage: Stage::Kernel,
        buffers: vec![input, output],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [8, 1, 1],
            threads_per_threadgroup: [8, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 1,
            offset: 0,
            length: 64,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: None,
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    };
    case.case_id = case.computed_case_id().unwrap();
    case
}

/// The declared reflection of kernel `entry` of `air_ll`.
fn kernel_reflection(air_ll: &str, entry: &str) -> crate::air::reflect::ShaderInterface {
    crate::air::reflect::inspect(air_ll, crate::case::Stage::Kernel, entry, &[]).unwrap()
}

/// The checker's rate-map errors of `case` against kernel `entry` of `air_ll`: the literals
/// against the placement.
fn rate_map_errors(case: &AuthoredCase, air_ll: &str, entry: &str) -> Vec<String> {
    let interface = interface::tests::of_text(air_ll, entry, &BTreeMap::new()).unwrap();
    let placement = crate::executor_contract::rate_map::rate_map_placement(
        air_ll,
        &kernel_reflection(air_ll, entry),
    );
    let mut errors = Vec::new();
    conformance::validate_rate_maps(case, &interface, &placement, &mut errors);
    errors
}

/// The requirements the checker gives `case` of kernel `entry` of `air_ll` (what an executor
/// refuses).
fn rate_map_requirements(
    case: &AuthoredCase,
    air_ll: &str,
    entry: &str,
) -> std::collections::BTreeSet<crate::requirement::ToolingRequirement> {
    crate::executor_contract::checked_requirements(case, air_ll, &kernel_reflection(air_ll, entry))
}

/// A buffer a coordinate-mapping call reads its rasterization-rate map from is authored as the
/// map (`rasterization_rate_map`), never as bytes, and no other buffer is: the parameter data's
/// layout is each implementation's own. Such a case raises no placement requirement: every executor
/// writes its own implementation's map.
#[test]
fn a_rate_map_buffer_is_authored_as_its_map() {
    use crate::requirement::ToolingRequirement;

    let (air_ll, entry) = RATE_MAP_FIXTURE;
    let authored = rate_map_case(air_ll, Some(identity_rate_map()), None);
    assert_eq!(authored.validate_literal_resources(), Ok(()));
    assert_eq!(
        rate_map_errors(&authored, air_ll, entry),
        Vec::<String>::new()
    );
    assert!(!rate_map_requirements(&authored, air_ll, entry)
        .contains(&ToolingRequirement::RasterizationRateMapPlacement));

    let bytes = rate_map_case(air_ll, None, None);
    let errors = rate_map_errors(&bytes, air_ll, entry);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].starts_with("buffer 0 holds the rasterization-rate map"));

    let misplaced = rate_map_case(air_ll, Some(identity_rate_map()), Some(identity_rate_map()));
    let errors = rate_map_errors(&misplaced, air_ll, entry);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].starts_with("buffer 1 is authored as a rasterization-rate map"));
}

/// Map data at another offset of a buffer stays authored bytes, which hold one implementation's
/// layout: the case carries `rasterization_rate_map_placement`, so the Metal oracle runs it and the
/// engine's candidate refuses it rather than decode Apple's bytes as this crate's. The pair
/// cannot evidence it: since X2c the engine's is the only candidate, so the evidence gap names
/// it.
#[test]
fn map_data_at_another_offset_stops_the_engine() {
    use crate::requirement::{Executor, ToolingRequirement};
    use std::collections::BTreeSet;

    let (fixture, entry) = RATE_MAP_FIXTURE;
    let air_ll = fixture.replace(
        "%7 = bitcast ptr addrspace(2) %0 to ptr addrspace(2)",
        "%7 = getelementptr inbounds i8, ptr addrspace(2) %0, i64 16",
    );
    assert_ne!(air_ll, fixture);

    let bytes = rate_map_case(&air_ll, None, None);
    assert_eq!(
        rate_map_errors(&bytes, &air_ll, entry),
        Vec::<String>::new()
    );
    let requirements = rate_map_requirements(&bytes, &air_ll, entry);
    assert!(
        requirements.contains(&ToolingRequirement::RasterizationRateMapPlacement),
        "{requirements:?}"
    );
    assert_eq!(
        crate::executor_contract::evidence_gap(&requirements).as_deref(),
        Some("engine_candidate: rasterization_rate_map_placement (the candidate is stopped)")
    );
    assert_eq!(
        crate::executor_contract::require_executor(&requirements, Executor::EngineCandidate),
        Err(BTreeSet::from([
            ToolingRequirement::RasterizationRateMapPlacement
        ]))
    );
    assert_eq!(
        crate::executor_contract::require_executor(&requirements, Executor::MetalOracle),
        Ok(())
    );

    let literal = rate_map_case(&air_ll, Some(identity_rate_map()), None);
    let errors = rate_map_errors(&literal, &air_ll, entry);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].starts_with("buffer 0 is authored as a rasterization-rate map"));
}

/// The public fixture whose maps are argument-buffer members, and its entry.
const ARGUMENT_BUFFER_RATE_MAP_FIXTURE: (&str, &str) = (
    include_str!("../../fixtures/public/kernel_rate_map_argument_buffer.ll"),
    "kernel_rate_map_argument_buffer",
);

/// The public fixture that picks its map from a thread-memory array of buffer pointers, and its
/// entry.
const POINTER_ARRAY_RATE_MAP_FIXTURE: (&str, &str) = (
    include_str!("../../fixtures/public/kernel_rate_map_pointer_array.ll"),
    "kernel_rate_map_pointer_array",
);

/// The buffers an argument-buffer member holds are authored as the maps the calls read through
/// them: `left` at byte 8 and `right` at byte 16 of the argument buffer at buffer 0, never as
/// bytes, and the `coordinates` member at byte 0, which holds plain data, is not.
#[test]
fn a_rate_map_an_argument_buffer_holds_is_authored_as_its_map() {
    use crate::case::{ArgumentBufferBufferResource, ResourceRole};
    use crate::requirement::ToolingRequirement;

    let (air_ll, entry) = ARGUMENT_BUFFER_RATE_MAP_FIXTURE;
    let member =
        |field_offset, map: Option<crate::case::RateMapLiteral>| ArgumentBufferBufferResource {
            buffer_binding: 0,
            field_offset,
            role: ResourceRole::Input,
            bytes_b64: map.is_none().then(|| "AACAPwAAgD8=".to_string()),
            initial_bytes_b64: None,
            rasterization_rate_map: map,
        };
    let case = |members: Vec<ArgumentBufferBufferResource>| {
        let mut case = rate_map_case(air_ll, None, None);
        case.entry = entry.into();
        case.argument_buffer_buffers = members;
        case
    };

    let authored = case(vec![
        member(0, None),
        member(8, Some(identity_rate_map())),
        member(16, Some(identity_rate_map())),
    ]);
    assert_eq!(
        rate_map_errors(&authored, air_ll, entry),
        Vec::<String>::new()
    );
    assert!(!rate_map_requirements(&authored, air_ll, entry)
        .contains(&ToolingRequirement::RasterizationRateMapPlacement));

    let bytes = case(vec![
        member(0, None),
        member(8, Some(identity_rate_map())),
        member(16, None),
    ]);
    let errors = rate_map_errors(&bytes, air_ll, entry);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].starts_with("argument-buffer buffer 0+16 holds the rasterization-rate map"));

    let misplaced = case(vec![
        member(0, Some(identity_rate_map())),
        member(8, Some(identity_rate_map())),
        member(16, Some(identity_rate_map())),
    ]);
    let errors = rate_map_errors(&misplaced, air_ll, entry);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].starts_with("argument-buffer buffer 0+0 is authored as a rasterization-rate"));
}

/// Both buffers whose pointers a thread-memory array holds are authored as the maps a call reads
/// through a load from the array.
#[test]
fn rate_maps_a_pointer_array_holds_are_authored_as_their_maps() {
    use crate::requirement::ToolingRequirement;

    let (air_ll, entry) = POINTER_ARRAY_RATE_MAP_FIXTURE;
    let mut authored = rate_map_case(air_ll, Some(identity_rate_map()), Some(identity_rate_map()));
    authored.entry = entry.into();
    assert_eq!(
        rate_map_errors(&authored, air_ll, entry),
        Vec::<String>::new()
    );
    assert!(!rate_map_requirements(&authored, air_ll, entry)
        .contains(&ToolingRequirement::RasterizationRateMapPlacement));

    let mut bytes = rate_map_case(air_ll, Some(identity_rate_map()), None);
    bytes.entry = entry.into();
    let errors = rate_map_errors(&bytes, air_ll, entry);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].starts_with("buffer 1 holds the rasterization-rate map"));
}

/// A `buffer_atomic` out-of-bounds target is one an AIR atomic operation accesses, and a target
/// of any other form one none does: the form is the structure's, so the read rule an out-of-bounds
/// read takes is the one of the access that made it. Over a synthetic module: `a` takes an
/// `air.atomic.*` call, `b` an `atomicrmw`, and `c` only a store.
#[test]
fn an_out_of_bounds_buffer_targets_form_follows_its_atomic_accesses() {
    use crate::case::out_of_bounds::ContainmentForm;
    let ll = r#"
define void @atomics(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
  %x = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %a, i32 1, i32 0, i32 2, i1 true)
  %y = atomicrmw add ptr addrspace(1) %b, i32 %x monotonic
  store i32 %y, ptr addrspace(1) %c
  ret void
}
declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)
!air.kernel = !{!0}
!0 = !{ptr @atomics, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"c"}
"#;
    let interface = interface::tests::of_text(ll, "atomics", &BTreeMap::new()).unwrap();
    assert!(interface.atomic_through(0));
    assert!(interface.atomic_through(1));
    assert!(!interface.atomic_through(2));
    let words = "AAAAAAAAAAAAAAAAAAAAAA==";
    let buffer = |binding: u32| serde_json::json!({"binding": binding, "role": "in_out", "initial_bytes_b64": words});
    // The out-of-bounds errors of a case targeting buffer `binding` with `form`.
    let errors_of = |binding: u32, form: ContainmentForm| {
        let case: AuthoredCase = serde_json::from_value(serde_json::json!({
            "air_sha256": "11".repeat(32),
            "case_id": "",
            "name": "atomics",
            "entry": "atomics",
            "stage": "kernel",
            "buffers": [buffer(0), buffer(1), buffer(2)],
            "dispatch": {"grid": [1, 1, 1], "threads_per_threadgroup": [1, 1, 1]},
            "output": {"kind": "buffer", "binding": 2, "offset": 0, "length": 16},
            "compare": {"kind": "exact"},
            "execution_safety": "loop_free",
            "out_of_bounds": {"targets": [
                {"resource": {"kind": "buffer", "binding": binding}, "form": form, "reach_after": 16}
            ]},
        }))
        .unwrap();
        let mut errors = Vec::new();
        conformance::validate(&case, &interface, &mut errors);
        errors.retain(|error| error.starts_with("out_of_bounds"));
        errors.join("\n")
    };
    for binding in [0, 1] {
        assert_eq!(errors_of(binding, ContainmentForm::BufferAtomic), "");
        for form in [
            ContainmentForm::StorageBuffer,
            ContainmentForm::DeviceAddress,
        ] {
            assert!(
                errors_of(binding, form).contains(&format!("form is buffer_atomic, not {form}")),
                "{binding} {form}"
            );
        }
    }
    assert_eq!(errors_of(2, ContainmentForm::StorageBuffer), "");
    assert_eq!(errors_of(2, ContainmentForm::DeviceAddress), "");
    assert!(errors_of(2, ContainmentForm::BufferAtomic).contains("needs an atomic operation"));
}

/// The authorable targets agree with their AIR (over the committed `engine/ob2_*` fixtures): a
/// loaded pointer's buffer is one the entry loads and stores through a pointer held by the argument
/// buffer's member; an argument buffer's texture array is the first element of a texture member of
/// at least two elements whose handles the entry loads at a dynamic offset; and thread memory is
/// the entry's one `alloca`, holding an array.
#[test]
fn the_in_shader_out_of_bounds_targets_follow_their_air() {
    use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsResource};
    let fixture = |name: &str| {
        let path = metal2vulkan_devtools::engine_dir().join(format!("{name}.ll"));
        let text = std::fs::read_to_string(path).unwrap();
        interface::tests::of_text(&text, name, &BTreeMap::new()).unwrap()
    };
    let errors_of = |interface: &interface::AirInterface, resource: OutOfBoundsResource, form| {
        let case: AuthoredCase = serde_json::from_value(serde_json::json!({
            "air_sha256": "11".repeat(32),
            "case_id": "",
            "name": "in-shader",
            "entry": "main",
            "stage": "kernel",
            "buffers": [],
            "dispatch": {"grid": [1, 1, 1], "threads_per_threadgroup": [1, 1, 1]},
            "output": {"kind": "buffer", "binding": 2, "offset": 0, "length": 16},
            "compare": {"kind": "exact"},
            "execution_safety": "loop_free",
            "out_of_bounds": {"targets": [{"resource": resource, "form": form}]},
        }))
        .unwrap();
        let mut errors = Vec::new();
        conformance::validate(&case, interface, &mut errors);
        errors.retain(|error| error.starts_with("out_of_bounds"));
        errors.join("\n")
    };
    let pointer = fixture("ob2_loaded_pointer");
    assert!(pointer.accesses(interface::Root::Loaded {
        param: 1,
        offset: Some(0)
    }));
    let embedded = |field_offset| OutOfBoundsResource::ArgumentBufferBuffer {
        buffer_binding: 1,
        field_offset,
    };
    assert_eq!(
        errors_of(&pointer, embedded(0), ContainmentForm::LoadedPointer),
        ""
    );
    assert!(
        errors_of(&pointer, embedded(8), ContainmentForm::LoadedPointer)
            .contains("binds no member there")
    );
    assert!(errors_of(
        &pointer,
        OutOfBoundsResource::ThreadMemory,
        ContainmentForm::SizedArray
    )
    .contains("exactly one thread-local object"));
    // A load through the pointer alone accesses it, and with neither a load nor a store the
    // target is refused: the fixture's AIR without its store through the pointer, then without
    // its load as well.
    let path = metal2vulkan_devtools::engine_dir().join("ob2_loaded_pointer.ll");
    let text = std::fs::read_to_string(path).unwrap();
    let store = "  store i32 %14, ptr addrspace(1) %20, align 4, !tbaa !35\n";
    let load = "%12 = load i32, ptr addrspace(1) %11, align 4, !tbaa !35";
    assert_eq!(
        (text.matches(store).count(), text.matches(load).count()),
        (1, 1)
    );
    let loads_only = text.replace(store, "");
    let neither = loads_only.replace(load, "%12 = add i32 %9, 0");
    for (ll, accessed) in [(loads_only, true), (neither, false)] {
        let interface =
            interface::tests::of_text(&ll, "ob2_loaded_pointer", &BTreeMap::new()).unwrap();
        assert_eq!(
            interface.accesses(interface::Root::Loaded {
                param: 1,
                offset: Some(0)
            }),
            accessed
        );
        let errors = errors_of(&interface, embedded(0), ContainmentForm::LoadedPointer);
        assert_eq!(
            errors.contains("loads and stores nothing through the pointer"),
            !accessed,
            "{errors}"
        );
    }

    let array = fixture("ob2_sized_array");
    assert_eq!(
        array.thread_objects,
        [interface::ThreadObject { holds_array: true }]
    );
    assert_eq!(
        errors_of(
            &array,
            OutOfBoundsResource::ThreadMemory,
            ContainmentForm::SizedArray
        ),
        ""
    );

    let handles = fixture("ob2_handle_array");
    let texture_array = |field_offset| OutOfBoundsResource::ArgumentBufferTextureArray {
        buffer_binding: 1,
        field_offset,
    };
    for first in [0, 16] {
        assert_eq!(
            errors_of(
                &handles,
                texture_array(first),
                ContainmentForm::HandleArrayIndex
            ),
            ""
        );
    }
    assert!(errors_of(
        &handles,
        texture_array(8),
        ContainmentForm::HandleArrayIndex
    )
    .contains("not the first element"));
    // The loaded pointer's argument buffer holds a buffer, and no handle the entry looks up.
    let errors = errors_of(
        &pointer,
        texture_array(0),
        ContainmentForm::HandleArrayIndex,
    );
    assert!(errors.contains("not the first element"), "{errors}");
    assert!(errors.contains("no AIR texture operation"), "{errors}");
}

/// The out-of-bounds errors `conformance` reports for `case` against `interface`.
fn out_of_bounds_errors(case: &serde_json::Value, interface: &interface::AirInterface) -> String {
    let case: AuthoredCase = serde_json::from_value(case.clone()).unwrap();
    let mut errors = Vec::new();
    conformance::validate(&case, interface, &mut errors);
    errors.retain(|error| error.starts_with("out_of_bounds"));
    errors.join("\n")
}

/// The committed `engine/{name}.ll` fixture's text.
fn ob3_fixture(name: &str) -> String {
    std::fs::read_to_string(metal2vulkan_devtools::engine_dir().join(format!("{name}.ll"))).unwrap()
}

/// An index buffer target's vertex function reads its `[[vertex_id]]` only clamped to a constant:
/// an index read past the buffer is the oracle's shadow, a vertex id far past every buffer, and an
/// unclamped use would carry it into one of Metal's accesses. Over the committed
/// `engine/ob3_index_buffer` fixture, which clamps it (`min(vid, 7u)`), then its AIR reading
/// the id unclamped and clamped to a dynamic bound.
#[test]
fn an_index_buffer_targets_vertex_function_reads_its_vertex_id_clamped() {
    use interface::VertexIdUse;
    let case = serde_json::json!({
        "air_sha256": "11".repeat(32),
        "case_id": "",
        "name": "indexed",
        "entry": "ob3_index_buffer",
        "stage": "vertex",
        "buffers": [{"binding": 0, "role": "output", "initial_bytes_b64": "AAAAAAAAAAA="}],
        "draw": {"primitive": "point", "vertex_start": 0, "vertex_count": 16,
                 "instance_count": 1,
                 "index_buffer": {"index_type": "uint32", "offset": 0,
                                  "bytes_b64": "AAAAAAEAAAACAAAAAwAAAA=="}},
        "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": 8},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
        "out_of_bounds": {"targets": [{"resource": {"kind": "index_buffer"},
                                       "form": "index_buffer", "reach_after": 48}]},
    });
    let text = ob3_fixture("ob3_index_buffer");
    let of =
        |ll: &str| interface::tests::of_text(ll, "ob3_index_buffer", &BTreeMap::new()).unwrap();
    let clamped = of(&text);
    assert_eq!(clamped.vertex_id, VertexIdUse::Clamped);
    assert_eq!(out_of_bounds_errors(&case, &clamped), "");
    let min = "%3 = tail call i32 @air.min.u.i32(i32 %1, i32 7) #2";
    assert_eq!(text.matches(min).count(), 1);
    for unclamped in [
        "%3 = add i32 %1, 0",
        "%3 = tail call i32 @air.min.u.i32(i32 %1, i32 %1) #2",
        "%3 = tail call i32 @air.max.u.i32(i32 %1, i32 7) #2\n  %unused = add i32 0, 0",
    ] {
        let mut ll = text.replace(min, unclamped);
        if unclamped.contains("air.max") {
            ll.push_str("\ndeclare i32 @air.max.u.i32(i32, i32)\n");
        }
        let interface = of(&ll);
        assert_eq!(interface.vertex_id, VertexIdUse::Unclamped, "{unclamped}");
        assert!(out_of_bounds_errors(&case, &interface)
            .contains("other than as an operand of an unsigned min with a constant"));
    }
    // A vertex function that never reads the id needs no clamp.
    let unread = text.replace(min, "%3 = add i32 0, 7");
    assert_eq!(of(&unread).vertex_id, VertexIdUse::Unused);
}

/// An attribute target's out-of-bounds read is judged by the format's zero, which holds because
/// the authored format has the lanes the shader reads, so no fetch fills a component with
/// Vulkan's `(0, 0, 1)`. Over the committed `vertex_narrow_attributes` fixture, whose attribute 1
/// is a `ushort2`: a target there authored with one lane fewer is refused (the attribute's shape
/// rule), and with the AIR's lanes it is accepted. The shape rule passes an attribute whose AIR
/// value is untyped, so a target's attribute must be typed: with the input's AIR value removed, the
/// same case is refused by that rule alone.
#[test]
fn an_attribute_targets_format_has_the_lanes_its_shader_reads() {
    let text = std::fs::read_to_string(
        metal2vulkan_devtools::fixtures_dir().join("vertex_narrow_attributes.ll"),
    )
    .unwrap();
    let interface =
        interface::tests::of_text(&text, "vertex_narrow_attributes", &BTreeMap::new()).unwrap();
    let location = interface
        .params
        .iter()
        .find_map(|param| match param.binding.role {
            metal2vulkan_air::abi::ParamRole::VertexInput { .. }
                if param.value.is_some_and(|shape| shape.lanes == 2) =>
            {
                param.slots().map(|(slot, _)| slot)
            }
            _ => None,
        })
        .expect("a two-lane vertex input");
    let errors_of = |interface: &interface::AirInterface, format: &str| {
        let case: AuthoredCase = serde_json::from_value(serde_json::json!({
            "air_sha256": "11".repeat(32),
            "case_id": "",
            "name": "narrow",
            "entry": "vertex_narrow_attributes",
            "stage": "vertex",
            "buffers": [],
            "vertex_inputs": [{"location": location, "format": format, "stride": 4,
                               "bytes_b64": "AAAAAA=="}],
            "draw": {"primitive": "point", "vertex_start": 0, "vertex_count": 2,
                     "instance_count": 1},
            "output": {"kind": "buffer", "binding": 2, "offset": 0, "length": 4},
            "compare": {"kind": "exact"},
            "execution_safety": "loop_free",
            "out_of_bounds": {"targets": [{"resource": {"kind": "attribute_buffer",
                                                        "location": location},
                                           "form": "vertex_input", "reach_after": 16}]},
        }))
        .unwrap();
        let mut errors = Vec::new();
        conformance::validate(&case, interface, &mut errors);
        errors
            .into_iter()
            .filter(|error| {
                error.contains(&format!("vertex input {location}"))
                    || error.starts_with("out_of_bounds")
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(errors_of(&interface, "ushort2"), "");
    assert!(errors_of(&interface, "ushort").contains("does not match the AIR value type"));
    let mut untyped = interface.clone();
    for param in &mut untyped.params {
        if matches!(
            param.binding.role,
            metal2vulkan_air::abi::ParamRole::VertexInput { .. }
        ) && param.slots().map(|(slot, _)| slot) == Some(location)
        {
            param.value = None;
        }
    }
    assert_eq!(
        errors_of(&untyped, "ushort2"),
        format!(
            "out_of_bounds target attribute buffer {location}: the AIR types no value of the \
             attribute, so its format's lanes cannot be checked against the shader's"
        )
    );
}

/// A tessellation input that is an out-of-bounds target must have a reflected AIR type, for the
/// reason of the vertex input's rule above: the format check passes an input the AIR states no type
/// for. A control-point target of a typed input passes; the same target with the reflected type
/// removed is refused, and an untyped input that is no target is not.
#[test]
fn a_tessellation_targets_input_has_an_air_type() {
    use crate::air::reflect::TessellationAttribute;
    use crate::case::{AttributeFormat, AttributeInput};
    let input = AttributeInput {
        location: 0,
        format: AttributeFormat::Uint,
        stride: 4,
        dynamic_stride: false,
        bytes_b64: "AAAAAAAAAAAAAAAAAAAAAA==".into(),
    };
    let errors_of = |type_name: Option<&str>, targets: &[u32]| {
        let mut errors = Vec::new();
        validate_tessellation_attributes(
            "control-point",
            std::slice::from_ref(&input),
            &[TessellationAttribute {
                location: 0,
                type_name: type_name.map(String::from),
            }],
            (1, 3),
            &targets.iter().copied().collect(),
            &mut errors,
        );
        errors.join("\n")
    };
    assert_eq!(errors_of(Some("uint"), &[0]), "");
    assert_eq!(
        errors_of(None, &[0]),
        "out_of_bounds target attribute buffer 0: the AIR states no type for the tessellation \
         control-point input, so its format's lanes cannot be checked against the shader's"
    );
    // Not a target: its 16 bytes hold the 3 records the draw fetches, and no type is required.
    assert_eq!(errors_of(None, &[]), "");
}

/// A case of the owned fixture `engine/mesh_draws.ll`'s entry `entry` of `stage`, whose draw is
/// `draw`, drawing into two 8x8 `RGBA32Uint` targets (the fragment function's two colors) and
/// selecting the first: with `ctl` (buffer 0, `uint4`) for an object case.
fn mesh_draws_case(
    stage: Stage,
    entry: &str,
    draw: crate::case::MeshDraw,
) -> (AuthoredCase, SourceRow) {
    use crate::case::{
        BufferResource, Comparison, ExecutionSafety, OutputSelection, RenderTargetResource,
        ResourceRole,
    };
    let air_ll = std::fs::read_to_string(metal2vulkan_devtools::engine_dir().join("mesh_draws.ll"))
        .expect("public fixture");
    let air_sha256 = crate::hash::sha256_bytes(air_ll.as_bytes());
    let target = |index| RenderTargetResource {
        index,
        format: TextureFormat::Rgba32Uint,
        dimensions: [8, 8],
        initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode([0u8; 8 * 8 * 16]),
    };
    let mut case = AuthoredCase {
        air_sha256: air_sha256.clone(),
        case_id: String::new(),
        name: format!("{entry}-draw"),
        entry: entry.into(),
        stage,
        buffers: (stage == Stage::Object)
            .then(|| BufferResource {
                binding: 0,
                role: ResourceRole::Input,
                bytes_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode([2u8, 0, 0, 0].repeat(4)),
                ),
                initial_bytes_b64: None,
                rasterization_rate_map: None,
            })
            .into_iter()
            .collect(),
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![target(0), target(1)],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: None,
        draw: None,
        tessellation: None,
        mesh_draw: Some(draw),
        output: OutputSelection::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [8, 8],
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: None,
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    };
    case.case_id = case.computed_case_id().expect("case id");
    let source = SourceRow {
        air_sha256,
        stage: stage.metadata_label().into(),
        entry: entry.into(),
        air_ll,
        blob_b64: None,
        lib_sha256s: Vec::new(),
        label: "public/engine/mesh_draws.ll".into(),
    };
    (case, source)
}

fn mesh_draw(
    object: Option<&str>,
    mesh: Option<&str>,
    fragment: Option<&str>,
) -> crate::case::MeshDraw {
    crate::case::MeshDraw {
        object: object.map(str::to_string),
        mesh: mesh.map(str::to_string),
        fragment: fragment.map(str::to_string),
        grid: crate::case::MeshGrid::Threadgroups([3, 2, 1]),
        threads_per_object_threadgroup: [4, 1, 1],
        threads_per_mesh_threadgroup: [2, 1, 1],
        rasterization: true,
        observation: None,
        object_launch: None,
    }
}

/// The checker's errors for the mesh case `case` over `source`.
fn mesh_case_errors(case: AuthoredCase, source: &SourceRow) -> Vec<String> {
    let scratch = crate::ScratchDir::new("mesh-case-check").expect("scratch");
    match check_case_against_source(scratch.path(), case, source) {
        Ok(_) => Vec::new(),
        Err(errors) => errors,
    }
}

/// The one error of a mesh case whose draw needs a stage it does not name: the evidence gap of
/// `mesh_companion`, which stops the Metal oracle first.
fn assert_only_the_companion_gap(errors: &[String]) {
    let [error] = errors else {
        panic!("{errors:?}");
    };
    assert!(error.contains("cannot evidence the case"), "{error}");
    assert!(
        error.contains("mesh_companion (the Metal oracle cannot run the case)"),
        "{error}"
    );
}

/// A mesh case over the owned fixture checks as Metal would link its pipeline, and both executors
/// draw it: a mesh function with its fragment partner or with the generated observer fragment of
/// its stated observation, and an object function with its mesh and fragment partners (the payload
/// the mesh function reads is the object function's) or with the generated stages of its launch
/// observation, whose observer fragment writes render target 0; and a mesh function whose payload a
/// generated object function writes. An object case that names no mesh partner and states no
/// launch observation needs a stage nothing supplies (`mesh_companion`).
#[test]
fn a_mesh_case_with_its_partners_checks_clean() {
    use base64::Engine as _;
    let (grid, source) = mesh_draws_case(
        Stage::Mesh,
        "mesh_draws_grid",
        mesh_draw(None, None, Some("mesh_draws_fragment")),
    );
    assert_eq!(mesh_case_errors(grid, &source), Vec::<String>::new());
    let (object, source) = mesh_draws_case(
        Stage::Object,
        "mesh_draws_object",
        mesh_draw(None, Some("mesh_draws_mesh"), Some("mesh_draws_fragment")),
    );
    assert_eq!(mesh_case_errors(object, &source), Vec::<String>::new());
    // A mesh case observed through the generated observer fragment in its fragment partner's
    // place.
    let mut observed = mesh_draw(None, None, None);
    observed.observation = Some(crate::case::MeshObservation::PrimitiveData { index: 0 });
    let (observed, source) = mesh_draws_case(Stage::Mesh, "mesh_draws_grid", observed);
    assert_eq!(mesh_case_errors(observed, &source), Vec::<String>::new());
    let mut launched = mesh_draw(None, None, None);
    launched.grid = crate::case::MeshGrid::Threadgroups([1, 1, 1]);
    launched.observation = Some(crate::case::MeshObservation::Launch);
    let (launched, source) = mesh_draws_case(Stage::Object, "mesh_draws_object", launched);
    assert_eq!(mesh_case_errors(launched, &source), Vec::<String>::new());
    // A mesh case whose payload a generated object function writes, in its object partner's
    // place; a launch shorter than the payload the mesh function reads is refused.
    let mut payload = mesh_draw(None, None, Some("mesh_draws_fragment"));
    payload.grid = crate::case::MeshGrid::Threadgroups([1, 1, 1]);
    payload.object_launch = Some(crate::case::ObjectLaunch {
        payload_b64: base64::engine::general_purpose::STANDARD.encode([0u8; 32]),
        mesh_threadgroups: [2, 1, 1],
    });
    let (payload, source) = mesh_draws_case(Stage::Mesh, "mesh_draws_mesh", payload.clone());
    assert_eq!(
        mesh_case_errors(payload.clone(), &source),
        Vec::<String>::new()
    );
    let mut short = payload;
    short.mesh_draw.as_mut().unwrap().object_launch = Some(crate::case::ObjectLaunch {
        payload_b64: base64::engine::general_purpose::STANDARD.encode([0u8; 16]),
        mesh_threadgroups: [2, 1, 1],
    });
    short.case_id = short.computed_case_id().unwrap();
    let errors = mesh_case_errors(short, &source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("payload holds 16 bytes, and the mesh function reads 32")),
        "{errors:?}"
    );
    let (meshless, source) = mesh_draws_case(
        Stage::Object,
        "mesh_draws_object",
        mesh_draw(None, None, Some("mesh_draws_fragment")),
    );
    assert_only_the_companion_gap(&mesh_case_errors(meshless, &source));
}

/// The checker refuses a rasterizing mesh case that names no fragment function, as a schema error
/// before any executor runs: Metal's descriptor validation aborts the process on such a pipeline
/// ("fragmentFunction must not be nil", the M3 Max under macOS 27.0.1), so no executor may reach
/// it.
#[test]
fn a_rasterizing_mesh_case_without_a_fragment_function_never_reaches_an_executor() {
    let (unnamed, source) =
        mesh_draws_case(Stage::Mesh, "mesh_draws_grid", mesh_draw(None, None, None));
    let errors = mesh_case_errors(unnamed, &source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("mesh_draw rasterizes and names no fragment function")),
        "{errors:?}"
    );
}

/// A partner the module declares at another stage, one it does not declare, and one that binds a
/// resource (the case authors only its own entry's) are refused by name.
#[test]
fn a_mesh_case_s_partners_are_checked_against_its_module() {
    let errors = |stage, entry: &str, draw| {
        let (case, source) = mesh_draws_case(stage, entry, draw);
        mesh_case_errors(case, &source)
    };
    let contains = |errors: Vec<String>, what: &str| {
        assert!(
            errors.iter().any(|error| error.contains(what)),
            "{what:?} not among {errors:?}"
        );
    };
    contains(
        errors(
            Stage::Mesh,
            "mesh_draws_grid",
            mesh_draw(None, None, Some("mesh_draws_mesh")),
        ),
        "names \"mesh_draws_mesh\", which the module declares as no fragment function",
    );
    contains(
        errors(
            Stage::Mesh,
            "mesh_draws_mesh",
            mesh_draw(Some("mesh_draws_object"), None, Some("mesh_draws_fragment")),
        ),
        "mesh_draw.object \"mesh_draws_object\" binds resources",
    );
    contains(
        errors(
            Stage::Mesh,
            "mesh_draws_grid",
            mesh_draw(None, None, Some("nowhere")),
        ),
        "names \"nowhere\", which the module declares as no fragment function",
    );
}
