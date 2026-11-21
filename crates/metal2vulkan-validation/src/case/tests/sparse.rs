use super::*;
use crate::case::sparse::{SparseMapping, SparseTextureResource};

/// [`example`] reading a sparse 256x128 `r32_float` 2D texture at binding 0, its left half mapped.
fn sparse_case(mappings: Vec<SparseMapping>) -> AuthoredCase {
    let mut case = example();
    case.textures.push(TextureResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Float,
        dimensions: [256, 128, 1],
        sample_count: 1,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode(vec![0u8; 4 * 256 * 128])),
        initial_bytes_b64: None,
        sparse: Some(SparseTextureResource { mappings }),
        mip_levels: 1,
    });
    case.case_id = case.computed_case_id().unwrap();
    case
}

fn region(level: u32, slice: u32, origin: [u32; 3], size: [u32; 3]) -> SparseMapping {
    SparseMapping::Region {
        level,
        slice,
        origin,
        size,
    }
}

fn errors(case: &AuthoredCase) -> String {
    case.validate_literal_resources()
        .err()
        .unwrap_or_default()
        .join("\n")
}

/// A sparse texture is part of the identity and the runtime's limit, which stops the engine's
/// candidate alone; a texture without the field serializes as before it existed.
#[test]
fn a_sparse_texture_is_identity_and_the_runtimes_limit() {
    let case = sparse_case(vec![region(0, 0, [0, 0, 0], [128, 128, 1])]);
    assert_eq!(errors(&case), "");
    let mut ordinary = case.clone();
    ordinary.textures[0].sparse = None;
    assert_ne!(ordinary.computed_case_id().unwrap(), case.case_id);
    assert!(!serde_json::to_string(&ordinary).unwrap().contains("sparse"));
    let mut unmapped = case.clone();
    unmapped.textures[0].sparse = Some(SparseTextureResource { mappings: vec![] });
    assert_eq!(errors(&unmapped), "", "every texel may be unmapped");
    assert_ne!(unmapped.computed_case_id().unwrap(), case.case_id);
    assert_eq!(
        crate::executor_contract::case_requirements(&case),
        [crate::requirement::ToolingRequirement::SparseTexture].into()
    );
    assert!(crate::executor_contract::case_requirements(&ordinary).is_empty());
    // The schema's spelling.
    let json = serde_json::to_value(&case.textures[0]).unwrap();
    assert_eq!(
        json["sparse"],
        serde_json::json!({"mappings": [
            {"kind": "region", "level": 0, "slice": 0, "origin": [0, 0, 0], "size": [128, 128, 1]}
        ]})
    );
    let tail: SparseMapping =
        serde_json::from_value(serde_json::json!({"kind": "mip_tail", "slice": 0})).unwrap();
    assert_eq!(tail, SparseMapping::MipTail { slice: 0 });
}

/// The checker refuses a mapping no executor can make whatever its tiles: outside the level, empty,
/// a level or slice the texture lacks, overlapping another, or a slice's tail mapped twice.
#[test]
fn a_mapping_no_device_can_make_is_refused() {
    for (mapping, expected) in [
        (
            region(0, 0, [128, 0, 0], [256, 128, 1]),
            "leave the level's",
        ),
        (region(0, 0, [0, 0, 0], [0, 128, 1]), "an empty region"),
        (
            region(1, 0, [0, 0, 0], [64, 64, 1]),
            "level 1, but the texture has 1 level",
        ),
        (
            region(0, 1, [0, 0, 0], [64, 64, 1]),
            "slice 1, but the texture has 1 slice",
        ),
        (
            SparseMapping::MipTail { slice: 1 },
            "slice 1, but the texture has 1 slice",
        ),
    ] {
        let error = errors(&sparse_case(vec![mapping]));
        assert!(error.contains(expected), "{mapping:?}: {error}");
    }
    let overlap = sparse_case(vec![
        region(0, 0, [0, 0, 0], [128, 128, 1]),
        region(0, 0, [128, 0, 0], [64, 64, 1]),
        region(0, 0, [64, 64, 0], [128, 64, 1]),
    ]);
    let error = errors(&overlap);
    assert!(
        error.contains("mapping 2: overlaps mapping 0's region"),
        "{error}"
    );
    let disjoint = sparse_case(vec![
        region(0, 0, [0, 0, 0], [128, 128, 1]),
        region(0, 0, [128, 0, 0], [128, 64, 1]),
    ]);
    assert_eq!(errors(&disjoint), "");
    let tails = sparse_case(vec![
        SparseMapping::MipTail { slice: 0 },
        SparseMapping::MipTail { slice: 0 },
    ]);
    assert!(errors(&tails).contains("slice 0's mip tail is mapped twice"));
}

/// A mip tail is its slice's one level whole, so the checker refuses a region of the same slice
/// beside it, before or after it, whatever a device's tail holds; another slice's region may stand
/// beside it.
#[test]
fn a_region_beside_its_slices_mip_tail_is_refused() {
    for mappings in [
        vec![
            SparseMapping::MipTail { slice: 0 },
            region(0, 0, [0, 0, 0], [128, 128, 1]),
        ],
        vec![
            region(0, 0, [128, 0, 0], [128, 128, 1]),
            SparseMapping::MipTail { slice: 0 },
        ],
    ] {
        let error = errors(&sparse_case(mappings.clone()));
        assert!(
            error.contains("a region of slice 0, whose mip tail is mapped"),
            "{mappings:?}: {error}"
        );
    }
    let mut array = sparse_case(vec![
        SparseMapping::MipTail { slice: 1 },
        region(0, 0, [0, 0, 0], [256, 64, 1]),
    ]);
    array.textures[0].texture_type = TextureType::D2Array;
    array.textures[0].dimensions = [256, 64, 2];
    assert_eq!(errors(&array), "");
}

/// The case model makes 2D, 2D array, and 3D textures sparse, with one sample, a color format, and
/// the input role; a 2D array's slices and a 3D texture's depth bound its regions.
#[test]
fn only_a_single_sample_color_input_texture_of_a_sparse_type_is_sparse() {
    let refused = |edit: &dyn Fn(&mut TextureResource), expected: &str| {
        let mut case = sparse_case(vec![]);
        edit(&mut case.textures[0]);
        let error = errors(&case);
        assert!(error.contains(expected), "{expected}: {error}");
    };
    refused(
        &|texture| {
            texture.texture_type = TextureType::Cube;
            texture.dimensions = [128, 128, 6];
        },
        "a sparse texture is 2D, 2D array, or 3D",
    );
    refused(
        &|texture| texture.role = ResourceRole::InOut,
        "a sparse texture is an input",
    );
    refused(
        &|texture| texture.format = TextureFormat::Depth32Float,
        "depth formats are not modelled",
    );
    refused(
        &|texture| {
            texture.texture_type = TextureType::D2Multisample;
            texture.sample_count = 4;
            texture.bytes_b64 = Some(
                base64::engine::general_purpose::STANDARD.encode(vec![0u8; 4 * 4 * 256 * 128]),
            );
        },
        "a sparse texture is 2D, 2D array, or 3D",
    );
    let mut array = sparse_case(vec![
        region(0, 1, [0, 0, 0], [256, 128, 1]),
        region(0, 0, [0, 0, 0], [256, 128, 1]),
    ]);
    array.textures[0].texture_type = TextureType::D2Array;
    array.textures[0].dimensions = [256, 64, 2];
    let error = errors(&array);
    assert!(error.contains("leave the level's [256, 64, 1]"), "{error}");
    array.textures[0].sparse = Some(SparseTextureResource {
        mappings: vec![
            region(0, 1, [0, 0, 0], [256, 64, 1]),
            region(0, 0, [0, 0, 0], [256, 64, 1]),
        ],
    });
    assert_eq!(
        errors(&array),
        "",
        "the same texels of two slices do not overlap"
    );
    let mut volume = sparse_case(vec![region(0, 0, [0, 0, 8], [32, 32, 8])]);
    volume.textures[0].texture_type = TextureType::D3;
    volume.textures[0].dimensions = [32, 32, 16];
    volume.textures[0].bytes_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode(vec![0u8; 4 * 32 * 32 * 16]));
    assert_eq!(errors(&volume), "");
}
