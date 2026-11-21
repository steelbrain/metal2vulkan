//! A render pass's rasterization-rate map: the schema, its identity, the checks, the literal bytes
//! a Vulkan executor binds beside it, and the requirement it carries.

use super::*;
use crate::executor_contract::case_requirements;

fn map(screen_size: [u32; 2], horizontal: &[f32], vertical: &[f32]) -> RateMapLiteral {
    RateMapLiteral {
        screen_size,
        layers: vec![RateMapLayerLiteral {
            horizontal: horizontal.iter().copied().map(RateSample).collect(),
            vertical: vertical.iter().copied().map(RateSample).collect(),
        }],
    }
}

/// A fragment case drawing into one `rgba32_float` target of `dimensions`, with the screen size
/// at buffer 0, in a pass with no rate map.
fn fragment_case(dimensions: [u32; 2]) -> AuthoredCase {
    let mut case = example();
    case.stage = Stage::Fragment;
    case.dispatch = None;
    case.draw = Some(Draw {
        primitive: Primitive::Triangle,
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    let screen = [dimensions[0] as f32, dimensions[1] as f32];
    let bytes: Vec<u8> = screen.iter().flat_map(|v| v.to_le_bytes()).collect();
    case.buffers = vec![BufferResource {
        binding: 0,
        role: ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    }];
    let texels = dimensions[0] as usize * dimensions[1] as usize * 16;
    case.render_targets = vec![RenderTargetResource {
        index: 0,
        format: TextureFormat::Rgba32Float,
        dimensions,
        initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode(vec![0xab; texels]),
    }];
    case.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions,
    };
    case.case_id = case.computed_case_id().unwrap();
    case
}

fn errors(case: &AuthoredCase) -> Vec<String> {
    let mut case = case.clone();
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().err().unwrap_or_default()
}

/// The field is part of the identity only when present, and a case carrying it validates in a
/// rasterizing render case whose attachments hold the map's screen.
#[test]
fn a_render_pass_carries_a_rate_map() {
    let plain = fragment_case([37, 23]);
    let text = serde_json::to_string(&plain).unwrap();
    assert!(!text.contains("render_pass_rate_map"), "{text}");
    let mut mapped = plain.clone();
    mapped.render_pass_rate_map = Some(map([37, 23], &[0.5, 1.0], &[1.0, 0.5]));
    mapped.case_id = mapped.computed_case_id().unwrap();
    assert_ne!(mapped.case_id, plain.case_id);
    assert_ne!(
        mapped.computed_input_sha256().unwrap(),
        plain.computed_input_sha256().unwrap()
    );
    assert_eq!(mapped.validate_literal_resources(), Ok(()));
    let text = serde_json::to_string(&mapped).unwrap();
    assert_eq!(serde_json::from_str::<AuthoredCase>(&text).unwrap(), mapped);
    // A larger target holds the screen too.
    let mut larger = fragment_case([40, 24]);
    larger.render_pass_rate_map = mapped.render_pass_rate_map.clone();
    assert_eq!(errors(&larger), Vec::<String>::new());
}

/// A pass map is refused outside a rasterizing render case, over an attachment that does not
/// hold its screen, with tessellation or an imageblock, with more than one layer, and when the
/// literal itself is malformed.
#[test]
fn a_render_pass_rate_map_is_checked() {
    let mapped = |dimensions| {
        let mut case = fragment_case(dimensions);
        case.render_pass_rate_map = Some(map([37, 23], &[0.5, 1.0], &[1.0, 0.5]));
        case
    };
    let has = |case: &AuthoredCase, what: &str| {
        let errors = errors(case);
        assert!(
            errors.iter().any(|error| error.contains(what)),
            "{what}: {errors:?}"
        );
    };
    has(&mapped([36, 23]), "do not hold the rate map's screen");
    let mut depth = mapped([37, 23]);
    depth.depth_stencil = Some(DepthStencilResource {
        dimensions: [37, 22],
        initial_depth_b64: Some(
            base64::engine::general_purpose::STANDARD.encode(vec![0u8; 37 * 22 * 4]),
        ),
        initial_stencil_b64: None,
        state: None,
    });
    has(&depth, "depth/stencil dimensions [37, 22] do not hold");
    let mut kernel = example();
    kernel.render_pass_rate_map = Some(map([4, 4], &[1.0], &[1.0]));
    has(
        &kernel,
        "only for a fragment case or an observed vertex case",
    );
    let mut unobserved = mapped([37, 23]);
    unobserved.stage = Stage::Vertex;
    unobserved.render_targets.clear();
    unobserved.output = OutputSelection::Buffer {
        binding: 0,
        offset: 0,
        length: 4,
    };
    has(
        &unobserved,
        "only for a fragment case or an observed vertex case",
    );
    let mut layered = mapped([37, 23]);
    let layer = layered.render_pass_rate_map.as_ref().unwrap().layers[0].clone();
    layered
        .render_pass_rate_map
        .as_mut()
        .unwrap()
        .layers
        .push(layer);
    has(&layered, "a layered rate-mapped pass is not modelled");
    let mut malformed = mapped([37, 23]);
    malformed.render_pass_rate_map.as_mut().unwrap().layers[0].horizontal[0] = RateSample(2.0);
    has(
        &malformed,
        "render_pass_rate_map layer 0 axis 0 has a sample outside [0, 1]",
    );
    let mut imageblock = mapped([37, 23]);
    imageblock.fragment_imageblock = Some(FragmentImageblockResource {
        dimensions: [1, 1],
        members: vec![],
    });
    has(
        &imageblock,
        "not modelled with tessellation or a fragment imageblock",
    );
}

/// A pass with neither a render target nor a depth/stencil attachment has no raster for its
/// map: Metal would draw a default-sized raster under a map of another screen, which no
/// executor reproduces, so the checker refuses it.
#[test]
fn a_rate_mapped_pass_needs_an_attachment() {
    let mut case = fragment_case([64, 64]);
    case.render_targets.clear();
    case.buffers.push(BufferResource {
        binding: 1,
        role: ResourceRole::Output,
        bytes_b64: None,
        initial_bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 8])),
        rasterization_rate_map: None,
    });
    case.output = OutputSelection::Buffer {
        binding: 1,
        offset: 0,
        length: 8,
    };
    assert_eq!(errors(&case), Vec::<String>::new());
    case.render_pass_rate_map = Some(map([64, 64], &[1.0], &[1.0]));
    assert_eq!(
        errors(&case),
        [
            "render_pass_rate_map needs a render target or a depth/stencil attachment: a pass \
          without one has no raster for the map"
        ]
    );
    // A depth/stencil attachment alone is a raster.
    case.depth_stencil = Some(DepthStencilResource {
        dimensions: [64, 64],
        initial_depth_b64: Some(
            base64::engine::general_purpose::STANDARD.encode(vec![0u8; 64 * 64 * 4]),
        ),
        initial_stencil_b64: None,
        state: None,
    });
    assert_eq!(errors(&case), Vec::<String>::new());
}

/// A Vulkan executor binds every map literal of a case as the product's compile of it, the map
/// the region path draws its pass with, whether or not the case's pass carries a map: the identity
/// of the option (b) is never bound.
#[test]
fn every_map_of_a_case_binds_the_products_compile() {
    let skewed = map([37, 23], &[0.5, 1.0], &[1.0, 0.5]);
    let mut case = fragment_case([37, 23]);
    case.buffers.push(BufferResource {
        binding: 1,
        role: ResourceRole::Input,
        bytes_b64: None,
        initial_bytes_b64: None,
        rasterization_rate_map: Some(skewed.clone()),
    });
    let rule = metal2vulkan::bind::RasterizationRateMap::new(&skewed.descriptor()).unwrap();
    // Quality 1 compiles to the identity.
    let identity = map([37, 23], &[1.0], &[1.0]).compiled().unwrap();
    assert!(identity.is_identity());
    assert_ne!(rule.to_bytes(), identity.to_bytes());
    let bound = |case: &AuthoredCase| {
        crate::literal::LiteralResources::prepare(case)
            .unwrap()
            .buffers[1]
            .bytes
            .clone()
    };
    assert_eq!(bound(&case), rule.to_bytes());
    case.render_pass_rate_map = Some(skewed.clone());
    assert_eq!(bound(&case), rule.to_bytes());
    assert_eq!(skewed.compiled().unwrap(), rule);
}

/// A rate-mapped pass is the runtime's limit: the Metal oracle renders it, and the engine's
/// candidate alone stops.
#[test]
fn a_rate_mapped_pass_is_the_runtimes_limit() {
    let mut case = fragment_case([37, 23]);
    assert!(case_requirements(&case).is_empty());
    case.render_pass_rate_map = Some(map([37, 23], &[1.0], &[1.0]));
    assert_eq!(
        case_requirements(&case),
        [crate::requirement::ToolingRequirement::RasterizationRateMap].into()
    );
}
