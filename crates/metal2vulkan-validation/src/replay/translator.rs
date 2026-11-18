//! What a call describes in the translator's types, which every backend takes: a sampler
//! descriptor, function constant values, a depth-stencil descriptor and a render pipeline
//! descriptor; and the keys of a render pass descriptor's setter records.

use super::args::{metal, narrow, refuse, Fields};
use super::player::PlayError;
use super::stream::Call;
use super::value::Value;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};

/// `newSamplerStateWithDescriptor:`'s descriptor, `[map]`, as the translator's: every key but
/// `borderColor`, `supportArgumentBuffers` and `resourceIndex`, which the backend reads from the
/// fields it returns.
pub(super) fn sampler(call: &Call) -> Result<(SamplerDescriptor, Fields<'_>), PlayError> {
    let known = [
        "minFilter",
        "magFilter",
        "lodAverage",
        "compareFunction",
        "sAddressMode",
        "tAddressMode",
        "rAddressMode",
        "mipFilter",
        "maxAnisotropy",
        "normalizedCoordinates",
        "supportArgumentBuffers",
        "reductionMode",
        "lodMinClamp",
        "lodMaxClamp",
        "lodBias",
        "borderColor",
        "resourceIndex",
    ];
    let f = Fields::arg(call, 0, &known)?;
    let mut s = SamplerDescriptor::new();
    if let Some(filter) = f.metal("minFilter")? {
        s.min_filter = filter;
    }
    if let Some(filter) = f.metal("magFilter")? {
        s.mag_filter = filter;
    }
    if let Some(average) = f.bool("lodAverage")? {
        s.lod_average = average;
    }
    if let Some(compare) = f.metal("compareFunction")? {
        s.compare_function = compare;
    }
    if let Some(mode) = f.metal("sAddressMode")? {
        s.s_address_mode = mode;
    }
    if let Some(mode) = f.metal("tAddressMode")? {
        s.t_address_mode = mode;
    }
    if let Some(mode) = f.metal("rAddressMode")? {
        s.r_address_mode = mode;
    }
    if let Some(filter) = f.metal("mipFilter")? {
        s.mip_filter = filter;
    }
    if let Some(anisotropy) = f.count("maxAnisotropy")? {
        s.max_anisotropy = anisotropy;
    }
    if let Some(normalized) = f.bool("normalizedCoordinates")? {
        s.normalized_coordinates = normalized;
    }
    if let Some(mode) = f.metal("reductionMode")? {
        s.reduction_mode = mode;
    }
    let lods: [(&str, &mut f32); 3] = [
        ("lodMinClamp", &mut s.lod_min_clamp),
        ("lodMaxClamp", &mut s.lod_max_clamp),
        ("lodBias", &mut s.lod_bias),
    ];
    for (key, lod) in lods {
        if let Some(value) = f.float(key)? {
            *lod = value as f32;
        }
    }
    Ok((s, f))
}

/// The packed bytes of a function constant of `MTLDataType` `raw`: its lanes' size times their
/// count (a `float3` is 12 bytes); none for a type no function constant takes.
fn constant_bytes(raw: u64) -> Option<usize> {
    // Each family's scalar, then its two-, three- and four-lane vectors, numbered in a row.
    let (lane, scalar) = match raw {
        3..=6 => (4, 3),
        16..=19 => (2, 16),
        29..=32 => (4, 29),
        33..=36 => (4, 33),
        37..=40 => (2, 37),
        41..=44 => (2, 41),
        45..=48 => (1, 45),
        49..=52 => (1, 49),
        53..=56 => (1, 53),
        81..=84 => (8, 81),
        85..=88 => (8, 85),
        _ => return None,
    };
    Some(lane * usize::try_from(raw - scalar + 1).ok()?)
}

/// The function constants of `list`, each `{index, dataType, value}`, the value its bytes packed
/// from the first (the rest zero).
pub(super) fn constants(call: &Call, list: &[Value]) -> Result<FunctionConstantValues, PlayError> {
    let mut values = FunctionConstantValues::defaults();
    for constant in list {
        let f = Fields::of(call, constant, &["index", "dataType", "value"])?;
        let index = f.required("index", Fields::count)?;
        let raw = f.required("dataType", Fields::uint)?;
        let ty = metal(call, raw)?;
        let bytes = f.required("value", Fields::list)?;
        let bytes: Option<Vec<u8>> = bytes
            .iter()
            .map(|byte| byte.as_uint().and_then(|byte| u8::try_from(byte).ok()))
            .collect();
        let bytes = bytes.ok_or_else(|| refuse(call, format!("constant {index}'s value")))?;
        let size = constant_bytes(raw)
            .filter(|&size| size <= bytes.len())
            .ok_or_else(|| refuse(call, format!("constant {index} of data type {raw}")))?;
        let value = FunctionConstantValue::new(ty, &bytes[..size]);
        values = values.with_index(index, value);
    }
    Ok(values)
}

const STENCIL_KEYS: [&str; 6] = [
    "stencilCompareFunction",
    "stencilFailureOperation",
    "depthFailureOperation",
    "depthStencilPassOperation",
    "readMask",
    "writeMask",
];

fn stencil(f: &Fields) -> Result<StencilDescriptor, PlayError> {
    let mut s = StencilDescriptor::default();
    s.stencil_compare_function = f.required("stencilCompareFunction", Fields::metal)?;
    s.stencil_failure_operation = f.required("stencilFailureOperation", Fields::metal)?;
    s.depth_failure_operation = f.required("depthFailureOperation", Fields::metal)?;
    s.depth_stencil_pass_operation = f.required("depthStencilPassOperation", Fields::metal)?;
    s.read_mask = f.required("readMask", Fields::count)?;
    s.write_mask = f.required("writeMask", Fields::count)?;
    Ok(s)
}

/// `newDepthStencilStateWithDescriptor:`.
pub(super) fn depth_stencil(call: &Call) -> Result<DepthStencilDescriptor, PlayError> {
    let known = ["depthCompareFunction", "depthWriteEnabled", "front", "back"];
    let f = Fields::arg(call, 0, &known)?;
    let mut d = DepthStencilDescriptor::default();
    if let Some(compare) = f.metal("depthCompareFunction")? {
        d.depth_compare_function = compare;
    }
    if let Some(enabled) = f.bool("depthWriteEnabled")? {
        d.depth_write_enabled = enabled;
    }
    if let Some(front) = f.fields("front", &STENCIL_KEYS)? {
        d.front_face_stencil = stencil(&front)?;
    }
    if let Some(back) = f.fields("back", &STENCIL_KEYS)? {
        d.back_face_stencil = stencil(&back)?;
    }
    Ok(d)
}

pub(super) const LINKED_KEYS: [&str; 4] =
    ["functions", "binaryFunctions", "groups", "privateFunctions"];

const COLOR_KEYS: [&str; 9] = [
    "pixelFormat",
    "blendingEnabled",
    "sourceRgbBlendFactor",
    "destinationRgbBlendFactor",
    "rgbBlendOperation",
    "sourceAlphaBlendFactor",
    "destinationAlphaBlendFactor",
    "alphaBlendOperation",
    "writeMask",
];

/// A pipeline's color attachment: none for `MTLPixelFormatInvalid`.
fn color_attachment(f: &Fields) -> Result<Option<ColorAttachmentDescriptor>, PlayError> {
    let format: PixelFormat = f.required("pixelFormat", Fields::metal)?;
    if format == PixelFormat::Invalid {
        return Ok(None);
    }
    let mut a = ColorAttachmentDescriptor::new(format);
    a.blending_enabled = f.required("blendingEnabled", Fields::bool)?;
    a.source_rgb_blend_factor = f.required("sourceRgbBlendFactor", Fields::metal)?;
    a.destination_rgb_blend_factor = f.required("destinationRgbBlendFactor", Fields::metal)?;
    a.rgb_blend_operation = f.required("rgbBlendOperation", Fields::metal)?;
    a.source_alpha_blend_factor = f.required("sourceAlphaBlendFactor", Fields::metal)?;
    a.destination_alpha_blend_factor = f.required("destinationAlphaBlendFactor", Fields::metal)?;
    a.alpha_blend_operation = f.required("alphaBlendOperation", Fields::metal)?;
    a.write_mask = f.required("writeMask", Fields::metal)?;
    Ok(Some(a))
}

/// `MTLBufferLayoutStrideDynamic`, a stride set per bind.
const DYNAMIC_STRIDE: u64 = u64::MAX;

/// A vertex descriptor, `{layouts, attributes}`: each at its index, and a layout of stride 0, as
/// Metal's array holds one, for each buffer an attribute reads that no layout names.
fn vertex_descriptor(call: &Call, f: &Fields) -> Result<VertexDescriptor, PlayError> {
    let mut out = VertexDescriptor::new();
    for layout in f.list("layouts")?.unwrap_or_default() {
        let known = ["bufferIndex", "stride", "stepFunction", "stepRate"];
        let l = Fields::of(call, layout, &known)?;
        let stride = l.required("stride", Fields::uint)?;
        let stride = match stride {
            DYNAMIC_STRIDE => None,
            stride => Some(narrow(call, stride, "stride")?),
        };
        let mut made = VertexBufferLayoutDescriptor::new(stride);
        made.step_function = l.required("stepFunction", Fields::metal)?;
        made.step_rate = l.required("stepRate", Fields::count)?;
        out.layouts
            .insert(l.required("bufferIndex", Fields::count)?, made);
    }
    for attribute in f.list("attributes")?.unwrap_or_default() {
        let known = ["attributeIndex", "format", "offset", "bufferIndex"];
        let a = Fields::of(call, attribute, &known)?;
        let made = VertexAttributeDescriptor::new(
            a.required("format", Fields::metal)?,
            a.required("offset", Fields::count)?,
            a.required("bufferIndex", Fields::count)?,
        );
        out.attributes
            .insert(a.required("attributeIndex", Fields::count)?, made);
    }
    let read: Vec<u32> = out.attributes.values().map(|a| a.buffer_index).collect();
    for buffer in read {
        out.layouts
            .entry(buffer)
            .or_insert_with(|| VertexBufferLayoutDescriptor::new(Some(0)));
    }
    Ok(out)
}

const PRIVATE_KEYS: [&str; 7] = [
    "openGlModeEnabled",
    "sampleMask",
    "sampleCoverage",
    "sampleCoverageInvert",
    "logicOperationEnabled",
    "logicOperation",
    "depthStencilWriteDisabled",
];

pub(super) const RENDER_PIPELINE_KEYS: [&str; 19] = [
    "label",
    "vertex",
    "fragment",
    "vertexDescriptor",
    "colors",
    "color0PixelFormat",
    "resourceIndex",
    "depthPixelFormat",
    "stencilPixelFormat",
    "rasterSampleCount",
    "alphaToCoverage",
    "alphaToOne",
    "rasterizationEnabled",
    "inputPrimitiveTopology",
    "maxVertexAmplificationCount",
    "supportIndirectCommandBuffers",
    "vertexLinked",
    "fragmentLinked",
    "privateState",
];

/// `newRenderPipelineStateWithDescriptor:error:`'s descriptor, `[map]` (the whole descriptor's map,
/// or the short one of a vertex and fragment function and color 0's format), as the translator's:
/// each stage of `stage`, which makes the stage of a function key and its linked functions' key.
/// `label` and `resourceIndex` are the backend's to read from the fields it returns.
pub(super) fn render_pipeline<'c>(
    call: &'c Call,
    stage: impl Fn(&Fields, &str, &str) -> Result<Option<StageSource>, PlayError>,
) -> Result<(RenderPipelineDescriptor, Fields<'c>), PlayError> {
    let f = Fields::arg(call, 0, &RENDER_PIPELINE_KEYS)?;
    let vertex = stage(&f, "vertex", "vertexLinked")?;
    let vertex = vertex.ok_or_else(|| refuse(call, "no vertex function"))?;
    let mut d = RenderPipelineDescriptor::new(vertex);
    d.fragment = stage(&f, "fragment", "fragmentLinked")?;
    if let Some(v) = f.fields("vertexDescriptor", &["layouts", "attributes"])? {
        d.vertex_descriptor = Some(vertex_descriptor(call, &v)?);
    }
    for color in f.list("colors")?.unwrap_or_default() {
        let Some([index, color]) = color.as_list() else {
            return Err(refuse(call, format!("{color} is no [index, attachment]")));
        };
        let slot = index.as_uint().and_then(|index| {
            usize::try_from(index)
                .ok()
                .and_then(|index| d.color_attachments.get_mut(index))
        });
        let slot = slot.ok_or_else(|| refuse(call, format!("color attachment {index}")))?;
        *slot = color_attachment(&Fields::of(call, color, &COLOR_KEYS)?)?;
    }
    if let Some(format) = f.metal::<PixelFormat>("color0PixelFormat")? {
        d.color_attachments[0] =
            (format != PixelFormat::Invalid).then(|| ColorAttachmentDescriptor::new(format));
    }
    let formats = [
        ("depthPixelFormat", &mut d.depth_format),
        ("stencilPixelFormat", &mut d.stencil_format),
    ];
    for (key, format) in formats {
        if let Some(raw) = f.metal::<PixelFormat>(key)? {
            *format = (raw != PixelFormat::Invalid).then_some(raw);
        }
    }
    if let Some(count) = f.count("rasterSampleCount")? {
        d.raster_sample_count = count;
    }
    let switches = [
        ("alphaToCoverage", &mut d.alpha_to_coverage),
        ("alphaToOne", &mut d.alpha_to_one),
        ("rasterizationEnabled", &mut d.rasterization_enabled),
        (
            "supportIndirectCommandBuffers",
            &mut d.support_indirect_command_buffers,
        ),
    ];
    for (key, switch) in switches {
        if let Some(on) = f.bool(key)? {
            *switch = on;
        }
    }
    if let Some(class) = f.metal("inputPrimitiveTopology")? {
        d.input_primitive_topology = class;
    }
    if let Some(count) = f.count("maxVertexAmplificationCount")? {
        d.max_vertex_amplification_count = count;
    }
    if let Some(p) = f.fields("privateState", &PRIVATE_KEYS)? {
        let private = &mut d.private;
        private.open_gl_mode_enabled = p.required("openGlModeEnabled", Fields::bool)?;
        private.sample_mask = p.required("sampleMask", Fields::uint)?;
        private.sample_coverage = p.required("sampleCoverage", Fields::float)? as f32;
        private.sample_coverage_invert = p.required("sampleCoverageInvert", Fields::bool)?;
        private.logic_operation_enabled = p.required("logicOperationEnabled", Fields::bool)?;
        private.logic_operation = p.required("logicOperation", Fields::uint)?;
        private.depth_stencil_write_disabled =
            p.required("depthStencilWriteDisabled", Fields::bool)?;
    }
    Ok((d, f))
}

/// The keys of `m2v.setPass`'s map, a render pass descriptor's own properties.
pub(super) const PASS_KEYS: [&str; 12] = [
    "visibilityResultBuffer",
    "rasterizationRateMap",
    "visibilityResultType",
    "renderTargetArrayLength",
    "renderTargetWidth",
    "renderTargetHeight",
    "defaultRasterSampleCount",
    "defaultColorSampleCount",
    "imageblockSampleLength",
    "threadgroupMemoryLength",
    "tileWidth",
    "tileHeight",
];

/// The keys every attachment's map holds.
pub(super) const ATTACHMENT_KEYS: [&str; 11] = [
    "texture",
    "resolveTexture",
    "level",
    "slice",
    "depthPlane",
    "resolveLevel",
    "resolveSlice",
    "resolveDepthPlane",
    "loadAction",
    "storeAction",
    "storeActionOptions",
];

/// The keys of one attachment's map: [`ATTACHMENT_KEYS`] and its own.
pub(super) fn attachment_keys(own: &[&'static str]) -> Vec<&'static str> {
    ATTACHMENT_KEYS.iter().chain(own).copied().collect()
}
