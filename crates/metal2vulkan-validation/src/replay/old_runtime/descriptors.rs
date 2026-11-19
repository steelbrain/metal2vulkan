//! The descriptors of a stream's calls as the old runtime's (`docs/REPLAY-STREAM.md`, section 9):
//! each a recorder struct's map, nil for a property it leaves unset, applied as the host's
//! platform applies it to the runtime's descriptor.

use super::args::{metal, narrow, option_sets, options, refuse, Fields};
use super::{placement, Object, OldRuntime};
use crate::replay::player::PlayError;
use crate::replay::stream::Call;
use crate::replay::translator::{self, attachment_keys, LINKED_KEYS, PASS_KEYS};
use crate::replay::value::Value;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, TextureSwizzleChannels};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, LinkedFunctions, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    sampler_border_color, ClearColor, FunctionDescriptor, HeapDescriptor, HeapType, IoSurfacePlane,
    IoSurfaceYCbCr, LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter,
    PipelineOptions, ResourceOptions, SamplerStateOptions, StoreAction, StoreActionOptions,
    SurfaceYCbCrMatrix, Texture, TextureDescriptor, MAX_COLOR_ATTACHMENTS,
};

/// The keys of a texture descriptor: `m2v.textureDescriptor`'s, and a drawable's
/// (`storageMode`).
pub(super) const TEXTURE_KEYS: [&str; 20] = [
    "textureType",
    "pixelFormat",
    "width",
    "height",
    "depth",
    "mipmapLevelCount",
    "sampleCount",
    "arrayLength",
    "usage",
    "resourceOptions",
    "storageMode",
    "swizzle",
    "resourceIndex",
    "protectionOptions",
    "framebufferOnly",
    "isDrawable",
    "writeSwizzleEnabled",
    "allowGpuOptimizedContents",
    "rotation",
    "outside",
];

/// The keys of an `MTLTextureSwizzleChannels`.
pub(super) const SWIZZLE_KEYS: [&str; 4] = ["red", "green", "blue", "alpha"];

/// The `MTLTextureSwizzleChannels` `s` holds, each channel set.
pub(super) fn swizzle(s: &Fields) -> Result<TextureSwizzleChannels, PlayError> {
    Ok(TextureSwizzleChannels {
        red: s.required("red", Fields::metal)?,
        green: s.required("green", Fields::metal)?,
        blue: s.required("blue", Fields::metal)?,
        alpha: s.required("alpha", Fields::metal)?,
    })
}

/// `f`'s properties set on `d`, in the map's order.
pub(super) fn texture(call: &Call, f: &Fields, d: &mut TextureDescriptor) -> Result<(), PlayError> {
    if let Some(kind) = f.metal("textureType")? {
        d.texture_type = kind;
    }
    if let Some(format) = f.metal("pixelFormat")? {
        d.pixel_format = format;
    }
    let sizes: [(&str, &mut u64); 6] = [
        ("width", &mut d.width),
        ("height", &mut d.height),
        ("depth", &mut d.depth),
        ("mipmapLevelCount", &mut d.mipmap_level_count),
        ("sampleCount", &mut d.sample_count),
        ("arrayLength", &mut d.array_length),
    ];
    for (key, size) in sizes {
        if let Some(value) = f.uint(key)? {
            *size = value;
        }
    }
    if let Some(usage) = f.metal("usage")? {
        d.usage = usage;
    }
    if let Some(raw) = f.uint("resourceOptions")? {
        d.resource_options = options(call, raw)?;
    }
    if let Some(mode) = f.uint("storageMode")? {
        let raw = (d.resource_options.raw() & !0xf0) | ((mode & 0xf) << 4);
        d.resource_options = options(call, raw)?;
    }
    if let Some(s) = f.fields("swizzle", &SWIZZLE_KEYS)? {
        d.swizzle = swizzle(&s)?;
    }
    if let Some(index) = f.list("resourceIndex")? {
        d.placement = placement(call, index)?;
    }
    // A texture outside the program's IDs (docs/REPLAY-STREAM.md, section 5).
    if f.bool("outside")? == Some(true) {
        if f.list("resourceIndex")?.is_some() {
            return Err(refuse(call, "a texture outside with a resource index"));
        }
        d.placement = metal2vulkan_runtime::IdPlacement::Outside;
    }
    f.only("protectionOptions", Fields::uint, 0)?;
    f.only("framebufferOnly", Fields::bool, false)?;
    f.only("isDrawable", Fields::bool, false)?;
    f.only("rotation", Fields::uint, 0)?;
    if let Some(enabled) = f.bool("writeSwizzleEnabled")? {
        d.write_swizzle_enabled = enabled;
    }
    if let Some(allow) = f.bool("allowGpuOptimizedContents")? {
        d.allow_gpu_optimized_contents = allow;
    }
    Ok(())
}

/// `newBufferWithDescriptor:`: the buffer's length and options. Its host memory reaches it nil: a
/// copied buffer's `contents` the player writes once the buffer is made, a no-copy buffer's bytes
/// arrive as `m2v.write`, and the pinned address is the recorded one the call answers, where the
/// backend makes the buffer (`address_placement`). A
/// no-copy buffer is played as a buffer of its length, so one of the read-only option
/// (`ResourceOptions::PRIVATE_BIT`) is made without it: its GPU writes, which Metal drops, are
/// kept.
pub(super) fn buffer(call: &Call) -> Result<(u64, ResourceOptions), PlayError> {
    let f = Fields::arg(
        call,
        0,
        &[
            "length",
            "options",
            "pinnedGpuAddress",
            "contents",
            "noCopy",
            "deallocator",
            "zeroFrom",
        ],
    )?;
    if f.get("contents").is_some() {
        return Err(refuse(call, "contents that are no bytes"));
    }
    let sets = f.list("options")?.unwrap_or_default();
    let mut raw = option_sets(call, 0, sets)?;
    if f.bool("noCopy")? == Some(true) {
        raw &= !ResourceOptions::PRIVATE_BIT;
    }
    Ok((f.required("length", Fields::uint)?, options(call, raw)?))
}

/// `newHeapWithDescriptor:`: Metal's new heap descriptor holds `Private` storage.
pub(super) fn heap(call: &Call) -> Result<HeapDescriptor, PlayError> {
    let known = ["size", "heapType", "options", "pinnedGpuAddress"];
    let f = Fields::arg(call, 0, &known)?;
    let kind = f.required("heapType", Fields::uint)?;
    let kind = HeapType::from_raw(kind).map_err(super::args::failed(call))?;
    let sets = f.list("options")?.unwrap_or_default();
    let options = options(call, option_sets(call, 0x20, sets)?)?;
    Ok(HeapDescriptor::new(
        kind,
        f.required("size", Fields::uint)?,
        options,
    ))
}

/// `newSamplerStateWithDescriptor:`: the descriptor, and the options of its argument-buffer
/// support and resource index.
pub(super) fn sampler(call: &Call) -> Result<(SamplerDescriptor, SamplerStateOptions), PlayError> {
    let (mut s, f) = super::super::translator::sampler(call)?;
    if let Some(raw) = f.uint("borderColor")? {
        s.border_color = sampler_border_color(raw);
    }
    let mut state = SamplerStateOptions::default();
    if let Some(support) = f.bool("supportArgumentBuffers")? {
        state.support_argument_buffers = support;
    }
    if let Some(index) = f.uint("resourceIndex")? {
        state.placement =
            metal2vulkan_runtime::IdPlacement::At(metal2vulkan_runtime::ResourceId(index));
    }
    Ok((s, state))
}

/// `newFunctionWithDescriptor:error:`, `[name, constants, options, specializedName]`: constant
/// values nil where there are none.
pub(super) fn function(call: &Call, name: &str) -> Result<FunctionDescriptor, PlayError> {
    let mut descriptor = FunctionDescriptor::new(name);
    match super::args::arg(call, 3)? {
        Value::Nil => {}
        Value::Str(specialized) => {
            descriptor = descriptor.with_specialized_name(specialized.as_str());
        }
        other => return Err(refuse(call, format!("specialized name {other}"))),
    }
    let constants = super::args::list(call, 1)?;
    if !constants.is_empty() {
        descriptor.constant_values = Some(super::super::translator::constants(call, constants)?);
    }
    descriptor.options = metal(call, super::args::uint(call, 2)?)?;
    Ok(descriptor)
}

/// Where a texture over a surface's plane lies in the buffer that stands for the surface.
pub(super) enum Surface {
    Plane(IoSurfacePlane),
    /// No surface: the buffer's own texture.
    None {
        offset: u64,
        bytes_per_row: u64,
    },
}

const FACT_KEYS: [&str; 7] = [
    "pixelFormat",
    "planeCount",
    "width",
    "height",
    "chromaOffset",
    "chromaBytesPerRow",
    "ycbcrMatrix",
];

/// `newTextureWithDescriptor:iosurface:plane:`'s plane, `{plane, offset, bytesPerRow, planar,
/// facts}`, for a texture of `format`: a two-plane YCbCr or XR10 format over a surface of two
/// planes reads its facts.
pub(super) fn surface_plane(call: &Call, format: PixelFormat) -> Result<Surface, PlayError> {
    let known = ["plane", "offset", "bytesPerRow", "planar", "facts"];
    let f = Fields::arg(call, 1, &known)?;
    let plane = f.required("plane", Fields::uint)?;
    let offset = f.required("offset", Fields::uint)?;
    let bytes_per_row = f.required("bytesPerRow", Fields::uint)?;
    let Some(facts) = f.fields("facts", &FACT_KEYS)? else {
        return Ok(Surface::None {
            offset,
            bytes_per_row,
        });
    };
    let planes = facts.required("planeCount", Fields::uint)?;
    if (planes == 0 && plane != 0) || (planes != 0 && plane >= planes) {
        return Err(refuse(call, format!("plane {plane} of {planes}")));
    }
    let at = IoSurfacePlane::new(offset, bytes_per_row, f.required("planar", Fields::bool)?);
    let ycbcr = metal2vulkan::metal::format::ycbcr(format);
    let xr10 = matches!(
        format,
        PixelFormat::RGB10A82PXR10 | PixelFormat::RGB10A82PXR10SRGB
    );
    if planes < 2 || !(ycbcr || xr10) {
        return Ok(Surface::Plane(at));
    }
    let raw = facts.required("pixelFormat", Fields::uint)?;
    let read = IoSurfaceYCbCr::new(
        narrow(call, raw, "pixel format")?,
        plane,
        [
            facts.required("width", Fields::uint)?,
            facts.required("height", Fields::uint)?,
        ],
        facts.required("chromaOffset", Fields::uint)?,
        facts.required("chromaBytesPerRow", Fields::uint)?,
    );
    let read = match facts.str("ycbcrMatrix")?.filter(|_| ycbcr) {
        None => read,
        Some(matrix) => read.with_matrix(match matrix {
            "Itu601" => SurfaceYCbCrMatrix::Itu601,
            "Itu709" => SurfaceYCbCrMatrix::Itu709,
            "Itu2020" => SurfaceYCbCrMatrix::Itu2020,
            "Smpte240M1995" => SurfaceYCbCrMatrix::Smpte240M,
            "DciP3" => SurfaceYCbCrMatrix::DciP3,
            "P3D65" => SurfaceYCbCrMatrix::P3D65,
            "Other" => SurfaceYCbCrMatrix::Other,
            _ => return Err(refuse(call, format!("YCbCr matrix {matrix}"))),
        }),
    };
    Ok(Surface::Plane(at.with_ycbcr(read)))
}

const COMPUTE_PIPELINE_KEYS: [&str; 10] = [
    "label",
    "function",
    "threadgroupSizeIsMultipleOfExecutionWidth",
    "maxTotalThreadsPerThreadgroup",
    "maxCallStackDepth",
    "supportIndirectCommandBuffers",
    "stageInput",
    "linked",
    "libraries",
    "openClMode",
];

/// What an attachment's setter record sets of the properties every attachment has, resolved: an
/// object property `Some(None)` where it is set to nil.
struct AttachmentSet {
    texture: Option<Option<Texture>>,
    resolve_texture: Option<Option<Texture>>,
    level: Option<u64>,
    slice: Option<u64>,
    depth_plane: Option<u64>,
    resolve_level: Option<u64>,
    resolve_slice: Option<u64>,
    resolve_depth_plane: Option<u64>,
    load_action: Option<LoadAction>,
    store_action: Option<StoreAction>,
    store_action_options: Option<StoreActionOptions>,
}

/// `set`'s properties set on the attachment descriptor `a`, of any of the three kinds.
macro_rules! set_attachment {
    ($set:expr, $a:expr) => {{
        let set = $set;
        let a = $a;
        if let Some(texture) = set.texture {
            a.texture = texture;
        }
        if let Some(texture) = set.resolve_texture {
            a.resolve_texture = texture;
        }
        let numbers = [
            (set.level, &mut a.level),
            (set.slice, &mut a.slice),
            (set.depth_plane, &mut a.depth_plane),
            (set.resolve_level, &mut a.resolve_level),
            (set.resolve_slice, &mut a.resolve_slice),
            (set.resolve_depth_plane, &mut a.resolve_depth_plane),
        ];
        for (value, field) in numbers {
            if let Some(value) = value {
                *field = value;
            }
        }
        if let Some(action) = set.load_action {
            a.load_action = action;
        }
        if let Some(action) = set.store_action {
            a.store_action = action;
        }
        if let Some(options) = set.store_action_options {
            a.store_action_options = options;
        }
    }};
}

impl OldRuntime {
    /// The functions a stage links, `{functions, binaryFunctions, groups, privateFunctions}`,
    /// each list's functions the stream's objects; a group is `[name, [functions]]`.
    fn linked(&self, call: &Call, f: Option<Fields>) -> Result<LinkedFunctions, PlayError> {
        let mut out = LinkedFunctions::new();
        let Some(f) = f else {
            return Ok(out);
        };
        let functions = |values: &[Value]| -> Result<Vec<_>, PlayError> {
            let linked = |value: &Value| match value.as_object() {
                Some(id) => Ok(self.function_of(call, id)?.linked_function()),
                None => Err(refuse(call, format!("{value} is no function"))),
            };
            values.iter().map(linked).collect()
        };
        out.functions = functions(f.list("functions")?.unwrap_or_default())?;
        out.binary_functions = functions(f.list("binaryFunctions")?.unwrap_or_default())?;
        out.private_functions = functions(f.list("privateFunctions")?.unwrap_or_default())?;
        for group in f.list("groups")?.unwrap_or_default() {
            let Some([Value::Str(name), Value::List(members)]) = group.as_list() else {
                return Err(refuse(call, format!("{group} is no [name, functions]")));
            };
            out.groups.insert(name.clone(), functions(members)?);
        }
        Ok(out)
    }

    /// The stage of function `key` of `f`, with the functions `linked` names.
    fn stage(
        &self,
        call: &Call,
        f: &Fields,
        key: &str,
        linked: &str,
    ) -> Result<Option<StageSource>, PlayError> {
        let Some(id) = f.object(key)? else {
            return Ok(None);
        };
        let linked = self.linked(call, f.fields(linked, &LINKED_KEYS)?)?;
        let function = self.function_of(call, id)?.function_descriptor();
        Ok(Some(StageSource::Function(function.with_linked(linked))))
    }

    /// `newRenderPipelineStateWithDescriptor:error:`: the whole descriptor's map, or the short
    /// one of a vertex and fragment function and color 0's format.
    pub(super) fn render_pipeline(
        &self,
        call: &Call,
    ) -> Result<(RenderPipelineDescriptor, PipelineOptions), PlayError> {
        let (d, f) =
            translator::render_pipeline(call, |f, key, linked| self.stage(call, f, key, linked))?;
        let mut options = PipelineOptions::default();
        options.label = f.str("label")?.map(str::to_owned);
        if let Some(index) = f.list("resourceIndex")? {
            options.placement = placement(call, index)?;
        }
        Ok((d, options))
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:`: a stage input, dynamic
    /// libraries and OpenCL mode are refused.
    pub(super) fn compute_pipeline(
        &self,
        call: &Call,
    ) -> Result<(ComputePipelineDescriptor, PipelineOptions), PlayError> {
        let f = Fields::arg(call, 0, &COMPUTE_PIPELINE_KEYS)?;
        let kernel = self.stage(call, &f, "function", "linked")?;
        let kernel = kernel.ok_or_else(|| refuse(call, "no function"))?;
        if f.get("stageInput").is_some() {
            return Err(refuse(call, "a stage input"));
        }
        if let Some(libraries) = f.fields("libraries", &["preloaded", "loaded"])? {
            let none =
                |key: &str| Ok::<_, PlayError>(libraries.list(key)?.unwrap_or_default().is_empty());
            if !none("preloaded")? || !none("loaded")? {
                return Err(refuse(call, "dynamic libraries"));
            }
        }
        f.only("openClMode", Fields::bool, false)?;
        let mut d = ComputePipelineDescriptor::new(kernel);
        if let Some(multiple) = f.bool("threadgroupSizeIsMultipleOfExecutionWidth")? {
            d.threadgroup_size_is_multiple_of_execution_width = multiple;
        }
        if let Some(most) = f.count("maxTotalThreadsPerThreadgroup")? {
            d.max_total_threads_per_threadgroup = (most != 0).then_some(most);
        }
        if let Some(depth) = f.count("maxCallStackDepth")? {
            d.max_call_stack_depth = depth;
        }
        if let Some(support) = f.bool("supportIndirectCommandBuffers")? {
            d.support_indirect_command_buffers = support;
        }
        let mut options = PipelineOptions::default();
        options.label = f.str("label")?.map(str::to_owned);
        Ok((d, options))
    }

    /// What an attachment's map sets of the properties every attachment has.
    fn attachment_set(&self, call: &Call, f: &Fields) -> Result<AttachmentSet, PlayError> {
        let texture = |key: &str| match f.set_object(key)? {
            Some(Some(id)) => self.texture_of(call, id).map(|texture| Some(Some(texture))),
            Some(None) => Ok(Some(None)),
            None => Ok(None),
        };
        Ok(AttachmentSet {
            texture: texture("texture")?,
            resolve_texture: texture("resolveTexture")?,
            level: f.uint("level")?,
            slice: f.uint("slice")?,
            depth_plane: f.uint("depthPlane")?,
            resolve_level: f.uint("resolveLevel")?,
            resolve_slice: f.uint("resolveSlice")?,
            resolve_depth_plane: f.uint("resolveDepthPlane")?,
            load_action: (f.uint("loadAction")?)
                .map(LoadAction::from_raw)
                .transpose()
                .map_err(super::args::failed(call))?,
            store_action: (f.uint("storeAction")?)
                .map(StoreAction::from_raw)
                .transpose()
                .map_err(super::args::failed(call))?,
            store_action_options: (f.uint("storeActionOptions")?)
                .map(StoreActionOptions::from_raw)
                .transpose()
                .map_err(super::args::failed(call))?,
        })
    }

    fn pass(&mut self, id: u64) -> &mut metal2vulkan_runtime::RenderPassDescriptor {
        match self.objects.get_mut(&id) {
            Some(Object::Pass(pass)) => pass,
            _ => unreachable!("a render pass descriptor's call"),
        }
    }

    /// A setter record of the render pass descriptor `call` is sent to: each property its map
    /// sets. A nil property is left as it is, and an object property of object 0 set to nil;
    /// `defaultColorSampleCount`, which changes nothing a pass shows, is not played; a rate map
    /// and accumulated visibility results are refused.
    pub(super) fn pass_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let failed = super::args::failed(call);
        match &*call.selector {
            "m2v.setPass" => {
                let f = Fields::arg(call, 0, &PASS_KEYS)?;
                // The pass never holds a rate map, so one set to nil changes nothing.
                if let Some(Some(_)) = f.set_object("rasterizationRateMap")? {
                    return Err(refuse(call, "a rasterization rate map"));
                }
                f.only("visibilityResultType", Fields::uint, 0)?;
                let visibility = match f.set_object("visibilityResultBuffer")? {
                    Some(Some(id)) => Some(Some(self.buffer_of(call, id)?)),
                    Some(None) => Some(None),
                    None => None,
                };
                let numbers = [
                    "renderTargetArrayLength",
                    "renderTargetWidth",
                    "renderTargetHeight",
                    "defaultRasterSampleCount",
                    "imageblockSampleLength",
                    "threadgroupMemoryLength",
                    "tileWidth",
                    "tileHeight",
                ]
                .map(|key| f.uint(key));
                let pass = self.pass(call.receiver);
                if let Some(visibility) = visibility {
                    pass.visibility_result_buffer = visibility;
                }
                let fields = [
                    &mut pass.render_target_array_length,
                    &mut pass.render_target_width,
                    &mut pass.render_target_height,
                    &mut pass.default_raster_sample_count,
                    &mut pass.imageblock_sample_length,
                    &mut pass.threadgroup_memory_length,
                    &mut pass.tile_width,
                    &mut pass.tile_height,
                ];
                for (value, field) in numbers.into_iter().zip(fields) {
                    if let Some(value) = value? {
                        *field = value;
                    }
                }
            }
            "m2v.setColorAttachment" => {
                let index = super::args::uint(call, 0)?;
                let index = usize::try_from(index)
                    .ok()
                    .filter(|&index| index < MAX_COLOR_ATTACHMENTS)
                    .ok_or_else(|| refuse(call, format!("color attachment {index}")))?;
                let f = Fields::arg(call, 1, &attachment_keys(&["clearColor"]))?;
                let set = self.attachment_set(call, &f)?;
                let clear = match f.fields("clearColor", &["red", "green", "blue", "alpha"])? {
                    Some(c) => Some(ClearColor::new(
                        c.required("red", Fields::float)?,
                        c.required("green", Fields::float)?,
                        c.required("blue", Fields::float)?,
                        c.required("alpha", Fields::float)?,
                    )),
                    None => None,
                };
                let a = &mut self.pass(call.receiver).color_attachments[index];
                set_attachment!(set, &mut *a);
                if let Some(clear) = clear {
                    a.clear_color = clear;
                }
            }
            "m2v.setDepthAttachment" => {
                let own = ["clearDepth", "depthResolveFilter"];
                let f = Fields::arg(call, 0, &attachment_keys(&own))?;
                let set = self.attachment_set(call, &f)?;
                let clear = f.float("clearDepth")?;
                let filter = (f.uint("depthResolveFilter")?)
                    .map(MultisampleDepthResolveFilter::from_raw)
                    .transpose()
                    .map_err(failed)?;
                let a = &mut self.pass(call.receiver).depth_attachment;
                set_attachment!(set, &mut *a);
                if let Some(clear) = clear {
                    a.clear_depth = clear;
                }
                if let Some(filter) = filter {
                    a.depth_resolve_filter = filter;
                }
            }
            "m2v.setStencilAttachment" => {
                let own = ["clearStencil", "stencilResolveFilter"];
                let f = Fields::arg(call, 0, &attachment_keys(&own))?;
                let set = self.attachment_set(call, &f)?;
                let clear = f.count("clearStencil")?;
                let filter = (f.uint("stencilResolveFilter")?)
                    .map(MultisampleStencilResolveFilter::from_raw)
                    .transpose()
                    .map_err(failed)?;
                let a = &mut self.pass(call.receiver).stencil_attachment;
                set_attachment!(set, &mut *a);
                if let Some(clear) = clear {
                    a.clear_stencil = clear;
                }
                if let Some(filter) = filter {
                    a.stencil_resolve_filter = filter;
                }
            }
            _ => return Err(refuse(call, "not played on a render pass descriptor")),
        }
        Ok(())
    }
}
