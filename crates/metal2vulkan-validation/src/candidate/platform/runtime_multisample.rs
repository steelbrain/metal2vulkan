//! The multisampled textures and attachments of a run through the runtime, filled and read as
//! `executor_contract::multisample` says. Metal uploads and copies no multisampled texture, and
//! neither does the runtime, so each slice is filled by one draw of the owned `sample_fill`
//! fixture (`fixtures/public/engine/sample_fill.metal`) whose fragment runs once per sample
//! (`sample_id`) and writes that sample's texel from a buffer: a color texel through an
//! unsigned-integer view of its texel size (the texture made with `PixelFormatView` usage,
//! [`color_view`], which takes a texture index of the run's grant, as shaders reach it), so its
//! bits cross unconverted, and a `Depth32Float` one as `depth(any)` under an always-passing
//! depth test that writes. A depth outside `[0, 1]` is refused, typed, as `runtime_depth`
//! refuses it. A stencil aspect holds one value in every sample (`uniform_stencil`), which the
//! case's pass clears it to. After the work, a dispatch of the fixture's kernels reads every
//! sample, one threadgroup each: a color texel through the unsigned-integer view, a depth bit-cast,
//! a stencil value as its first component, as whole words in the attachment layout
//! (`unpack_words`). Nothing is resolved.

use super::runtime::{self, Placed};
use crate::candidate::engine::{graphics, product_refusal};
use crate::case::indirect_commands::metal_primitive_type;
use crate::case::Primitive;
use crate::check::admission::{ExecutorError, NotSupported};
use crate::executor_contract::depth_range::holds_depth_outside_unit_range;
use crate::executor_contract::multisample::{
    readback_bytes, unpack_words, ReadbackView, SampleRead,
};
use crate::gpu::DeviceType;
use crate::requirement::ToolingRequirement;
use metal2vulkan::library::{Function, Library};
use metal2vulkan::metal::CompareFunction;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor,
    RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime as rt;

/// The owned fixture's opaque text, which the run assembles (`check::loop_free::assemble`).
const SAMPLE_FILL: &str = include_str!("../../../fixtures/public/engine/sample_fill.ll");

/// One slice of a multisampled texture and the texels it is filled with, read as `read` (a color
/// view or depth): every sample of each pixel, pixels row-major, as the case's bytes lay them out.
/// A color texture renders through `view`, its [`color_view`]; a depth one has none.
pub(super) struct Fill {
    pub(super) texture: rt::Texture,
    pub(super) view: Option<rt::Texture>,
    pub(super) slice: u32,
    pub(super) read: SampleRead,
    pub(super) texels: Vec<u8>,
}

/// Render each of `fills` into its slice on `device`, as the module documentation says.
pub(super) fn filled(
    device: &rt::Device,
    device_type: DeviceType,
    fills: &[Fill],
    error: &impl Fn(&str, rt::Error) -> ExecutorError,
) -> Result<(), ExecutorError> {
    if fills.is_empty() {
        return Ok(());
    }
    let library = library()?;
    let mut always = rt::DepthStencilDescriptor::default();
    always.depth_compare_function = CompareFunction::Always;
    always.depth_write_enabled = true;
    let written = device.new_depth_stencil_state(&always);
    let made = fills
        .iter()
        .map(|fill| Filling::make(device, &library, fill))
        .collect::<Result<Vec<_>, ExecutorError>>()?;
    let fail = |what: &'static str| move |e: rt::Error| error(what, e);
    runtime::submitted(
        device,
        device_type,
        "a multisampled texture's fill through the runtime",
        |command_buffer| {
            for (fill, made) in fills.iter().zip(&made) {
                let mut pass = rt::RenderPassDescriptor::default();
                match &made.view {
                    Some(view) => {
                        let attachment = &mut pass.color_attachments[0];
                        attachment.texture = Some(view.clone());
                        attachment.slice = fill.slice;
                        attachment.load_action = rt::LoadAction::DontCare;
                        attachment.store_action = rt::StoreAction::Store;
                    }
                    None => {
                        let attachment = &mut pass.depth_attachment;
                        attachment.texture = Some(fill.texture.clone());
                        attachment.slice = fill.slice;
                        attachment.load_action = rt::LoadAction::DontCare;
                        attachment.store_action = rt::StoreAction::Store;
                    }
                }
                let mut encoder = command_buffer
                    .render_command_encoder(&pass)
                    .map_err(fail("a multisampled fill's pass"))?;
                encoder
                    .set_render_pipeline_state(&made.state)
                    .map_err(fail("the multisampled fill's pipeline state"))?;
                encoder
                    .set_fragment_buffer(Some(&made.texels.buffer), 0, 0)
                    .map_err(fail("a multisampled fill's texels"))?;
                encoder
                    .set_fragment_buffer(Some(&made.shape.buffer), 0, 1)
                    .map_err(fail("a multisampled fill's shape"))?;
                if made.view.is_none() {
                    encoder
                        .set_depth_stencil_state(Some(&written))
                        .map_err(fail("the multisampled fill's depth-stencil state"))?;
                }
                encoder
                    .draw_primitives(metal_primitive_type(Primitive::Triangle), 0, 3, 1, 0)
                    .map_err(fail("a multisampled fill's draw"))?;
                encoder.end_encoding();
            }
            Ok(())
        },
    )
}

/// What one fill draws with: its pipeline state, its texels as the fragment reads them, the
/// slice's shape, and the color view it renders through (`None` for depth).
struct Filling {
    state: rt::RenderPipelineState,
    texels: Placed,
    shape: Placed,
    view: Option<rt::Texture>,
}

impl Filling {
    fn make(device: &rt::Device, library: &Library, fill: &Fill) -> Result<Filling, ExecutorError> {
        let made = fill.texture.descriptor();
        let samples = made.sample_count;
        let shape = Placed::new(
            device,
            "a multisampled fill's shape",
            &shape(made.width, samples, 1),
        )?;
        let vertex = stage(library, "sample_fill_vertex")?;
        let mut descriptor = RenderPipelineDescriptor::new(vertex);
        descriptor.raster_sample_count = samples;
        descriptor.input_primitive_topology = graphics::topology_class(Primitive::Triangle);
        let (texels, view) = match fill.read {
            SampleRead::Color(reading) => {
                let format = view_format(reading);
                descriptor.fragment = Some(stage(library, "sample_fill_color")?);
                descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
                let view = fill.view.clone().ok_or_else(|| {
                    ExecutorError::Failed("a multisampled color fill has no view".into())
                })?;
                (widened(&fill.texels, reading.texel_bytes())?, Some(view))
            }
            SampleRead::Depth => {
                if holds_depth_outside_unit_range(&fill.texels) {
                    let requirement = ToolingRequirement::DepthOutsideUnitRange;
                    return Err(NotSupported::Requirements([requirement].into()).into());
                }
                descriptor.fragment = Some(stage(library, "sample_fill_depth")?);
                descriptor.depth_format = Some(rt::PixelFormat::Depth32Float);
                (fill.texels.clone(), None)
            }
            SampleRead::Stencil => {
                return Err(ExecutorError::Failed(
                    "a multisampled stencil aspect is cleared, never filled sample by sample"
                        .into(),
                ))
            }
        };
        let state = runtime::made_within(|numbered| {
            device.new_render_pipeline_state(&descriptor, |samplers| numbered(samplers))
        })?;
        Ok(Filling {
            state,
            texels: Placed::new(device, "a multisampled fill's texels", &texels)?,
            shape,
            view,
        })
    }
}

/// The unsigned-integer view of the multisampled color `texture` that its fill renders through and
/// its read reads, of `read`'s texel size, at texture index `at` of the run's grant: shaders reach
/// it, so it takes an index (`Refusal::TextureIndex`).
pub(super) fn color_view(
    texture: &rt::Texture,
    read: ReadbackView,
    at: u32,
) -> Result<rt::Texture, rt::Error> {
    let made = texture.descriptor();
    texture.new_texture_view(
        view_format(read),
        made.texture_type,
        0..1,
        0..made.array_length.max(1),
        Some(at),
    )
}

/// Every sample of the 2D multisampled `texture`, read as `read`, in the attachment layout. A
/// color texture is read through `view`, its [`color_view`]; a depth or stencil one is bound
/// itself, at its own index (`view` then `None`).
pub(super) fn read_back(
    device: &rt::Device,
    device_type: DeviceType,
    texture: &rt::Texture,
    read: SampleRead,
    view: Option<&rt::Texture>,
    error: &impl Fn(&str, rt::Error) -> ExecutorError,
) -> Result<Vec<u8>, ExecutorError> {
    let made = texture.descriptor();
    let (dimensions, samples) = ([made.width, made.height], made.sample_count);
    let library = library()?;
    let entry = match read {
        SampleRead::Color(_) => "sample_read_color",
        SampleRead::Depth => "sample_read_depth",
        SampleRead::Stencil => "sample_read_stencil",
    };
    let function = library.function(entry).map_err(|e| product_refusal(&e))?;
    let descriptor = ComputePipelineDescriptor::new(described(function));
    let state = runtime::made_within(|numbered| {
        device.new_compute_pipeline_state(&descriptor, |samplers| numbered(samplers))
    })?;
    let bound = match (read, view) {
        (SampleRead::Color(_), Some(view)) => view,
        (SampleRead::Color(_), None) => {
            return Err(ExecutorError::Failed(
                "a multisampled color read has no view".into(),
            ))
        }
        (SampleRead::Depth | SampleRead::Stencil, _) => texture,
    };
    let length = readback_bytes(dimensions, samples, read).map_err(ExecutorError::Failed)?;
    let words = Placed::new(device, "the multisampled read's words", &vec![0; length])?;
    let words_each = read.words() as u32;
    let shape = Placed::new(
        device,
        "the multisampled read's shape",
        &shape(made.width, samples, words_each),
    )?;
    let fail = |what: &'static str| move |e: rt::Error| error(what, e);
    runtime::submitted(
        device,
        device_type,
        "a multisampled texture's read through the runtime",
        |command_buffer| {
            let mut encoder = command_buffer
                .compute_command_encoder(rt::DispatchType::Serial)
                .map_err(fail("the multisampled read's encoder"))?;
            encoder
                .set_compute_pipeline_state(&state)
                .map_err(fail("the multisampled read's pipeline state"))?;
            encoder
                .set_texture(Some(bound), 0)
                .map_err(fail("the multisampled read's texture"))?;
            encoder
                .set_buffer(Some(&words.buffer), 0, 0)
                .map_err(fail("the multisampled read's words"))?;
            encoder
                .set_buffer(Some(&shape.buffer), 0, 1)
                .map_err(fail("the multisampled read's shape"))?;
            // One thread per sample, one per threadgroup: no grid edge to guard.
            encoder
                .dispatch_threadgroups([made.width, made.height, samples], [1, 1, 1])
                .map_err(fail("the multisampled read's dispatch"))?;
            encoder.end_encoding();
            Ok(())
        },
    )?;
    unpack_words(&words.bytes(), dimensions, samples, read).map_err(ExecutorError::Failed)
}

/// The fixture's library, from its assembled text.
fn library() -> Result<Library, ExecutorError> {
    let bitcode = crate::check::loop_free::assemble(SAMPLE_FILL)
        .map_err(|e| ExecutorError::Failed(format!("the multisampled fill's module: {e}")))?;
    Library::from_air(&bitcode).map_err(|e| product_refusal(&e))
}

/// The fixture's function `entry` as a stage.
fn stage(library: &Library, entry: &str) -> Result<StageSource, ExecutorError> {
    let function = library.function(entry).map_err(|e| product_refusal(&e))?;
    Ok(described(function))
}

/// `function` as a specialized stage source.
fn described(function: Function) -> StageSource {
    StageSource::Function(FunctionDescriptor::new(function).with_specialized(true))
}

/// The fixture's `SampleShape`: the slice's width in pixels, its samples, and the words a read
/// writes of each sample.
fn shape(width: u32, samples: u32, words: u32) -> Vec<u8> {
    [width, samples, words]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// The `uint4` each sample's texel of `texel` bytes is to the fill's fragment: its little-endian
/// words, the last zero-extended, then zeros.
fn widened(texels: &[u8], texel: usize) -> Result<Vec<u8>, ExecutorError> {
    if texel == 0 || !texels.len().is_multiple_of(texel) {
        return Err(ExecutorError::Failed(format!(
            "a multisampled fill of {} bytes is no whole number of {texel}-byte texels",
            texels.len()
        )));
    }
    Ok(texels
        .chunks_exact(texel)
        .flat_map(|chunk| {
            let mut wide = [0u8; 16];
            wide[..texel].copy_from_slice(chunk);
            wide
        })
        .collect())
}

/// The pixel format of `view`.
fn view_format(view: ReadbackView) -> rt::PixelFormat {
    match view {
        ReadbackView::R8Uint => rt::PixelFormat::R8Uint,
        ReadbackView::R16Uint => rt::PixelFormat::R16Uint,
        ReadbackView::R32Uint => rt::PixelFormat::R32Uint,
        ReadbackView::Rg32Uint => rt::PixelFormat::RG32Uint,
        ReadbackView::Rgba32Uint => rt::PixelFormat::RGBA32Uint,
    }
}
