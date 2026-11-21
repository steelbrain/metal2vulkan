//! The fill of a 32-bit float depth texture a run through the runtime binds
//! ([`super::runtime_textures`]). The runtime takes no upload of one
//! (`metal2vulkan_runtime::Gap::FloatDepthUpload`), so the run renders the case's depths into it,
//! as a guest fills one: one command buffer holds a pass per level and slice, each one draw of the
//! owned `depth_fill` pair over the whole level (`fixtures/public/engine/depth_fill.metal`), whose
//! fragment writes the depth at each pixel from a buffer of the level's texels under an
//! always-passing depth test that writes. A render write is clamped to the viewport's `[0, 1]`, so
//! a depth outside it is refused, typed (`ToolingRequirement::DepthOutsideUnitRange`, which the
//! case checker raises first): the run never clamps one silently.

use super::runtime::{self, Placed};
use crate::candidate::engine::{graphics, product_refusal};
use crate::case::indirect_commands::metal_primitive_type;
use crate::case::Primitive;
use crate::check::admission::{ExecutorError, NotSupported};
use crate::executor_contract::depth_range::holds_depth_outside_unit_range;
use crate::gpu::DeviceType;
use crate::requirement::ToolingRequirement;
use metal2vulkan::library::Library;
use metal2vulkan::metal::CompareFunction;
use metal2vulkan::pipeline::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime as rt;

/// The owned fixture's opaque text, which the run assembles (`check::loop_free::assemble`).
const DEPTH_FILL: &str = include_str!("../../../fixtures/public/engine/depth_fill.ll");

/// One level and slice of a depth texture and the depths it is filled with: `width` texels a row,
/// rows tightly packed.
pub(super) struct Fill {
    pub(super) texture: rt::Texture,
    pub(super) level: u32,
    pub(super) slice: u32,
    pub(super) width: u32,
    pub(super) depths: Vec<u8>,
}

/// Render each of `fills` into its level and slice on `device`, as the module documentation
/// says; `error` maps the runtime's errors as the run's.
pub(super) fn filled(
    device: &rt::Device,
    device_type: DeviceType,
    fills: &[Fill],
    error: &impl Fn(&str, rt::Error) -> ExecutorError,
) -> Result<(), ExecutorError> {
    if fills
        .iter()
        .any(|fill| holds_depth_outside_unit_range(&fill.depths))
    {
        let requirement = ToolingRequirement::DepthOutsideUnitRange;
        return Err(NotSupported::Requirements([requirement].into()).into());
    }
    let state = pipeline(device)?;
    let written = device.new_depth_stencil_state(&rt::DepthStencilDescriptor {
        depth_compare_function: CompareFunction::Always,
        depth_write_enabled: true,
        ..rt::DepthStencilDescriptor::default()
    });
    let buffers = fills
        .iter()
        .map(|fill| {
            let texels = Placed::new(device, "a depth fill's texels", &fill.depths)?;
            let width = Placed::new(device, "a depth fill's width", &fill.width.to_le_bytes())?;
            Ok((texels, width))
        })
        .collect::<Result<Vec<_>, ExecutorError>>()?;
    let fail = |what: &'static str| move |e: rt::Error| error(what, e);
    runtime::submitted(
        device,
        device_type,
        "a depth fill through the runtime",
        |command_buffer| {
            for (fill, (texels, width)) in fills.iter().zip(&buffers) {
                let mut pass = rt::RenderPassDescriptor::default();
                let attachment = &mut pass.depth_attachment;
                attachment.texture = Some(fill.texture.clone());
                attachment.level = fill.level;
                attachment.slice = fill.slice;
                attachment.load_action = rt::LoadAction::DontCare;
                attachment.store_action = rt::StoreAction::Store;
                let mut encoder = command_buffer
                    .render_command_encoder(&pass)
                    .map_err(fail("a depth fill's pass"))?;
                encoder
                    .set_render_pipeline_state(&state)
                    .map_err(fail("the depth fill's pipeline state"))?;
                encoder
                    .set_fragment_buffer(Some(&texels.buffer), 0, 0)
                    .map_err(fail("a depth fill's texels"))?;
                encoder
                    .set_fragment_buffer(Some(&width.buffer), 0, 1)
                    .map_err(fail("a depth fill's width"))?;
                encoder
                    .set_depth_stencil_state(Some(&written))
                    .map_err(fail("the depth fill's depth-stencil state"))?;
                encoder
                    .draw_primitives(metal_primitive_type(Primitive::Triangle), 0, 3, 1, 0)
                    .map_err(fail("a depth fill's draw"))?;
                encoder.end_encoding();
            }
            Ok(())
        },
    )
}

/// The depth fill's pipeline state: the fixture's pair rendering into a `Depth32Float`
/// attachment alone, made within the run's ceilings (`runtime::made_within`).
fn pipeline(device: &rt::Device) -> Result<rt::RenderPipelineState, ExecutorError> {
    let bitcode = crate::check::loop_free::assemble(DEPTH_FILL)
        .map_err(|e| ExecutorError::Failed(format!("the depth fill's module: {e}")))?;
    let library = Library::from_air(&bitcode).map_err(|e| product_refusal(&e))?;
    let stage = |entry: &str| {
        library
            .function(entry)
            .map(|function| {
                StageSource::Function(FunctionDescriptor::new(function).with_specialized(true))
            })
            .map_err(|e| product_refusal(&e))
    };
    let mut descriptor = RenderPipelineDescriptor::new(stage("depth_fill_vertex")?);
    descriptor.fragment = Some(stage("depth_fill_fragment")?);
    descriptor.depth_format = Some(rt::PixelFormat::Depth32Float);
    descriptor.input_primitive_topology = graphics::topology_class(Primitive::Triangle);
    runtime::made_within(|numbered| {
        device.new_render_pipeline_state(&descriptor, |samplers| numbered(samplers))
    })
}
