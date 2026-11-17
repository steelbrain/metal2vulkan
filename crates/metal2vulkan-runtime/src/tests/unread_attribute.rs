//! A render state over a vertex function whose attribute reaches only a varying a function constant
//! gates off is made from a vertex descriptor without that attribute, and draws Metal's pixels.
//!
//! The functions are the owned fixture
//! `validation/fixtures/public/engine/fca_vertex_unread_attribute.ll`'s
//! `fca_vertex_unread_attribute` (the position from attribute 0, the constant `shade` varying, and
//! attribute 1 into the `color` varying constant 0 gates) and `fca_fragment_unread_attribute`
//! (`color` where the constant holds, else `shade`), over a 4x4 `RGBA8Unorm` target that one
//! triangle covers. Metal's answer, on an Apple M3 Max (macOS 27.0.1, validation layer off and on):
//! with the constant false a descriptor of attribute 0 alone makes the state and the target reads
//! `shade`, `(64, 128, 191, 255)`; with it true that descriptor is refused, and one of both
//! attributes draws `color`, attribute 1's `(1, 0, 0, 1)` scaled by 0.5 and biased by 0.25:
//! `(191, 64, 64, 191)`. Each channel is compared within one code of Metal's: near a tie the
//! device may store the other of the two codes nearest the channel's scaled value (Vulkan 1.4,
//! 3.10.2), as the fragment module records (`PrecisionNote::DeviceColorConversion`; NVIDIA's
//! 615.71.09 stores `shade`'s 0.5 as 127). The two answers lie far more than a code apart.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::{library, pass, words_of};
use crate::{
    BlitOption, CommandBufferStatus, Device, DeviceKind, DeviceOptions, Error, Library, Origin,
    PipelineOptions, RenderPipelineState, ResourceOptions, ShaderCacheOptions, Size,
    TextureDescriptor,
};
use metal2vulkan::error::{DescriptorError, Error as CompileError};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{DataType, PixelFormat, PrimitiveType, TextureUsage, VertexFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};

/// The side of the target.
const SIDE: u64 = 4;

/// Three vertices of `{ position, color }` that cover the target, `color` `(1, 0, 0, 1)`.
const VERTICES: [[f32; 8]; 3] = [
    [-1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    [3.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    [-1.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
];

/// A device of the kind the cfg selects, with an empty shader cache of its own.
fn device(cache: &std::path::Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A state of the fixture's functions with constant 0 `has_color`, drawing into `RGBA8Unorm`, its
/// vertex descriptor attributes `0..attributes` (`Float4` each, 16 bytes apart in buffer 0's
/// 32-byte records).
fn state(
    device: &Device,
    library: &Library,
    has_color: bool,
    attributes: u32,
) -> Result<RenderPipelineState, Error> {
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::Bool, [u8::from(has_color)]),
    );
    let function = |name| {
        let function = library
            .new_function_with_name_constant_values(name, &constants)
            .unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("fca_vertex_unread_attribute"));
    descriptor.fragment = Some(function("fca_fragment_unread_attribute"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let mut vertex = VertexDescriptor::new();
    for a in 0..attributes {
        vertex.attributes.insert(
            a,
            VertexAttributeDescriptor::new(VertexFormat::Float4, 16 * a, 0),
        );
    }
    vertex
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(32)));
    descriptor.vertex_descriptor = Some(vertex);
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// The bytes of texel (1, 1) of the target one triangle of `state` covers, committed and waited
/// for.
fn draw(device: &Device, state: &RenderPipelineState) -> [u8; 4] {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let bytes: Vec<u8> = VERTICES
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_bytes(&bytes, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    words_of(&back)[(SIDE + 1) as usize].to_le_bytes()
}

/// Whether each channel of `got` lies within one code of Metal's `metal` (the module docs).
fn metal_s_codes(got: [u8; 4], metal: [u8; 4]) -> bool {
    got.iter()
        .zip(metal)
        .all(|(&code, m)| code.abs_diff(m) <= 1)
}

/// Gated off, the state is made without attribute 1 and draws `shade`; gated on, it is refused
/// without attribute 1, and with it draws `color` (the module docs).
#[test]
fn a_state_without_an_attribute_only_a_gated_off_varying_reads_draws_metal_s_pixels() {
    let scratch = Scratch::new("unread-attribute");
    let device = device(&scratch.0);
    let library = library(&device, "engine/fca_vertex_unread_attribute.ll");
    let off = state(&device, &library, false, 1)
        .unwrap_or_else(|e| panic!("gated off, attribute 0 alone: {e}"));
    let shade = draw(&device, &off);
    eprintln!(
        "UNREAD-ATTRIBUTE: {} gated off draws {shade:?}",
        device.name()
    );
    assert!(
        metal_s_codes(shade, [64, 128, 191, 255]),
        "gated off: Metal's shade (64, 128, 191, 255), drew {shade:?}"
    );
    match state(&device, &library, true, 1) {
        Err(Error::Compile(CompileError::InvalidDescriptor(
            DescriptorError::VertexAttribute { attribute: 1 },
        ))) => {}
        other => panic!("gated on, attribute 0 alone: {:?}", other.map(|_| ())),
    }
    let on = state(&device, &library, true, 2)
        .unwrap_or_else(|e| panic!("gated on, both attributes: {e}"));
    let color = draw(&device, &on);
    eprintln!(
        "UNREAD-ATTRIBUTE: {} gated on draws {color:?}",
        device.name()
    );
    assert!(
        metal_s_codes(color, [191, 64, 64, 191]),
        "gated on: Metal's color (191, 64, 64, 191), drew {color:?}"
    );
}
