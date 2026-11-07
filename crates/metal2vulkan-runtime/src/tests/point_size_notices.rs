//! `Degradation::DevicePointSize` is announced at the draws that rasterize points and at no other:
//! `[[point_size]]` sizes points alone (Metal has no point fill mode, and Vulkan reads `PointSize`
//! only where it rasterizes points), so a triangle or line draw of a state whose vertex function
//! writes a point size is Metal's exactly, whatever the device's largest point
//! (`Device::degrade_point_size`, `ShapeInputs::settings`).
//!
//! The device announces the kind at every occurrence (`DeviceOptions::announce_each`), so neither
//! the state's creation nor a bind of its variant can announce it unheard: a triangle and a line
//! draw before the first point draw and after it announce nothing, and each point draw announces
//! it once where the device's `pointSizeRange` ends below Metal's 511 pixels (lavapipe's 256,
//! ANV's 255) and nothing where it reaches them.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are those of the owned
//! fixture `validation/fixtures/public/engine/rt6_render_state.ll` (`llvm-as`): `rt6_point`, a
//! point of 4 pixels that writes `[[point_size]]` and reads its depth from vertex buffer 0, and
//! `rt6_white`.

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    Degradation, DegradeNotice, Device, DeviceKind, DeviceOptions, LoadAction, MessageSink,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ShaderCacheOptions, StoreAction, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Metal's largest point, in pixels.
const METAL_LARGEST_POINT: u32 = 511;

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects that announces `DevicePointSize` at every occurrence,
/// its notices into `heard`.
fn device(cache: &Path, heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
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
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        announce_each: [Degradation::DevicePointSize].into_iter().collect(),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// How many `DevicePointSize` notices `heard` holds, emptied.
fn point_size_notices(heard: &Heard) -> usize {
    let mut heard = heard.lock().unwrap();
    let count = (heard.iter())
        .filter(|notice| notice.degradation == Degradation::DevicePointSize)
        .count();
    heard.clear();
    count
}

/// The fixture's state of `rt6_point` and `rt6_white` into one `RGBA8Unorm` color, of the
/// unspecified topology class, which draws any primitive type.
fn state(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/rt6_render_state.ll");
    let function = |name| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("rt6_point"));
    descriptor.fragment = Some(function("rt6_white"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Draw `primitive` once with `state` into a cleared 8 by 8 target, its vertex buffer 0 the depth
/// 0.5, and wait.
fn draw(device: &Device, state: &RenderPipelineState, primitive: PrimitiveType) {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(device.new_texture(&d).unwrap());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    let depth: Vec<u8> = [0.5f32, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    encoder.set_vertex_bytes(&depth, 0).unwrap();
    let vertices = match primitive {
        PrimitiveType::Point => 1,
        PrimitiveType::Line => 2,
        _ => 3,
    };
    encoder
        .draw_primitives(primitive, 0, vertices, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A triangle or line draw of a state whose vertex function writes `[[point_size]]` announces no
/// `DevicePointSize`, before a point draw of it and after one, nor does the state's creation; each
/// point draw announces it where the device's largest point is below Metal's, and none does where
/// it is not.
#[test]
fn only_a_draw_of_points_announces_the_device_s_point_size() {
    let scratch = Scratch::new("point-size-notices");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let limited = device.compile_device().profile().limits().max_point_size < METAL_LARGEST_POINT;
    let state = state(&device);
    assert_eq!(point_size_notices(&heard), 0, "the state's creation");
    let point = usize::from(limited);
    let draws = [
        (PrimitiveType::Triangle, 0),
        (PrimitiveType::Line, 0),
        (PrimitiveType::Point, point),
        (PrimitiveType::TriangleStrip, 0),
        (PrimitiveType::LineStrip, 0),
        (PrimitiveType::Point, point),
        (PrimitiveType::Triangle, 0),
    ];
    for (n, (primitive, expected)) in draws.into_iter().enumerate() {
        draw(&device, &state, primitive);
        assert_eq!(
            point_size_notices(&heard),
            expected,
            "draw {n}, of {primitive:?}, on a device whose largest point is {}below Metal's",
            if limited { "" } else { "not " }
        );
    }
}
