//! The labels a program gives a command buffer and its encoders (`command_buffer/labels.rs`):
//! read back as set, and each ended encoder keeps its own in its place, none where none was set,
//! under both render lowering policies (a render encoder that logs its calls ends on the device's
//! lowering thread). A device test: compiled only under the validation crate's GPU safety cfgs,
//! on the kind of device the cfg selects (docs/VALIDATION.md).

use crate::tests::device_recovery::{Scratch, WAIT};
use crate::{
    Device, DeviceKind, DeviceOptions, EncoderKind, RenderPassDescriptor, ResourceOptions,
    ShaderCacheOptions,
};
use metal2vulkan::device::RenderLoweringPolicy;
use std::path::Path;

/// A device of the kind the cfg selects that lowers render calls as `lowering` says, its shader
/// cache in `cache`.
fn device(cache: &Path, lowering: RenderLoweringPolicy) -> Device {
    let mut options = DeviceOptions {
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
    options.policy.render_lowering = lowering;
    Device::system_default_with(&options).unwrap()
}

#[test]
fn labels_read_back_and_each_ended_encoder_keeps_its_own() {
    for lowering in [RenderLoweringPolicy::AtCall, RenderLoweringPolicy::Worker] {
        let scratch = Scratch::new();
        let device = device(&scratch.0, lowering);
        let queue = device.new_command_queue().unwrap();
        let filled = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        assert_eq!(command_buffer.label(), None);
        command_buffer.set_label("frame");
        assert_eq!(command_buffer.label().as_deref(), Some("frame"));
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        assert_eq!(blit.label(), None);
        blit.set_label("fill");
        assert_eq!(blit.label().as_deref(), Some("fill"));
        blit.fill_buffer(&filled, 0..4096, 1).unwrap();
        blit.end_encoding().unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let mut render = command_buffer.render_command_encoder(&pass).unwrap();
        assert_eq!(render.label(), None, "{lowering:?}");
        render.set_label("pass");
        assert_eq!(render.label().as_deref(), Some("pass"), "{lowering:?}");
        render.end_encoding().unwrap();
        let compute = command_buffer.compute_command_encoder().unwrap();
        assert_eq!(compute.label(), None, "{lowering:?}");
        compute.end_encoding().unwrap();
        let ended: Vec<_> = (command_buffer.ended_encoders().into_iter())
            .map(|encoder| (encoder.kind, encoder.label))
            .collect();
        let labeled = |label: &str| Some(label.to_owned());
        assert_eq!(
            ended,
            [
                (EncoderKind::Blit, labeled("fill")),
                (EncoderKind::Render, labeled("pass")),
                (EncoderKind::Compute, None),
            ],
            "{lowering:?}"
        );
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.label().as_deref(), Some("frame"));
    }
}
