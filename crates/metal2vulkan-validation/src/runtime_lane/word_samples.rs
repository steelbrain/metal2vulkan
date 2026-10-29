//! A partly masked draw into a 4-sample `RG8Unorm_sRGB` attachment on the device the cfg selects,
//! under the validation layer. On a device that renders the format only through the view of its
//! word (`metal::format::WordAttachment`) and orders fetch through rasterization-order input
//! attachments (lavapipe), the fragment function merges its word into each sample's own texel,
//! read through the attachment's input attachment at the invocation's sample
//! (`lower::fragment::word_input`), the pipeline running once per covered sample.
//!
//! The owned fixture `wm_samples` draws an 8 x 8 target: a pass clears it to (0.2, 0.6) and draws
//! (0.25, 0.75) over its lower left half under the full mask, so the samples of each texel on the
//! edge hold either color; a second pass loads it and draws (0.6, 0.1) over all of it under the red
//! bit alone. Every sample must then hold the second draw's red and keep its own green, and both
//! colors must be the ones a single-sample texture of the format stores under the same draws (a
//! path `runtime_lane::color_writes::partly_masked_packed_formats_keep_their_fields` holds to
//! Metal's codes). Each sample is read by a kernel through `texture2d_ms`, decoded by the device's
//! transfer, so equal codes read equal values.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::format::WordAttachment;
use metal2vulkan::metal::{ColorWriteMask, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, CommandQueue, Device, Library, LoadAction, PipelineOptions,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};

/// The target's side and samples.
const SIDE: u64 = 8;
const SAMPLES: u64 = 4;
const TEXELS: usize = (SIDE * SIDE) as usize;

const PF: PixelFormat = PixelFormat::RG8UnormSRGB;
const CLEAR: [f64; 4] = [0.2, 0.6, 0.0, 1.0];
const FIRST: [f32; 4] = [0.25, 0.75, 0.0, 1.0];
const SECOND: [f32; 4] = [0.6, 0.1, 0.0, 1.0];

struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn function(&self, name: &str) -> StageSource {
        StageSource::Function(
            self.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    }

    /// A private `RG8Unorm_sRGB` texture `samples` deep that renders and reads.
    fn texture(&self, samples: u64) -> Texture {
        let mut d = TextureDescriptor::texture_2d(PF, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = samples;
        }
        self.device
            .new_texture(&d)
            .unwrap_or_else(|e| panic!("a {samples}-sample texture: {e}"))
    }

    /// A pass into `target`, cleared to [`CLEAR`] where `clear` and else loaded, drawing `color`
    /// with `vertex` under `mask`.
    fn pass(
        &self,
        target: &Texture,
        clear: bool,
        vertex: &str,
        mask: ColorWriteMask,
        color: [f32; 4],
    ) {
        let mut c = ColorAttachmentDescriptor::new(PF);
        c.write_mask = mask;
        let mut d = RenderPipelineDescriptor::new(self.function(vertex));
        d.fragment = Some(self.function("wm_fill"));
        d.color_attachments[0] = Some(c);
        d.raster_sample_count = target.sample_count() as u32;
        let pipeline = self
            .device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap_or_else(|e| panic!("{vertex} under {mask:?}: {e}"));
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(target.clone());
        a.store_action = StoreAction::Store;
        match clear {
            true => {
                let [r, g, b, alpha] = CLEAR;
                a.load_action = LoadAction::Clear;
                a.clear_color = ClearColor::new(r, g, b, alpha);
            }
            false => a.load_action = LoadAction::Load,
        }
        let bytes: Vec<u8> = color.iter().flat_map(|v| v.to_le_bytes()).collect();
        let values = buffer_with(&self.device, &bytes).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// Every texel's red and green bits as `kernel` reads them, `per` a texel (its samples).
    fn read(&self, kernel: &str, t: &Texture, per: usize) -> Vec<[u32; 2]> {
        let state = self
            .device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(self.function(kernel)),
                &PipelineOptions::default(),
            )
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
        let length = 16 * TEXELS * per;
        let out = buffer_with(&self.device, &vec![0xee; length]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        encoder.set_texture(Some(t), 0).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, SIDE, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let bytes = read_buffer(&out, 0, length as u64).unwrap();
        bytes
            .chunks(16)
            .map(|t| {
                let lane = |i: usize| u32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
                [lane(0), lane(1)]
            })
            .collect()
    }
}

/// Each sample of a 4-sample `RG8Unorm_sRGB` target takes a red-masked draw's red and keeps its
/// own green: the samples of the texels on the first draw's edge keep greens that differ, each the
/// one a single-sample texture stores under the same draw.
#[test]
fn a_partly_masked_draw_merges_each_samples_own_word() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
    let bitcode = llvm_assemble(&engine_dir().join("wm_samples.ll")).unwrap();
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        library: device.new_library_with_air(&bitcode).unwrap(),
        device,
    };
    let name = lane.device.name().to_owned();
    let word = WordAttachment::of(PF, lane.device.compile_device().profile());
    let red = ColorWriteMask::RED;
    // The single-sample answers: the clear's green under a red-masked draw, and a full draw's
    // green.
    let one = lane.texture(1);
    lane.pass(&one, true, "wm_vertex", red, SECOND);
    let kept = lane.read("wm_read", &one, 1);
    lane.pass(&one, true, "wm_vertex", ColorWriteMask::ALL, FIRST);
    let drawn = lane.read("wm_read", &one, 1);
    let ([red_bits, clear_green], [_, first_green]) = (kept[0], drawn[0]);
    assert!(kept.iter().all(|t| *t == kept[0]) && drawn.iter().all(|t| *t == drawn[0]));
    assert_ne!(clear_green, first_green);
    // The 4-sample target: half drawn, then red-masked over all of it.
    let ms = lane.texture(SAMPLES);
    lane.pass(&ms, true, "wm_half_vertex", ColorWriteMask::ALL, FIRST);
    let before = lane.read("wm_read_samples", &ms, SAMPLES as usize);
    lane.pass(&ms, false, "wm_vertex", red, SECOND);
    let after = lane.read("wm_read_samples", &ms, SAMPLES as usize);
    let mut mixed = 0;
    for texel in 0..TEXELS {
        let samples = texel * SAMPLES as usize..(texel + 1) * SAMPLES as usize;
        let greens: Vec<u32> = before[samples.clone()].iter().map(|s| s[1]).collect();
        for (s, &green) in greens.iter().enumerate() {
            assert!(
                green == clear_green || green == first_green,
                "{name}: texel {texel} sample {s} held green {green:#x} before the masked draw"
            );
        }
        if greens.contains(&clear_green) && greens.contains(&first_green) {
            mixed += 1;
        }
        for (s, (b, a)) in before[samples.clone()]
            .iter()
            .zip(&after[samples])
            .enumerate()
        {
            assert_eq!(
                *a,
                [red_bits, b[1]],
                "{name}: texel {texel} sample {s}: before {b:x?}"
            );
        }
    }
    println!(
        "WORD-SAMPLES {name}: {} texels of {SAMPLES} samples merged, {mixed} on the edge with \
         both greens; {}",
        TEXELS,
        match word {
            Some(_) => "through the view of its word",
            None => "rendered in its own format",
        }
    );
    assert!(mixed > 0, "{name}: no texel's samples differ");
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}
