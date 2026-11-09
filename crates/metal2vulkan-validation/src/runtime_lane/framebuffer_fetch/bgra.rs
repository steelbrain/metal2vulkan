//! A BGRA-order color attachment fetched on the device the cfg selects, under the validation
//! layer: Metal's answers, measured on the M3 Max (macOS 27.0.1, our own MSL probe of these passes:
//! every byte of `BGRA8Unorm` and of `RGBA8Unorm`, single-sampled and at 4 samples, a mask of every
//! channel and one of red and alpha, one to three draws).
//!
//! A `SIDE`² private `BGRA8Unorm` (or, beside it, `RGBA8Unorm`) render target, single-sampled or
//! of `MSAA` samples, holds in its texel or sample `k` (`k = i · samples + s` for sample `s` of
//! texel `i = y · SIDE + x`) the memory bytes `k`, `37 k + k / 256`, `101 k + 7 (k / 256)`,
//! `k / 16` (each mod 256): each byte takes every value. Beside it, color 1 is an `RGBA16Float`
//! target the function never writes (the shape of the guest pipeline refused on ANV), cleared to
//! (0.25, 0.5, 0.75, 1) and stored. A pass loads and stores the target and draws `fetch_ordered`
//! (`fragment_fetch`: `float4` read in a raster order group, half of it returned) over the
//! fullscreen triangle `N` times, at the target's samples, no blending, under a mask of every
//! channel or of red and alpha. Metal stores in each written channel's byte `c` the byte
//! [`halved`] gives, once per draw and on every sample: the code read as the correctly rounded
//! `c / 255`, halved, and written as the exact product with 255 rounded to nearest (never a tie
//! for `0 < c < 255`); a channel the mask leaves out keeps its byte, and color 1 its clear.
//!
//! In the pixel interlock on a device that accesses `B8G8R8A8_UNORM` without a format in neither
//! direction (ANV: no `shaderStorageImageReadWithoutFormat`, and SPIR-V names no image format for
//! it), the compile layer reads and writes the `BGRA8Unorm` attachment through a view in
//! `R8G8B8A8_UNORM`, exchanging red and blue (`StorageAttachmentUse::view_alias`), which the test
//! asserts of the pipeline the translator makes for the descriptor there; elsewhere through its own
//! format. Whether the target's own image took the view (its free `STORAGE`, then fetched in place)
//! or the pass rendered it into a scratch made mutable to it (`fetch-through-scratch`), and at 4x
//! into sample planes made so, is printed. Through input attachments (lavapipe) the device reads
//! the attachment itself. Every byte is Metal's in each form, but where the pipeline records
//! `device-color-conversion` (the device's conversion of the written color, every device but
//! MoltenVK's): there a channel within one code of Metal's holds, counted and printed
//! (`device_rounded`).
//!
//! A single-sampled target is filled by a blit and read back by one. A multisampled one is filled
//! sample by sample by `rt6_samples`' fill (each code drawn as `c / 255`, which the write stores as
//! `c`) and read back sample by sample through its `RGBA8Uint` view
//! (`candidate::runtime_samples`); the fill is checked to store the bytes drawn.

use super::{device_rounded, fetches, form, library, reads_planes, scratched, through_inputs};
use crate::candidate::runtime_samples::{encode_through, shape};
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::runtime_lane::degrade_notices::made_render_kinds;
use crate::runtime_lane::resolves::{lane, quiet, Lane, FULL};
use crate::runtime_lane::{buffer_with, WAIT};
use metal2vulkan::metal::{ColorWriteMask, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, StorageAttachmentUse,
};
use metal2vulkan::vk::Format;
use metal2vulkan::Budget;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, Origin, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};

/// The target's side, as the probe's.
const SIDE: u64 = 64;

/// The multisampled targets' samples, as the probe's.
const MSAA: u32 = 4;

/// The fullscreen triangle's depth.
const Z: f32 = 0.5;

/// Color 1's clear, and the `half` bits it stores.
const CLEAR: [f64; 4] = [0.25, 0.5, 0.75, 1.0];
const CLEARED: [u16; 4] = [0x3400, 0x3800, 0x3a00, 0x3c00];

/// What Metal stores in an 8-bit unorm byte holding `c` when a draw returns half of what it
/// fetched: the code read as the correctly rounded `c / 255`, halved, written as the exact product
/// with 255 rounded to nearest (the module docs; the M3 probe matched it in every byte).
fn halved(c: u8) -> u8 {
    let read = f32::from(c) / 255.0;
    (f64::from(read) * 0.5 * 255.0).round_ties_even() as u8
}

/// A target: its format, samples and the color write mask of the fetching pipeline.
#[derive(Clone, Copy, Debug)]
struct Target {
    pf: PixelFormat,
    samples: u32,
    mask: ColorWriteMask,
}

impl Target {
    /// The target's Vulkan format.
    fn format(self) -> Format {
        metal2vulkan::metal::format::direct_format(self.pf).unwrap()
    }

    /// The bytes of each texel or sample, before the pass (the module docs), in memory order.
    fn initial(self) -> Vec<u8> {
        (0..(SIDE * SIDE) as usize * self.samples as usize)
            .flat_map(|k| [k, 37 * k + k / 256, 101 * k + 7 * (k / 256), k / 16].map(|b| b as u8))
            .collect()
    }

    /// Which memory bytes of a texel the mask writes: a `BGRA8Unorm` texel holds blue, green, red
    /// and alpha in that order.
    fn written(self) -> [bool; 4] {
        let [r, g, b, a] = [
            ColorWriteMask::RED,
            ColorWriteMask::GREEN,
            ColorWriteMask::BLUE,
            ColorWriteMask::ALPHA,
        ]
        .map(|c| self.mask.contains(c));
        match self.pf {
            PixelFormat::BGRA8Unorm => [b, g, r, a],
            _ => [r, g, b, a],
        }
    }

    /// Metal's bytes after `draws` draws (the module docs).
    fn expected(self, draws: usize) -> Vec<u8> {
        let written = self.written();
        let mut bytes = self.initial();
        for (i, byte) in bytes.iter_mut().enumerate() {
            if written[i % 4] {
                for _ in 0..draws {
                    *byte = halved(*byte);
                }
            }
        }
        bytes
    }

    /// A private `SIDE`² texture of `pf` and `usage`, of the target's samples.
    fn texture(self, lane: &Lane, pf: PixelFormat, usage: TextureUsage) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        if self.samples > 1 {
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = self.samples.into();
        }
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::private();
        lane.device.new_texture(&descriptor).unwrap()
    }

    /// The color target, holding [`Target::initial`].
    fn filled(self, lane: &Lane) -> Texture {
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        if self.samples > 1 {
            let texture = self.texture(lane, self.pf, usage | TextureUsage::PIXEL_FORMAT_VIEW);
            self.fill_samples(lane, &texture);
            return texture;
        }
        let texture = self.texture(lane, self.pf, usage);
        let bytes = self.initial();
        let source = buffer_with(&lane.device, &bytes).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            4 * SIDE,
            bytes.len() as u64,
            Size::new(SIDE, SIDE, 1),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        texture
    }

    /// Draw [`Target::initial`] into the multisampled `texture`, sample by sample: each code as
    /// `c / 255`, each float's channel the byte it lands on (a `BGRA` format's red its third
    /// byte).
    fn fill_samples(self, lane: &Lane, texture: &Texture) {
        let bgra = self.pf == PixelFormat::BGRA8Unorm;
        let floats: Vec<u8> = self
            .initial()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|b| {
                let (red, blue) = if bgra { (b[2], b[0]) } else { (b[0], b[2]) };
                [red, b[1], blue, b[3]].map(|c| f32::from(c) / 255.0)
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        let samples = library(lane, "rt6_samples");
        let function = |name| {
            samples
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("rt6_fill_vertex")));
        descriptor.fragment = Some(StageSource::Function(function("rt6_fill_rgba32float")));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(self.pf));
        descriptor.raster_sample_count = self.samples;
        let pipeline = lane
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let values = buffer_with(&lane.device, &floats).unwrap();
        let shape = shape(&lane.device, SIDE as u32, self.samples).unwrap();
        let mut fill = RenderPassDescriptor::new();
        let target = &mut fill.color_attachments[0];
        target.texture = Some(texture.clone());
        target.load_action = LoadAction::DontCare;
        target.store_action = StoreAction::Store;
        lane.run(&fill, |_, encoder, _| {
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder.set_fragment_buffer(Some(&values), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        })
        .unwrap();
    }

    /// Every texel's bytes (every sample's, in order, of a multisampled target) of `texture`.
    fn read(self, lane: &Lane, texture: &Texture) -> Vec<u8> {
        if self.samples == 1 {
            return lane.read(texture, (0, 0), 4, BlitOption::None);
        }
        let view = texture.new_texture_view(PixelFormat::RGBA8Uint).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let readback = encode_through(
            &lane.device,
            &command_buffer,
            &view,
            "rt6_read_bytes4",
            SampleRead::Color(ReadbackView::R32Uint),
            [SIDE as u32, SIDE as u32],
            self.samples,
        )
        .unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        readback.bytes().unwrap()
    }

    /// The descriptor of `fetch_ordered` over the fullscreen triangle, drawing the target at its
    /// samples under its mask, color 1 an `RGBA16Float` attachment it never writes.
    fn descriptor(
        self,
        lane: &Lane,
        fetch: &metal2vulkan_runtime::Library,
    ) -> RenderPipelineDescriptor {
        let function = |library: &metal2vulkan_runtime::Library, name| {
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function(&lane.library, FULL)));
        descriptor.fragment = Some(StageSource::Function(function(fetch, "fetch_ordered")));
        let mut color = ColorAttachmentDescriptor::new(self.pf);
        color.write_mask = self.mask;
        descriptor.color_attachments[0] = Some(color);
        descriptor.color_attachments[1] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        descriptor.raster_sample_count = self.samples;
        descriptor
    }

    /// Run `draws` draws of `fetch` over a fresh target: its bytes after, and color 1's (a
    /// single-sampled one's; none at more samples, where it is stored `DontCare`).
    fn case(
        self,
        lane: &Lane,
        fetch: &RenderPipelineState,
        draws: usize,
    ) -> (Vec<u8>, Option<Vec<u8>>) {
        let color = self.filled(lane);
        let other = self.texture(lane, PixelFormat::RGBA16Float, TextureUsage::RENDER_TARGET);
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Load;
        target.store_action = StoreAction::Store;
        let target = &mut pass.color_attachments[1];
        target.texture = Some(other.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(CLEAR[0], CLEAR[1], CLEAR[2], CLEAR[3]);
        target.store_action = match self.samples {
            1 => StoreAction::Store,
            _ => StoreAction::DontCare,
        };
        lane.run(&pass, |lane, encoder, held| {
            let z = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
            encoder.set_render_pipeline_state(fetch).unwrap();
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            for _ in 0..draws {
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
            held.push(z);
        })
        .unwrap();
        let other = (self.samples == 1).then(|| lane.read(&other, (0, 0), 8, BlitOption::None));
        (self.read(lane, &color), other)
    }
}

/// The texels or samples of `read` that differ from `want`, by index: each read's and Metal's
/// bytes.
fn differing(read: &[u8], want: &[u8]) -> Vec<(usize, [u8; 4], [u8; 4])> {
    let texel = |bytes: &[u8]| -> [u8; 4] { bytes.try_into().unwrap() };
    read.as_chunks::<4>()
        .0
        .iter()
        .zip(want.as_chunks::<4>().0.iter())
        .enumerate()
        .filter(|(_, (r, w))| r != w)
        .map(|(k, (r, w))| (k, texel(r), texel(w)))
        .collect()
}

/// Every case's bytes are Metal's, on every device, for `BGRA8Unorm` and `RGBA8Unorm`,
/// single-sampled and at `MSAA` samples, under both masks, but within one code where the
/// pipeline records `device-color-conversion` (the module docs); color 1 keeps its clear. On a device
/// that fetches through the pixel interlock without reading `B8G8R8A8_UNORM` without a format
/// (ANV), the `BGRA8Unorm` pipeline's storage view is `R8G8B8A8_UNORM` (the module docs).
#[test]
fn a_bgra_target_is_fetched_as_metal_fetches_it() {
    let (messages, _cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let fetch = library(&lane, "fragment_fetch");
    let device = lane.device.name();
    let inputs = through_inputs(&lane);
    let profile = lane.device.compile_device().profile();
    let formatless = profile.images().storage_read_without_format;
    let red_alpha = ColorWriteMask::RED.union(ColorWriteMask::ALPHA);
    let mut aliased = 0;
    for pf in [PixelFormat::BGRA8Unorm, PixelFormat::RGBA8Unorm] {
        for samples in [1, MSAA] {
            for mask in [ColorWriteMask::ALL, red_alpha] {
                let target = Target { pf, samples, mask };
                let what = format!("{pf:?} at {samples}x, mask {mask:?}, on {device}");
                if samples > 1 && mask == ColorWriteMask::ALL {
                    let filled = target.filled(&lane);
                    assert_eq!(
                        target.read(&lane, &filled),
                        target.initial(),
                        "{what}: the fill stored other bytes than drawn"
                    );
                }
                let descriptor = target.descriptor(&lane, &fetch);
                let converts = made_render_kinds(&lane.device, &descriptor)
                    .contains(&Degradation::DeviceColorConversion);
                // The view the pipeline binds the fetched attachment through, in the interlock.
                let view = match inputs {
                    true => None,
                    false => {
                        let made = lane
                            .device
                            .compile_device()
                            .compile_render_pipeline(&descriptor, &Budget::standard())
                            .unwrap_or_else(|e| panic!("{what}: {e}"));
                        let storage = &made.graphics.storage_attachments;
                        assert_eq!(storage.len(), 1, "{what}: {storage:?}");
                        let want = StorageAttachmentUse::view_alias_of(target.format(), profile);
                        assert_eq!(storage[0].view_alias, want, "{what}");
                        // Without reads of no format, a BGRA-order attachment takes its twin.
                        if pf == PixelFormat::BGRA8Unorm && !formatless {
                            assert_eq!(want, Some(Format::R8G8B8A8_UNORM), "{what}");
                        }
                        aliased += usize::from(want.is_some());
                        Some(storage[0].view_format())
                    }
                };
                let state = lane
                    .device
                    .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                for draws in 1..=3 {
                    let what = format!("{what}, x{draws}");
                    let before = lane.device.work_counts().storage_forms_taken;
                    let (read, other) = target.case(&lane, &state, draws);
                    // A fresh target fetched through a scratch takes its storage form at the
                    // pass's commit (`texture/storage_form.rs`); one whose own image took the view
                    // is fetched in place from the first pass.
                    let switched = lane.device.work_counts().storage_forms_taken - before;
                    let want = target.expected(draws);
                    let (wrong, rounded) = device_rounded(differing(&read, &want), converts);
                    let total = want.len() / 4;
                    let shown = &wrong[..wrong.len().min(8)];
                    assert!(
                        wrong.is_empty(),
                        "{what}: {} of {total} texels or samples differ from Metal's, first \
                         (index, read, Metal's) {shown:?}",
                        wrong.len()
                    );
                    if let Some(other) = other {
                        let cleared: Vec<u8> = CLEARED
                            .iter()
                            .flat_map(|h| h.to_le_bytes())
                            .collect::<Vec<u8>>()
                            .repeat((SIDE * SIDE) as usize);
                        assert_eq!(other, cleared, "{what}: color 1 lost its clear");
                    }
                    let planes = samples > 1 && reads_planes(&lane);
                    let place = match (inputs, samples, switched) {
                        (true, ..) => "",
                        (false, 1, 0) => " (in place)",
                        (false, 1, _) => " (through a scratch, then its storage form)",
                        (false, ..) if planes => " (sample planes)",
                        (false, ..) => "",
                    };
                    let bytes = match rounded {
                        0 => "Metal's bytes".to_owned(),
                        n => format!("{n} channels one code from Metal's, the device's conversion"),
                    };
                    eprintln!(
                        "BGRA framebuffer fetch {what} through {}{}{place}: {bytes}",
                        form(&lane),
                        view.map_or(String::new(), |v| format!(", viewed in {v:?}")),
                    );
                }
            }
        }
    }
    eprintln!(
        "BGRA framebuffer fetch on {device}: {aliased} pipelines viewed in an RGBA twin, a fetch \
         through a scratch announced: {}",
        scratched(&messages)
    );
    quiet(lane, &messages);
}
