//! `A8Unorm` textures on a device that holds no `A8_UNORM` image, on the device the cfg selects,
//! under the validation layer. The device is made with `VK_KHR_maintenance5` and its feature
//! withheld, so at the runtime's Vulkan 1.2 instance it exposes no `A8_UNORM` (core only from
//! Vulkan 1.4), and an `A8Unorm` texture is an `R8_UNORM` image holding Metal's alpha in red
//! (`FormatEmulation::AlphaInRed`): a storage slot reads the byte as alpha and writes a texel's
//! alpha (`StorageImageClass::AlphaInRed`), and a color attachment takes the fragment's alpha in
//! red, the write mask's alpha bit, the clear's alpha and Metal's alpha blend there.
//!
//! Every byte is checked against Metal's, measured with our own probe `a8` on an Apple M3 Max
//! under macOS 27.0.1, 8 x 8 textures: a kernel's `float` and `half` writes of (0.2, 0.4, 0.6,
//! 0.8) store 204; a read, a sample and a `read_write` read of the byte 153 are (0, 0, 0, 0.6), and
//! that slot's write of (0.1, 0.3, 0.5, 0.7) stores 178; a render of (0.2, 0.4, 0.6, 0.8) and a
//! clear to it store 204, and of the write masks only the alpha bit writes; each of the 167 blend
//! states of [`BLENDS`] over the byte 102 stores its byte (an output of alpha 0.65, a second
//! output of alpha 0.45, a blend color of alpha 0.35, and color factors that would read other
//! values); a 4-sample render resolves as `R8Unorm`'s does (in the device's own `A8_UNORM` too,
//! where its resolve is Metal's rule). A blend is the device's blend unit's,
//! which rounds its own way (`PrecisionNote::DeviceColorConversion`): each is checked within a code
//! of Metal's, and exactly against the same blend into the device's own `A8_UNORM` texture where
//! the device has the format (a device made with the extension). A fetch of such an attachment
//! reads `(0, 0, 0, a)`, at one sample and at four (through sample planes where the device reads
//! them), and the write after it stores the alpha as the write mask lets it; alpha-to-coverage
//! through it as attachment 0 covers as through
//! `RGBA8Unorm` (probe `a8`'s uses, the same M3 Max: a fetch of the byte 153 reads
//! `(0, 0, 0, 0.6)`, and at every alpha in 1/32 steps the masks and resolves of the two formats
//! are the same; a resolve here is held within a code where the device's average makes it). A
//! texture buffer's bytes are viewed as `R8_UNORM` there, and an argument buffer's storage texture
//! is converted by its handle's class: each reads the byte as alpha and writes a texel's alpha as
//! Metal's does (the same probe's texture buffers, swizzled ones among them, and argument buffers).

use super::degrade_notices::{made_render_kinds, resolve_kind};
use super::driver_deviations::stage_notes;
use super::framebuffer_fetch::fetches;
use super::pixel_samplers::metal_sampler;
use super::{
    buffer_with, converted_holds, half_within, layer_reported_nothing_at_all, read_buffer,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::format::alpha_in_red;
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, PrimitiveType, ResourceUsage,
    TextureSwizzle, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, PrecisionNote, RenderPipelineDescriptor,
    StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, Error, Library,
    LoadAction, PipelineOptions, Region, RenderPassDescriptor, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, TextureSwizzleChannels,
};

/// The textures' side.
const SIDE: u64 = 8;
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The color the writes, renders and clears take.
const COLOR: [f32; 4] = [0.2, 0.4, 0.6, 0.8];
/// The byte Metal stores of [`COLOR`]'s alpha.
const STORED: u8 = 204;

/// One blend state of the probe: Metal's alpha operation and source and destination alpha factors,
/// and the byte Metal stored over 102.
type Blend = (BlendOperation, BlendFactor, BlendFactor, u8);

/// The probe's blends (the module docs): each alpha factor against `One`, and itself on both
/// sides, under `Add`, `Subtract` and `ReverseSubtract`, and `Min` and `Max`.
const BLENDS: [Blend; 167] = {
    use BlendFactor as F;
    use BlendOperation as O;
    [
        (O::Add, F::Zero, F::Zero, 0),
        (O::Add, F::Zero, F::One, 102),
        (O::Add, F::One, F::Zero, 166),
        (O::Add, F::One, F::One, 255),
        (O::Add, F::One, F::SourceColor, 232),
        (O::Add, F::One, F::OneMinusSourceColor, 201),
        (O::Add, F::One, F::SourceAlpha, 232),
        (O::Add, F::One, F::OneMinusSourceAlpha, 201),
        (O::Add, F::One, F::DestinationColor, 207),
        (O::Add, F::One, F::OneMinusDestinationColor, 227),
        (O::Add, F::One, F::DestinationAlpha, 207),
        (O::Add, F::One, F::OneMinusDestinationAlpha, 227),
        (O::Add, F::One, F::SourceAlphaSaturated, 255),
        (O::Add, F::One, F::BlendColor, 201),
        (O::Add, F::One, F::OneMinusBlendColor, 232),
        (O::Add, F::One, F::BlendAlpha, 201),
        (O::Add, F::One, F::OneMinusBlendAlpha, 232),
        (O::Add, F::One, F::Source1Color, 212),
        (O::Add, F::One, F::OneMinusSource1Color, 222),
        (O::Add, F::One, F::Source1Alpha, 212),
        (O::Add, F::One, F::OneMinusSource1Alpha, 222),
        (O::Add, F::SourceColor, F::One, 210),
        (O::Add, F::SourceColor, F::SourceColor, 174),
        (O::Add, F::OneMinusSourceColor, F::One, 160),
        (O::Add, F::OneMinusSourceColor, F::OneMinusSourceColor, 94),
        (O::Add, F::SourceAlpha, F::One, 210),
        (O::Add, F::SourceAlpha, F::SourceAlpha, 174),
        (O::Add, F::OneMinusSourceAlpha, F::One, 160),
        (O::Add, F::OneMinusSourceAlpha, F::OneMinusSourceAlpha, 94),
        (O::Add, F::DestinationColor, F::One, 168),
        (O::Add, F::DestinationColor, F::DestinationColor, 107),
        (O::Add, F::OneMinusDestinationColor, F::One, 201),
        (
            O::Add,
            F::OneMinusDestinationColor,
            F::OneMinusDestinationColor,
            161,
        ),
        (O::Add, F::DestinationAlpha, F::One, 168),
        (O::Add, F::DestinationAlpha, F::DestinationAlpha, 107),
        (O::Add, F::OneMinusDestinationAlpha, F::One, 201),
        (
            O::Add,
            F::OneMinusDestinationAlpha,
            F::OneMinusDestinationAlpha,
            161,
        ),
        (O::Add, F::SourceAlphaSaturated, F::One, 255),
        (
            O::Add,
            F::SourceAlphaSaturated,
            F::SourceAlphaSaturated,
            255,
        ),
        (O::Add, F::BlendColor, F::One, 160),
        (O::Add, F::BlendColor, F::BlendColor, 94),
        (O::Add, F::OneMinusBlendColor, F::One, 210),
        (O::Add, F::OneMinusBlendColor, F::OneMinusBlendColor, 174),
        (O::Add, F::BlendAlpha, F::One, 160),
        (O::Add, F::BlendAlpha, F::BlendAlpha, 94),
        (O::Add, F::OneMinusBlendAlpha, F::One, 210),
        (O::Add, F::OneMinusBlendAlpha, F::OneMinusBlendAlpha, 174),
        (O::Add, F::Source1Color, F::One, 177),
        (O::Add, F::Source1Color, F::Source1Color, 120),
        (O::Add, F::OneMinusSource1Color, F::One, 193),
        (
            O::Add,
            F::OneMinusSource1Color,
            F::OneMinusSource1Color,
            147,
        ),
        (O::Add, F::Source1Alpha, F::One, 177),
        (O::Add, F::Source1Alpha, F::Source1Alpha, 120),
        (O::Add, F::OneMinusSource1Alpha, F::One, 193),
        (
            O::Add,
            F::OneMinusSource1Alpha,
            F::OneMinusSource1Alpha,
            147,
        ),
        (O::Subtract, F::Zero, F::Zero, 0),
        (O::Subtract, F::Zero, F::One, 0),
        (O::Subtract, F::One, F::Zero, 166),
        (O::Subtract, F::One, F::One, 64),
        (O::Subtract, F::One, F::SourceColor, 99),
        (O::Subtract, F::One, F::OneMinusSourceColor, 130),
        (O::Subtract, F::One, F::SourceAlpha, 99),
        (O::Subtract, F::One, F::OneMinusSourceAlpha, 130),
        (O::Subtract, F::One, F::DestinationColor, 125),
        (O::Subtract, F::One, F::OneMinusDestinationColor, 105),
        (O::Subtract, F::One, F::DestinationAlpha, 125),
        (O::Subtract, F::One, F::OneMinusDestinationAlpha, 105),
        (O::Subtract, F::One, F::SourceAlphaSaturated, 64),
        (O::Subtract, F::One, F::BlendColor, 130),
        (O::Subtract, F::One, F::OneMinusBlendColor, 99),
        (O::Subtract, F::One, F::BlendAlpha, 130),
        (O::Subtract, F::One, F::OneMinusBlendAlpha, 99),
        (O::Subtract, F::One, F::Source1Color, 120),
        (O::Subtract, F::One, F::OneMinusSource1Color, 110),
        (O::Subtract, F::One, F::Source1Alpha, 120),
        (O::Subtract, F::One, F::OneMinusSource1Alpha, 110),
        (O::Subtract, F::SourceColor, F::One, 6),
        (O::Subtract, F::SourceColor, F::SourceColor, 41),
        (O::Subtract, F::OneMinusSourceColor, F::One, 0),
        (
            O::Subtract,
            F::OneMinusSourceColor,
            F::OneMinusSourceColor,
            22,
        ),
        (O::Subtract, F::SourceAlpha, F::One, 6),
        (O::Subtract, F::SourceAlpha, F::SourceAlpha, 41),
        (O::Subtract, F::OneMinusSourceAlpha, F::One, 0),
        (
            O::Subtract,
            F::OneMinusSourceAlpha,
            F::OneMinusSourceAlpha,
            22,
        ),
        (O::Subtract, F::DestinationColor, F::One, 0),
        (O::Subtract, F::DestinationColor, F::DestinationColor, 25),
        (O::Subtract, F::OneMinusDestinationColor, F::One, 0),
        (
            O::Subtract,
            F::OneMinusDestinationColor,
            F::OneMinusDestinationColor,
            38,
        ),
        (O::Subtract, F::DestinationAlpha, F::One, 0),
        (O::Subtract, F::DestinationAlpha, F::DestinationAlpha, 25),
        (O::Subtract, F::OneMinusDestinationAlpha, F::One, 0),
        (
            O::Subtract,
            F::OneMinusDestinationAlpha,
            F::OneMinusDestinationAlpha,
            38,
        ),
        (O::Subtract, F::SourceAlphaSaturated, F::One, 64),
        (
            O::Subtract,
            F::SourceAlphaSaturated,
            F::SourceAlphaSaturated,
            64,
        ),
        (O::Subtract, F::BlendColor, F::One, 0),
        (O::Subtract, F::BlendColor, F::BlendColor, 22),
        (O::Subtract, F::OneMinusBlendColor, F::One, 6),
        (
            O::Subtract,
            F::OneMinusBlendColor,
            F::OneMinusBlendColor,
            41,
        ),
        (O::Subtract, F::BlendAlpha, F::One, 0),
        (O::Subtract, F::BlendAlpha, F::BlendAlpha, 22),
        (O::Subtract, F::OneMinusBlendAlpha, F::One, 6),
        (
            O::Subtract,
            F::OneMinusBlendAlpha,
            F::OneMinusBlendAlpha,
            41,
        ),
        (O::Subtract, F::Source1Color, F::One, 0),
        (O::Subtract, F::Source1Color, F::Source1Color, 29),
        (O::Subtract, F::OneMinusSource1Color, F::One, 0),
        (
            O::Subtract,
            F::OneMinusSource1Color,
            F::OneMinusSource1Color,
            35,
        ),
        (O::Subtract, F::Source1Alpha, F::One, 0),
        (O::Subtract, F::Source1Alpha, F::Source1Alpha, 29),
        (O::Subtract, F::OneMinusSource1Alpha, F::One, 0),
        (
            O::Subtract,
            F::OneMinusSource1Alpha,
            F::OneMinusSource1Alpha,
            35,
        ),
        (O::ReverseSubtract, F::Zero, F::Zero, 0),
        (O::ReverseSubtract, F::Zero, F::One, 102),
        (O::ReverseSubtract, F::One, F::Zero, 0),
        (O::ReverseSubtract, F::One, F::One, 0),
        (O::ReverseSubtract, F::One, F::SourceColor, 0),
        (O::ReverseSubtract, F::One, F::OneMinusSourceColor, 0),
        (O::ReverseSubtract, F::One, F::SourceAlpha, 0),
        (O::ReverseSubtract, F::One, F::OneMinusSourceAlpha, 0),
        (O::ReverseSubtract, F::One, F::DestinationColor, 0),
        (O::ReverseSubtract, F::One, F::OneMinusDestinationColor, 0),
        (O::ReverseSubtract, F::One, F::DestinationAlpha, 0),
        (O::ReverseSubtract, F::One, F::OneMinusDestinationAlpha, 0),
        (O::ReverseSubtract, F::One, F::SourceAlphaSaturated, 0),
        (O::ReverseSubtract, F::One, F::BlendColor, 0),
        (O::ReverseSubtract, F::One, F::OneMinusBlendColor, 0),
        (O::ReverseSubtract, F::One, F::BlendAlpha, 0),
        (O::ReverseSubtract, F::One, F::OneMinusBlendAlpha, 0),
        (O::ReverseSubtract, F::One, F::Source1Color, 0),
        (O::ReverseSubtract, F::One, F::OneMinusSource1Color, 0),
        (O::ReverseSubtract, F::One, F::Source1Alpha, 0),
        (O::ReverseSubtract, F::One, F::OneMinusSource1Alpha, 0),
        (O::ReverseSubtract, F::SourceColor, F::One, 0),
        (O::ReverseSubtract, F::SourceColor, F::SourceColor, 0),
        (O::ReverseSubtract, F::OneMinusSourceColor, F::One, 44),
        (
            O::ReverseSubtract,
            F::OneMinusSourceColor,
            F::OneMinusSourceColor,
            0,
        ),
        (O::ReverseSubtract, F::SourceAlpha, F::One, 0),
        (O::ReverseSubtract, F::SourceAlpha, F::SourceAlpha, 0),
        (O::ReverseSubtract, F::OneMinusSourceAlpha, F::One, 44),
        (
            O::ReverseSubtract,
            F::OneMinusSourceAlpha,
            F::OneMinusSourceAlpha,
            0,
        ),
        (O::ReverseSubtract, F::DestinationColor, F::One, 36),
        (
            O::ReverseSubtract,
            F::DestinationColor,
            F::DestinationColor,
            0,
        ),
        (O::ReverseSubtract, F::OneMinusDestinationColor, F::One, 3),
        (
            O::ReverseSubtract,
            F::OneMinusDestinationColor,
            F::OneMinusDestinationColor,
            0,
        ),
        (O::ReverseSubtract, F::DestinationAlpha, F::One, 36),
        (
            O::ReverseSubtract,
            F::DestinationAlpha,
            F::DestinationAlpha,
            0,
        ),
        (O::ReverseSubtract, F::OneMinusDestinationAlpha, F::One, 3),
        (
            O::ReverseSubtract,
            F::OneMinusDestinationAlpha,
            F::OneMinusDestinationAlpha,
            0,
        ),
        (O::ReverseSubtract, F::SourceAlphaSaturated, F::One, 0),
        (
            O::ReverseSubtract,
            F::SourceAlphaSaturated,
            F::SourceAlphaSaturated,
            0,
        ),
        (O::ReverseSubtract, F::BlendColor, F::One, 44),
        (O::ReverseSubtract, F::BlendColor, F::BlendColor, 0),
        (O::ReverseSubtract, F::OneMinusBlendColor, F::One, 0),
        (
            O::ReverseSubtract,
            F::OneMinusBlendColor,
            F::OneMinusBlendColor,
            0,
        ),
        (O::ReverseSubtract, F::BlendAlpha, F::One, 44),
        (O::ReverseSubtract, F::BlendAlpha, F::BlendAlpha, 0),
        (O::ReverseSubtract, F::OneMinusBlendAlpha, F::One, 0),
        (
            O::ReverseSubtract,
            F::OneMinusBlendAlpha,
            F::OneMinusBlendAlpha,
            0,
        ),
        (O::ReverseSubtract, F::Source1Color, F::One, 27),
        (O::ReverseSubtract, F::Source1Color, F::Source1Color, 0),
        (O::ReverseSubtract, F::OneMinusSource1Color, F::One, 11),
        (
            O::ReverseSubtract,
            F::OneMinusSource1Color,
            F::OneMinusSource1Color,
            0,
        ),
        (O::ReverseSubtract, F::Source1Alpha, F::One, 27),
        (O::ReverseSubtract, F::Source1Alpha, F::Source1Alpha, 0),
        (O::ReverseSubtract, F::OneMinusSource1Alpha, F::One, 11),
        (
            O::ReverseSubtract,
            F::OneMinusSource1Alpha,
            F::OneMinusSource1Alpha,
            0,
        ),
        (O::Min, F::Zero, F::Zero, 102),
        (O::Max, F::Zero, F::Zero, 166),
    ]
};

/// The blend's source, its second output, and the blend color (the module docs).
const SOURCE: [f32; 8] = [0.2, 0.4, 0.6, 0.65, 0.9, 0.15, 0.25, 0.45];
const BLEND_COLOR: [f32; 4] = [0.1, 0.3, 0.5, 0.35];
/// The byte a blend's target holds before it.
const BELOW: u8 = 102;

struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
    /// The owned fixture `a8r_buffers`: texture-buffer and argument-buffer kernels.
    buffers: Library,
}

impl Lane {
    /// The lane's device, `VK_KHR_maintenance5` and its feature withheld where `withheld`, with the owned fixtures
    /// `a8r_targets` and `a8r_buffers`.
    fn new(messages: &Messages, cache: &ScratchCache, withheld: bool) -> Lane {
        Lane::withholding(messages, cache, withheld, &[])
    }

    /// [`Lane::new`], the device made without `features` too.
    fn withholding(
        messages: &Messages,
        cache: &ScratchCache,
        withheld: bool,
        features: &[VkFeature],
    ) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
            if withheld {
                options
                    .withheld_extensions
                    .insert(VkExtension::KhrMaintenance5);
                options.withheld_features.insert(VkFeature::Maintenance5);
            }
            options.withheld_features.extend(features.iter().copied());
        })
        .unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("a8r_targets.ll")).unwrap();
        let buffers = llvm_assemble(&engine_dir().join("a8r_buffers.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            buffers: device.new_library_with_air(&buffers).unwrap(),
            device,
        }
    }

    /// Whether the device holds `A8Unorm` in red.
    fn in_red(&self) -> bool {
        alpha_in_red(PixelFormat::A8Unorm, self.device.compile_device().profile())
    }

    fn function(&self, name: &str) -> StageSource {
        StageSource::Function(
            self.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    }

    /// A shared texture of `pf`, `samples` deep, of `usage`, holding `fill` where single-sampled.
    fn texture(&self, pf: PixelFormat, samples: u64, usage: TextureUsage, fill: u8) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        d.usage = usage;
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = samples;
            d.resource_options = ResourceOptions::private();
        } else {
            d.resource_options = ResourceOptions::shared();
        }
        let t = self
            .device
            .new_texture(&d)
            .unwrap_or_else(|e| panic!("a {pf:?} texture: {e}"));
        if samples == 1 {
            let bytes = vec![fill; TEXELS];
            t.replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &bytes, SIDE, 0)
                .unwrap();
        }
        t
    }

    fn bytes(&self, t: &Texture) -> Vec<u8> {
        self.wide_bytes(t, 1)
    }

    /// The bytes of `t`, of `texel` bytes a texel.
    fn wide_bytes(&self, t: &Texture, texel: u64) -> Vec<u8> {
        let mut out = vec![0u8; TEXELS * texel as usize];
        t.get_bytes(
            &mut out,
            SIDE * texel,
            0,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
        out
    }

    /// A texture of `pf`, `samples` deep, of `usage`, its texels as made (a pass clears it).
    fn bare(&self, pf: PixelFormat, samples: u64, usage: TextureUsage) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        d.usage = usage;
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = samples;
            d.resource_options = ResourceOptions::private();
        } else {
            d.resource_options = ResourceOptions::shared();
        }
        self.device
            .new_texture(&d)
            .unwrap_or_else(|e| panic!("a {pf:?} texture: {e}"))
    }

    /// `kernel` over every texel of `t`, with `color` at buffer 0, and buffer 1's `float4`s.
    fn dispatch(&self, kernel: &str, t: &Texture, color: [f32; 4]) -> Vec<[f32; 4]> {
        let state = self
            .device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(self.function(kernel)),
                &PipelineOptions::default(),
            )
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
        let color: Vec<u8> = color.iter().flat_map(|v| v.to_le_bytes()).collect();
        let color = buffer_with(&self.device, &color).unwrap();
        let out = buffer_with(&self.device, &vec![0xee; 16 * TEXELS]).unwrap();
        let sampler = self
            .device
            .new_sampler_state(&SamplerDescriptor::new())
            .unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&color), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        encoder.set_texture(Some(t), 0).unwrap();
        encoder.set_sampler_state(Some(&sampler), 0).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, SIDE, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let bytes = read_buffer(&out, 0, 16 * TEXELS as u64).unwrap();
        bytes
            .chunks(16)
            .map(|t| {
                std::array::from_fn(|i| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap()))
            })
            .collect()
    }

    /// A pipeline of `vertex` and `fragment` into `color`, at `samples`, with alpha-to-coverage
    /// where `coverage`; the runtime's refusal where it refuses it.
    fn pipeline(
        &self,
        (vertex, fragment): (&str, &str),
        color: ColorAttachmentDescriptor,
        samples: u32,
        coverage: bool,
    ) -> Result<metal2vulkan_runtime::RenderPipelineState, Error> {
        let d = self.descriptor((vertex, fragment), color, samples, coverage);
        self.device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
    }

    /// The descriptor of [`Lane::pipeline`]'s state.
    fn descriptor(
        &self,
        (vertex, fragment): (&str, &str),
        color: ColorAttachmentDescriptor,
        samples: u32,
        coverage: bool,
    ) -> RenderPipelineDescriptor {
        let mut d = RenderPipelineDescriptor::new(self.function(vertex));
        d.fragment = Some(self.function(fragment));
        d.color_attachments[0] = Some(color);
        d.raster_sample_count = samples;
        d.alpha_to_coverage = coverage;
        d
    }

    /// A pass into `target` (resolved into `resolve`), loaded or cleared to `clear`, drawing
    /// `values` through `pipeline` where one is given.
    fn pass(
        &self,
        target: &Texture,
        resolve: Option<&Texture>,
        clear: Option<[f64; 4]>,
        draw: Option<(&metal2vulkan_runtime::RenderPipelineState, &[f32])>,
    ) {
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(target.clone());
        match clear {
            Some([r, g, b, alpha]) => {
                a.load_action = LoadAction::Clear;
                a.clear_color = ClearColor::new(r, g, b, alpha);
            }
            None => a.load_action = LoadAction::Load,
        }
        match resolve {
            Some(r) => {
                a.resolve_texture = Some(r.clone());
                a.store_action = StoreAction::MultisampleResolve;
            }
            None => a.store_action = StoreAction::Store,
        }
        let held = draw.map(|(_, values)| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            buffer_with(&self.device, &bytes).unwrap()
        });
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        if let (Some((pipeline, _)), Some(values)) = (draw, &held) {
            e.set_render_pipeline_state(pipeline).unwrap();
            let [r, g, b, alpha] = BLEND_COLOR;
            e.set_blend_color(r, g, b, alpha);
            e.set_fragment_buffer(Some(values), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// A pass into `target` (resolved into `resolve`), loaded or cleared to `clear`, drawing
    /// [`COLOR`] through `pipeline`, a function that reads its attachment and writes what it read
    /// at each texel to buffer 1 (`a8r_fetch`): those reads; the draw's refusal where the runtime
    /// refuses it.
    fn fetch_pass(
        &self,
        target: &Texture,
        resolve: Option<&Texture>,
        clear: Option<[f64; 4]>,
        pipeline: &metal2vulkan_runtime::RenderPipelineState,
    ) -> Result<Vec<[f32; 4]>, Error> {
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(target.clone());
        match clear {
            Some([r, g, b, alpha]) => {
                a.load_action = LoadAction::Clear;
                a.clear_color = ClearColor::new(r, g, b, alpha);
            }
            None => a.load_action = LoadAction::Load,
        }
        match resolve {
            Some(r) => {
                a.resolve_texture = Some(r.clone());
                a.store_action = StoreAction::MultisampleResolve;
            }
            None => a.store_action = StoreAction::Store,
        }
        let color: Vec<u8> = COLOR.iter().flat_map(|v| v.to_le_bytes()).collect();
        let color = buffer_with(&self.device, &color).unwrap();
        let out = buffer_with(&self.device, &vec![0xee; 16 * TEXELS]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(pipeline).unwrap();
        e.set_fragment_buffer(Some(&color), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&out), 0, 1).unwrap();
        let drawn = e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        drawn?;
        let bytes = read_buffer(&out, 0, 16 * TEXELS as u64).unwrap();
        Ok(bytes
            .chunks(16)
            .map(|t| {
                std::array::from_fn(|i| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap()))
            })
            .collect())
    }

    /// Whether the device's fetch of a multisampled attachment reads its sample planes: the pixel
    /// interlock (no rasterization-order color attachment access, with which a fetch reads input
    /// attachments) on a device without multisampled storage images (`sample_planes.rs`).
    fn reads_planes(&self) -> bool {
        let profile = self.device.compile_device().profile();
        !profile.fragment().rasterization_order_color_access
            && !profile.images().storage_multisample
    }

    /// `kernel` of `a8r_buffers` over a `grid` of [`TEXELS`] threads (one row for a texture buffer,
    /// [`SIDE`] rows for the argument buffer's textures), with `colors` at buffer 0, the texture buffer
    /// `t` at texture 0 where one is given, `held` at buffer 2 where given, with the textures it
    /// names resident: buffer 1's `float4`s.
    fn buffer_dispatch(
        &self,
        kernel: &str,
        grid: [u64; 2],
        colors: &[[f32; 4]],
        t: Option<&Texture>,
        held: Option<(&Buffer, &[&Texture])>,
    ) -> Vec<[f32; 4]> {
        let function = self
            .buffers
            .new_function_with_name(kernel)
            .unwrap()
            .function_descriptor();
        let state = self
            .device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(function)),
                &PipelineOptions::default(),
            )
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
        let colors: Vec<u8> = colors
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let colors = buffer_with(&self.device, &colors).unwrap();
        let out = buffer_with(&self.device, &vec![0xee; 16 * TEXELS]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&colors), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        if let Some(t) = t {
            encoder.set_texture(Some(t), 0).unwrap();
        }
        if let Some((held, textures)) = held {
            encoder.set_buffer(Some(held), 0, 2).unwrap();
            for t in textures {
                let usage = ResourceUsage::READ | ResourceUsage::WRITE;
                encoder.use_resource(t.as_resource(), usage).unwrap();
            }
        }
        let [x, y] = grid;
        encoder
            .dispatch_threads(Size::new(x, y, 1), Size::new(x, y, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let bytes = read_buffer(&out, 0, 16 * TEXELS as u64).unwrap();
        bytes
            .chunks(16)
            .map(|t| {
                std::array::from_fn(|i| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap()))
            })
            .collect()
    }

    /// A texture buffer of [`TEXELS`] `A8Unorm` texels of `usage`, made with `swizzle`, over a
    /// shared buffer holding `bytes`, and the buffer.
    fn texture_buffer(
        &self,
        usage: TextureUsage,
        swizzle: TextureSwizzleChannels,
        bytes: &[u8],
    ) -> (Texture, Buffer) {
        let buffer = buffer_with(&self.device, bytes).unwrap();
        let options = buffer.resource_options();
        let mut d =
            TextureDescriptor::texture_buffer(PixelFormat::A8Unorm, TEXELS as u64, options, usage);
        d.swizzle = swizzle;
        let t = buffer
            .new_texture(&d, 0, TEXELS as u64)
            .unwrap_or_else(|e| panic!("an A8Unorm texture buffer: {e}"));
        (t, buffer)
    }

    /// The byte a draw of [`SOURCE`] stores over [`BELOW`] under `blend`; `None` where the device
    /// refuses the pipeline as a need of its own (`dualSrcBlend`).
    fn blended(&self, (op, src, dst, ..): Blend) -> Option<u8> {
        let mut c = ColorAttachmentDescriptor::new(PixelFormat::A8Unorm);
        c.blending_enabled = true;
        c.source_rgb_blend_factor = BlendFactor::OneMinusDestinationColor;
        c.destination_rgb_blend_factor = BlendFactor::SourceAlphaSaturated;
        c.rgb_blend_operation = BlendOperation::Add;
        c.source_alpha_blend_factor = src;
        c.destination_alpha_blend_factor = dst;
        c.alpha_blend_operation = op;
        let dual = [src, dst].iter().any(|f| {
            matches!(
                f,
                BlendFactor::Source1Color
                    | BlendFactor::OneMinusSource1Color
                    | BlendFactor::Source1Alpha
                    | BlendFactor::OneMinusSource1Alpha
            )
        });
        let fragment = if dual { "a8r_dual" } else { "a8r_fill" };
        let pipeline = match self.pipeline(("a8r_vertex", fragment), c, 1, false) {
            Ok(p) => p,
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => return None,
            Err(e) => panic!("{op:?} {src:?} {dst:?}: {e}"),
        };
        let target = self.texture(PixelFormat::A8Unorm, 1, TextureUsage::RENDER_TARGET, BELOW);
        self.pass(&target, None, None, Some((&pipeline, &SOURCE)));
        let bytes = self.bytes(&target);
        assert!(
            bytes.iter().all(|&b| b == bytes[0]),
            "{op:?} {src:?} {dst:?}: {bytes:?}"
        );
        Some(bytes[0])
    }
}

/// Every texel of `got` is `want`, else the first that differs.
fn every(what: &str, got: &[u8], want: u8) {
    if let Some(i) = got.iter().position(|&b| b != want) {
        panic!("{what}: texel {i} holds {}, Metal's {want}", got[i]);
    }
}

/// A kernel's writes store the texel's alpha, and its reads, samples and `read_write` reads give
/// the byte as alpha, `(0, 0, 0, a)`, as Metal's do.
#[test]
fn a8_without_a8_unorm_writes_and_reads_alpha_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::new(&messages, &cache, true);
    assert!(
        lane.in_red(),
        "{}: A8Unorm is not held in red",
        lane.device.name()
    );
    let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    for kernel in ["a8r_write", "a8r_write_half"] {
        let t = lane.texture(PixelFormat::A8Unorm, 1, usage, 0x11);
        lane.dispatch(kernel, &t, COLOR);
        every(kernel, &lane.bytes(&t), STORED);
    }
    let alpha = [0.0, 0.0, 0.0, 153.0 / 255.0];
    for kernel in ["a8r_read", "a8r_sample"] {
        let t = lane.texture(PixelFormat::A8Unorm, 1, usage, 153);
        let got = lane.dispatch(kernel, &t, COLOR);
        assert!(got.iter().all(|&v| v == alpha), "{kernel}: {got:?}");
    }
    let t = lane.texture(PixelFormat::A8Unorm, 1, usage, 153);
    let got = lane.dispatch("a8r_read_write", &t, [0.1, 0.3, 0.5, 0.7]);
    assert!(got.iter().all(|&v| v == alpha), "a8r_read_write: {got:?}");
    every("a8r_read_write", &lane.bytes(&t), 178);
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// An `A8Unorm` storage texture's reads on the device made without
/// `shaderStorageImageReadWithoutFormat` (as Intel's ANV is), with its own `A8_UNORM` where it has
/// the format: no SPIR-V format names it, so its reads are declared without a format, which the
/// format's own `STORAGE_READ_WITHOUT_FORMAT` bit allows under `VK_KHR_format_feature_flags2`,
/// the device's enable at the runtime's Vulkan 1.2 instance in place of the feature: a read and a
/// `read_write` read of the byte 153 are `(0, 0, 0, 0.6)`, and that slot's write of
/// (0.1, 0.3, 0.5, 0.7) stores 178, as Metal's (probe `a8`). A device without the extension, or
/// whose `A8_UNORM` lacks the bit, refuses the kernel naming what it lacks.
#[test]
fn a8_storage_reads_without_the_feature_go_by_the_format_s_bit() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let feature = VkFeature::ShaderStorageImageReadWithoutFormat;
    let lane = Lane::withholding(&messages, &cache, false, &[feature]);
    let name = lane.device.name().to_owned();
    if lane.in_red() {
        println!("A8-FORMATLESS {name}: no A8_UNORM, its own format not checked");
        drop(lane);
        layer_reported_nothing_at_all(&messages);
        return;
    }
    let profile = lane.device.compile_device().profile();
    let enabled = profile.enabled();
    assert!(!enabled.features.contains(&feature), "{name}: {feature:?}");
    let flags2 = enabled
        .extensions
        .contains(&VkExtension::KhrFormatFeatureFlags2);
    let bit = profile.supports_format(
        metal2vulkan::vk::Format::A8_UNORM,
        metal2vulkan::device::FormatUsage::StorageImageReadWithoutFormat,
    );
    let made = |kernel: &str| {
        lane.device.new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(lane.function(kernel)),
            &PipelineOptions::default(),
        )
    };
    if !(flags2 && bit) {
        for kernel in ["a8r_read", "a8r_read_write"] {
            let Err(Error::Unsupported(refused)) = made(kernel) else {
                panic!("{name}: {kernel} made without the feature, the extension or the bit");
            };
            println!(
                "A8-FORMATLESS {name}: {kernel} refused (flags2 {flags2}, bit {bit}): {refused}"
            );
        }
        drop(lane);
        layer_reported_nothing_at_all(&messages);
        return;
    }
    let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let alpha = [0.0, 0.0, 0.0, 153.0 / 255.0];
    for kernel in ["a8r_read", "a8r_read_write"] {
        made(kernel).unwrap_or_else(|e| panic!("{name}: {kernel} by the format's bit: {e}"));
        let t = lane.texture(PixelFormat::A8Unorm, 1, usage, 153);
        let got = lane.dispatch(kernel, &t, [0.1, 0.3, 0.5, 0.7]);
        assert!(got.iter().all(|&v| v == alpha), "{name}: {kernel}: {got:?}");
        if kernel == "a8r_read_write" {
            every(&format!("{name}: {kernel}"), &lane.bytes(&t), 178);
        }
    }
    println!("A8-FORMATLESS {name}: read by A8_UNORM's own bit, as Metal's");
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// A render stores the fragment's alpha, a clear the color's, only the alpha bit of a write mask
/// writes, and each blend of [`BLENDS`] stores Metal's byte (within a code), and the
/// device's own `A8_UNORM` texture's where it has the format.
#[test]
fn a8_without_a8_unorm_renders_masks_clears_and_blends_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::new(&messages, &cache, true);
    let name = lane.device.name().to_owned();
    assert!(lane.in_red(), "{name}: A8Unorm is not held in red");
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let unblended = |mask: ColorWriteMask| {
        let mut c = ColorAttachmentDescriptor::new(PixelFormat::A8Unorm);
        c.write_mask = mask;
        lane.pipeline(("a8r_vertex", "a8r_fill"), c, 1, false)
            .unwrap()
    };
    let t = lane.texture(PixelFormat::A8Unorm, 1, usage, 0x11);
    lane.pass(
        &t,
        None,
        None,
        Some((&unblended(ColorWriteMask::ALL), &COLOR)),
    );
    every("a render", &lane.bytes(&t), STORED);
    lane.pass(&t, None, Some(COLOR.map(f64::from)), None);
    every("a clear", &lane.bytes(&t), STORED);
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    for (mask, want) in [
        (rgb, 0x11),
        (ColorWriteMask::ALPHA, STORED),
        (ColorWriteMask::RED, 0x11),
        (ColorWriteMask::empty(), 0x11),
    ] {
        let t = lane.texture(PixelFormat::A8Unorm, 1, usage, 0x11);
        lane.pass(&t, None, None, Some((&unblended(mask), &COLOR)));
        every(&format!("mask {mask:?}"), &lane.bytes(&t), want);
    }
    let mut got = Vec::new();
    let (mut wrong, mut refused, mut off_by_one) = (Vec::new(), 0, 0);
    for blend in BLENDS {
        let (op, src, dst, want) = blend;
        let Some(byte) = lane.blended(blend) else {
            refused += 1;
            got.push(None);
            continue;
        };
        got.push(Some(byte));
        match byte.abs_diff(want) {
            0 => {}
            1 => off_by_one += 1,
            _ => wrong.push(format!("{op:?} {src:?} {dst:?}: {byte}, Metal's {want}")),
        }
    }
    drop(lane);
    layer_reported_nothing_at_all(&messages);
    // The same blends into the device's own `A8_UNORM` texture, where it has the format.
    let own = Lane::new(&messages, &cache, false);
    let mut unlike = Vec::new();
    if !own.in_red() {
        for (blend, held) in BLENDS.into_iter().zip(&got) {
            let (op, src, dst, ..) = blend;
            if let (Some(direct), Some(held)) = (own.blended(blend), held) {
                if direct != *held {
                    unlike.push(format!(
                        "{op:?} {src:?} {dst:?}: {held}, A8_UNORM's {direct}"
                    ));
                }
            }
        }
    }
    drop(own);
    layer_reported_nothing_at_all(&messages);
    for line in wrong.iter().chain(&unlike) {
        println!("A8-IN-RED-DIFFERS {name}: {line}");
    }
    println!(
        "A8-IN-RED {name}: {} blends checked, {refused} refused, {off_by_one} a code from \
         Metal's, {} further, {} differ from A8_UNORM's",
        BLENDS.len() - refused,
        wrong.len(),
        unlike.len()
    );
    assert!(wrong.is_empty() && unlike.is_empty());
    assert!(refused < BLENDS.len() / 4, "{refused} refused");
}

/// A 4-sample render over the lower left half of the target resolves as `R8Unorm`'s does, at the
/// edge too but where the device rounds the average its own way (`device-rounded-resolve`, within
/// one code of Metal's), which may round the edge's tie a code apart, and as Metal's away from it
/// (229 below it, 0 above), where every texel's samples agree: on the device without `A8_UNORM` (its `R8_UNORM` image's resolve), and with its own
/// where it has the format, which the device's average resolves as it does `R8Unorm` (probe `a8`:
/// an `A8Unorm` resolve is `RGBA8Unorm`'s alpha). Where the fill's stage records
/// `device-color-conversion`, a texel below it may hold the code above Metal's, counted.
#[test]
fn a8_resolves_as_r8_unorm() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    for withheld in [true, false] {
        let lane = Lane::new(&messages, &cache, withheld);
        let name = lane.device.name().to_owned();
        if withheld {
            assert!(lane.in_red(), "{name}: A8Unorm is not held in red");
        } else if lane.in_red() {
            println!("A8-RESOLVE {name}: no A8_UNORM, its own format not checked");
            continue;
        }
        let form = if withheld { "R8_UNORM" } else { "A8_UNORM" };
        a8_resolve_leg(&lane, &format!("{name} {form}"));
        drop(lane);
        layer_reported_nothing_at_all(&messages);
    }
}

/// One leg of [`a8_resolves_as_r8_unorm`] on `lane`, named `what`.
fn a8_resolve_leg(lane: &Lane, what: &str) {
    let kind = resolve_kind(&lane.device, 4);
    let resolved = |pf: PixelFormat| {
        let ms = lane.texture(pf, 4, TextureUsage::RENDER_TARGET, 0);
        let one = lane.texture(
            pf,
            1,
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            0,
        );
        let pipeline = lane
            .pipeline(
                ("a8r_half_vertex", "a8r_fill"),
                ColorAttachmentDescriptor::new(pf),
                4,
                false,
            )
            .unwrap();
        lane.pass(
            &ms,
            Some(&one),
            Some([0.0; 4]),
            Some((&pipeline, &[0.9; 4])),
        );
        lane.bytes(&one)
    };
    let a8 = resolved(PixelFormat::A8Unorm);
    let r8 = resolved(PixelFormat::R8Unorm);
    // An edge texel's samples disagree, and where the device rounds the average its own way
    // (`device-rounded-resolve`: within one code of Metal's) it may round `A8Unorm`'s tie apart
    // from `R8Unorm`'s (RADV on the RX 550: 114 and 115 of 114.5). Every other texel is the same.
    let rounded = kind == Some(Degradation::DeviceRoundedResolve);
    let mut ties = 0usize;
    for (i, (&a, &r)) in a8.iter().zip(&r8).enumerate() {
        let edge = i % SIDE as usize == i / SIDE as usize;
        if edge && rounded && a.abs_diff(r) == 1 {
            ties += 1;
            continue;
        }
        assert_eq!(
            a, r,
            "{what}: A8Unorm's resolve against R8Unorm's at texel {i}: {a8:?}, {r8:?}"
        );
    }
    // Where the fill's stage leaves its color's conversion to the device
    // (`device-color-conversion`: the device's rounding of the product, where Metal rounds the
    // exact one to nearest), 0.9 may store a code above Metal's 229.
    let descriptor = lane.descriptor(
        ("a8r_half_vertex", "a8r_fill"),
        ColorAttachmentDescriptor::new(PixelFormat::A8Unorm),
        4,
        false,
    );
    let converts =
        made_render_kinds(&lane.device, &descriptor).contains(&Degradation::DeviceColorConversion);
    let mut device_codes = 0usize;
    for y in 0..SIDE as usize {
        for x in 0..SIDE as usize {
            let want = match x.cmp(&y) {
                std::cmp::Ordering::Less => 229,
                std::cmp::Ordering::Greater => 0,
                std::cmp::Ordering::Equal => continue,
            };
            let got = a8[y * SIDE as usize + x];
            if converts && got != want && got.abs_diff(want) == 1 {
                device_codes += 1;
                continue;
            }
            assert_eq!(got, want, "{what}: texel ({x}, {y}): {a8:?}");
        }
    }
    println!(
        "A8-RESOLVE {what}: as R8Unorm's ({kind:?}) but {ties} edge texels a code apart, \
         {device_codes} texels a code from Metal's (the device's color conversion: {converts})"
    );
}

/// The fetches of the probe (`a8`'s uses): a read of the byte 153 through `[[color(0)]]`, every
/// texel `(0, 0, 0, 0.6)`, then a write of [`COLOR`] under each write mask, the byte left (only the
/// alpha bit writes); and at four samples, a target cleared to 0.6 read so at every sample and
/// resolved to the write's 204.
const FETCH_MASKS: [(ColorWriteMask, u8); 3] = [
    (ColorWriteMask::ALPHA, STORED),
    (
        ColorWriteMask::RED
            .union(ColorWriteMask::GREEN)
            .union(ColorWriteMask::BLUE),
        153,
    ),
    (ColorWriteMask::empty(), 153),
];

/// A color input (`[[color(0)]]`) of an `A8Unorm` attachment reads Metal's alpha in `.a`, and the
/// function's write after it stores the alpha as the write mask lets it, at one sample and at four,
/// as Metal's do: on the device without `A8_UNORM` (its `R8_UNORM` image read through the input
/// attachment or the interlocked storage form, whichever the device orders fetch with), and with
/// its own `A8_UNORM` where it has the format. A device without a fetch form skips it
/// ([`fetches`]).
#[test]
fn a8_color_input_reads_alpha_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    for withheld in [true, false] {
        let lane = Lane::new(&messages, &cache, withheld);
        if !fetches(&lane.device) {
            continue;
        }
        let name = lane.device.name().to_owned();
        if withheld {
            assert!(lane.in_red(), "{name}: A8Unorm is not held in red");
        } else if lane.in_red() {
            println!("A8-FETCH {name}: no A8_UNORM, its own format not checked");
            continue;
        }
        let form = if withheld { "R8_UNORM" } else { "A8_UNORM" };
        let alpha = [0.0, 0.0, 0.0, 153.0 / 255.0];
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        for (mask, want) in FETCH_MASKS {
            let mut c = ColorAttachmentDescriptor::new(PixelFormat::A8Unorm);
            c.write_mask = mask;
            let pipeline = lane
                .pipeline(("a8r_vertex", "a8r_fetch"), c, 1, false)
                .unwrap_or_else(|e| panic!("{name} {form}: a fetch under {mask:?}: {e}"));
            let t = lane.texture(PixelFormat::A8Unorm, 1, usage, 153);
            let read = lane
                .fetch_pass(&t, None, None, &pipeline)
                .unwrap_or_else(|e| panic!("{name} {form}: a fetch under {mask:?}: {e}"));
            assert!(
                read.iter().all(|&v| v == alpha),
                "{name} {form}: the fetch under {mask:?} read {read:?}"
            );
            every(
                &format!("{name} {form}: the write under {mask:?}"),
                &lane.bytes(&t),
                want,
            );
        }
        let pipeline = lane
            .pipeline(
                ("a8r_vertex", "a8r_fetch"),
                ColorAttachmentDescriptor::new(PixelFormat::A8Unorm),
                4,
                false,
            )
            .unwrap_or_else(|e| panic!("{name} {form}: a fetch at four samples: {e}"));
        let ms = lane.texture(PixelFormat::A8Unorm, 4, TextureUsage::RENDER_TARGET, 0);
        let one = lane.texture(PixelFormat::A8Unorm, 1, usage, 0);
        // Through sample planes where the device reads them, `A8_UNORM`'s held in its own format
        // (`bind::PlanesCopy::Alpha8`), as its bits copy holds an `R8_UNORM` image's.
        let planes = if lane.reads_planes() {
            "through sample planes"
        } else {
            "directly"
        };
        let ran = lane.fetch_pass(&ms, Some(&one), Some([0.6; 4]), &pipeline);
        let read =
            ran.unwrap_or_else(|e| panic!("{name} {form}: a fetch at four samples, {planes}: {e}"));
        assert!(
            read.iter().all(|&v| v == alpha),
            "{name} {form}: the fetch at four samples read {read:?}"
        );
        every(
            &format!("{name} {form}: the resolve"),
            &lane.bytes(&one),
            STORED,
        );
        println!("A8-FETCH {name} {form}: as Metal's, at four samples {planes}");
        drop(lane);
        layer_reported_nothing_at_all(&messages);
    }
}

/// [`a8_color_input_reads_alpha_as_metal`]'s four-sample fetch through sample planes on the device
/// made as Intel's ANV is for this (without `rasterizationOrderColorAttachmentAccess`,
/// `shaderStorageImageMultisample` and `shaderStorageImageReadWithoutFormat`), with its own
/// `A8_UNORM`: the fetch reads the planes in the pixel interlock, declared without a format by the
/// format's own bit, and the planes are held in `A8_UNORM` itself (`bind::PlanesCopy::Alpha8`), as
/// no unsigned view reads its class. A target cleared to 0.6 reads `(0, 0, 0, 0.6)` at every
/// sample, and the write of [`COLOR`] after it resolves to 204, as Metal's (probe `a8`'s uses). A
/// device without a fetch form skips it ([`fetches`]).
#[test]
fn a8_color_input_through_sample_planes_reads_alpha_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let withheld = [
        VkFeature::RasterizationOrderColorAttachmentAccess,
        VkFeature::ShaderStorageImageMultisample,
        VkFeature::ShaderStorageImageReadWithoutFormat,
    ];
    let lane = Lane::withholding(&messages, &cache, false, &withheld);
    let name = lane.device.name().to_owned();
    if lane.in_red() || !lane.reads_planes() {
        println!(
            "A8-PLANES {name}: no A8_UNORM, or no sample planes (in red {}, planes {})",
            lane.in_red(),
            lane.reads_planes()
        );
        drop(lane);
        layer_reported_nothing_at_all(&messages);
        return;
    }
    if !fetches(&lane.device) {
        return;
    }
    let pipeline = lane
        .pipeline(
            ("a8r_vertex", "a8r_fetch"),
            ColorAttachmentDescriptor::new(PixelFormat::A8Unorm),
            4,
            false,
        )
        .unwrap_or_else(|e| panic!("{name}: a fetch at four samples: {e}"));
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let ms = lane.texture(PixelFormat::A8Unorm, 4, TextureUsage::RENDER_TARGET, 0);
    let one = lane.texture(PixelFormat::A8Unorm, 1, usage, 0);
    let read = lane
        .fetch_pass(&ms, Some(&one), Some([0.6; 4]), &pipeline)
        .unwrap_or_else(|e| panic!("{name}: a fetch at four samples through sample planes: {e}"));
    let alpha = [0.0, 0.0, 0.0, 153.0 / 255.0];
    assert!(
        read.iter().all(|&v| v == alpha),
        "{name}: the fetch through sample planes read {read:?}"
    );
    every(&format!("{name}: the resolve"), &lane.bytes(&one), STORED);
    println!("A8-PLANES {name}: as Metal's, through A8_UNORM sample planes");
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// Alpha-to-coverage through an `A8Unorm` attachment 0 at four samples covers by the output's
/// alpha as through an `RGBA8Unorm` one, as Metal's does (probe `a8`'s uses: at every alpha in
/// 1/32 steps, the masks and the resolved bytes of the two are the same, every texel 0 at alpha 0
/// and 255 at alpha 1): the output (0.9, 0.1, 0.5, a), whose blue differs from its alpha, over a
/// target cleared to 0, resolved, against the resolved alpha of the same draw into `RGBA8Unorm`,
/// on the device without `A8_UNORM` and with its own where it has the format. How the alpha
/// becomes a mask is the device's (Vulkan names no algorithm), so the bytes are held to
/// `RGBA8Unorm`'s on the same device: exactly where both resolve by Metal's rule, else within the
/// code the device's average may round a tie to (`device-rounded-resolve`), which still tells
/// every count of covered samples
/// apart (adjacent counts lie a quarter of the alpha's code apart, two codes or more past alpha 0).
#[test]
fn a8_alpha_to_coverage_covers_as_rgba8() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    for withheld in [true, false] {
        let lane = Lane::new(&messages, &cache, withheld);
        let name = lane.device.name().to_owned();
        if withheld {
            assert!(lane.in_red(), "{name}: A8Unorm is not held in red");
        } else if lane.in_red() {
            println!("A8-COVERAGE {name}: no A8_UNORM, its own format not checked");
            continue;
        }
        let form = if withheld { "R8_UNORM" } else { "A8_UNORM" };
        let resolved = |pf: PixelFormat, alpha: f32| {
            let pipeline = lane
                .pipeline(
                    ("a8r_vertex", "a8r_fill"),
                    ColorAttachmentDescriptor::new(pf),
                    4,
                    true,
                )
                .unwrap_or_else(|e| panic!("{name} {form}: {pf:?} under coverage: {e}"));
            let ms = lane.bare(pf, 4, TextureUsage::RENDER_TARGET);
            let one = lane.bare(
                pf,
                1,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            );
            lane.pass(
                &ms,
                Some(&one),
                Some([0.0; 4]),
                Some((&pipeline, &[0.9, 0.1, 0.5, alpha])),
            );
            one
        };
        let rounded = resolve_kind(&lane.device, 4) == Some(Degradation::DeviceRoundedResolve);
        let codes = u8::from(rounded);
        for step in 0..=32u8 {
            let alpha = f32::from(step) / 32.0;
            let a8 = lane.bytes(&resolved(PixelFormat::A8Unorm, alpha));
            let rgba = lane.wide_bytes(&resolved(PixelFormat::RGBA8Unorm, alpha), 4);
            let rgba_alpha: Vec<u8> = rgba.chunks(4).map(|t| t[3]).collect();
            let apart = a8
                .iter()
                .zip(&rgba_alpha)
                .position(|(a, b)| a.abs_diff(*b) > codes);
            assert_eq!(
                apart, None,
                "{name} {form}: alpha {step}/32, A8Unorm's resolve against RGBA8Unorm's alpha \
                 (within {codes} codes): {a8:?} against {rgba_alpha:?}"
            );
            match step {
                0 => every(&format!("{name} {form}: alpha 0"), &a8, 0),
                32 => every(&format!("{name} {form}: alpha 1"), &a8, 255),
                _ => {}
            }
        }
        println!("A8-COVERAGE {name} {form}: as RGBA8Unorm's at 33 alphas, within {codes} codes");
        drop(lane);
        layer_reported_nothing_at_all(&messages);
    }
}

/// A texture buffer's dispatch: one row of [`TEXELS`] threads.
const ROW: [u64; 2] = [TEXELS as u64, 1];

/// The bytes of a texture buffer's reads: 152 at the even texels, 0 at the odd ones.
fn buffer_bytes() -> Vec<u8> {
    (0..TEXELS)
        .map(|i| if i % 2 == 0 { 152 } else { 0 })
        .collect()
}

/// An `A8Unorm` texture buffer reads its bytes as alpha, `(0, 0, 0, a)`, as `float` and as `half`,
/// through its swizzle where it was made with one, stores a written texel's alpha, and a
/// `read_write` one does both, as Metal's does (probe `a8`'s texture buffers, an M3 Max under
/// macOS 27.0.1: a read of the byte 152 is (0, 0, 0, 0.596078455), as `half` (0, 0, 0, 0.596191406),
/// of the byte 0 all zero; made with the swizzle (A, 0, 0, 1) the read is (0.596078455, 0, 0, 1),
/// with (A, A, R, G) (0.596078455, 0.596078455, 0, 0); a write of (0.2, 0.4, 0.6, 0.8) stores 204;
/// a `read_write` read of 152 is (0, 0, 0, 0.596078455) and its write of (0.1, 0.3, 0.5, 0.7)
/// stores 178. Metal's validation refuses a swizzle with `ShaderWrite`), through the bytes'
/// `R8_UNORM` view on the device without `A8_UNORM`, and through its own where it has the format.
/// A read is within the device's conversion of the byte where it does not sample through Metal's
/// own texture unit (`converted_holds`), and a `half` read within a `half` unit in the last place
/// where its stage records the device's rounding (`half_within`, `native-half-conversion`).
#[test]
fn a8_texture_buffer_reads_and_writes_alpha_as_metal() {
    use TextureSwizzle::{Alpha as A, Green as G, One as O, Red as R, Zero as Z};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    for withheld in [true, false] {
        let lane = Lane::new(&messages, &cache, withheld);
        let name = lane.device.name().to_owned();
        if withheld {
            assert!(lane.in_red(), "{name}: A8Unorm is not held in red");
        } else if lane.in_red() {
            println!("A8-BUFFER {name}: no A8_UNORM, its own format not checked");
            continue;
        }
        let form = if withheld { "R8_UNORM" } else { "A8_UNORM" };
        let a = 152.0 / 255.0;
        // The half nearest 152 / 255.
        let half = 1221.0 / 2048.0;
        let reads = [
            (
                "a8r_buffer_read",
                TextureSwizzleChannels::DEFAULT,
                [0.0, 0.0, 0.0, a],
            ),
            (
                "a8r_buffer_read_half",
                TextureSwizzleChannels::DEFAULT,
                [0.0, 0.0, 0.0, half],
            ),
            (
                "a8r_buffer_read",
                TextureSwizzleChannels::new(A, Z, Z, O),
                [a, 0.0, 0.0, 1.0],
            ),
            (
                "a8r_buffer_read_half",
                TextureSwizzleChannels::new(A, Z, Z, O),
                [half, 0.0, 0.0, 1.0],
            ),
            (
                "a8r_buffer_read",
                TextureSwizzleChannels::new(A, A, R, G),
                [a, a, 0.0, 0.0],
            ),
        ];
        // Where the device does not sample through Metal's own texture unit a `float` read is its
        // conversion of the byte; where the `half` read's stage records the device's rounding, it
        // is within a `half` unit in the last place of Metal's.
        let exact = metal_sampler(&lane.device);
        let fixture = llvm_assemble(&engine_dir().join("a8r_buffers.ll")).unwrap();
        let native_half = stage_notes(&lane.device, &fixture, "a8r_buffer_read_half")
            .contains(&PrecisionNote::NativeHalfConversion);
        for (kernel, swizzle, want) in reads {
            let (t, _bytes) =
                lane.texture_buffer(TextureUsage::SHADER_READ, swizzle, &buffer_bytes());
            let got = lane.buffer_dispatch(kernel, ROW, &[COLOR], Some(&t), None);
            for (i, v) in got.iter().enumerate() {
                let want = match i % 2 {
                    0 => want,
                    _ => want.map(|c| if c == 1.0 { 1.0 } else { 0.0 }),
                };
                let holds = (0..4).all(|c| match kernel {
                    "a8r_buffer_read_half" if native_half => half_within(v[c], want[c]),
                    _ => converted_holds(v[c], want[c], exact),
                });
                assert!(
                    holds,
                    "{name} {form}: {kernel} {swizzle:?} at texel {i}: {v:?}, Metal's {want:?}"
                );
            }
        }
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let (t, bytes) =
            lane.texture_buffer(usage, TextureSwizzleChannels::DEFAULT, &[0x11; TEXELS]);
        lane.buffer_dispatch("a8r_buffer_write", ROW, &[COLOR], Some(&t), None);
        let stored = read_buffer(&bytes, 0, TEXELS as u64).unwrap();
        every(&format!("{name} {form}: a8r_buffer_write"), &stored, STORED);
        let (t, bytes) =
            lane.texture_buffer(usage, TextureSwizzleChannels::DEFAULT, &[152; TEXELS]);
        let got = lane.buffer_dispatch(
            "a8r_buffer_read_write",
            ROW,
            &[[0.1, 0.3, 0.5, 0.7]],
            Some(&t),
            None,
        );
        assert!(
            got.iter().all(|&v| v == [0.0, 0.0, 0.0, a]),
            "{name} {form}: a8r_buffer_read_write read {got:?}"
        );
        let stored = read_buffer(&bytes, 0, TEXELS as u64).unwrap();
        every(
            &format!("{name} {form}: a8r_buffer_read_write"),
            &stored,
            178,
        );
        println!("A8-BUFFER {name} {form}: as Metal's");
        drop(lane);
        layer_reported_nothing_at_all(&messages);
    }
}

/// The storage textures an argument buffer holds read an `A8Unorm` texture's byte as alpha,
/// `(0, 0, 0, a)`, and store a written texel's alpha, as Metal's do (probe `a8`'s argument buffers,
/// an M3 Max under macOS 27.0.1: a `read_write` member of the byte 153 reads (0, 0, 0, 0.6) and
/// its write of (0.8, 0.6, 0.4, 0.2) stores 51, and a `write` member's write of
/// (0.2, 0.4, 0.6, 0.8) stores 204), through the texture's `R8_UNORM` image on the device without
/// `A8_UNORM`, and its own where it has the format. A device without
/// `shaderStorageImageReadWithoutFormat` (ANV) refuses the kernel, `ArgumentBufferTextures`.
#[test]
fn a8_argument_buffer_storage_reads_and_writes_alpha_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    for withheld in [true, false] {
        let lane = Lane::new(&messages, &cache, withheld);
        let name = lane.device.name().to_owned();
        if withheld {
            assert!(lane.in_red(), "{name}: A8Unorm is not held in red");
        } else if lane.in_red() {
            println!("A8-HELD {name}: no A8_UNORM, its own format not checked");
            continue;
        }
        let form = if withheld { "R8_UNORM" } else { "A8_UNORM" };
        // A heap's storage reads are declared without a format and need the device's feature
        // (its formats' own bits do not serve a heap, docs/FAST-PATHS.md): refused without it.
        let profile = lane.device.compile_device().profile();
        let feature = VkFeature::ShaderStorageImageReadWithoutFormat;
        if !profile.enabled().features.contains(&feature) {
            let function = lane
                .buffers
                .new_function_with_name("a8r_held")
                .unwrap()
                .function_descriptor();
            let made = lane.device.new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(function)),
                &PipelineOptions::default(),
            );
            let Err(Error::Unsupported(refused)) = made else {
                panic!("{name} {form}: the argument buffer's kernel made without {feature:?}");
            };
            assert_eq!(
                refused.feature,
                Feature::ArgumentBufferTextures,
                "{name} {form}: {refused}"
            );
            println!("A8-HELD {name} {form}: refused without {feature:?}: {refused}");
            drop(lane);
            layer_reported_nothing_at_all(&messages);
            continue;
        }
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let rw = lane.texture(PixelFormat::A8Unorm, 1, usage, 153);
        let w = lane.texture(PixelFormat::A8Unorm, 1, usage, 0x11);
        let ids: Vec<u8> = [&rw, &w]
            .iter()
            .flat_map(|t| t.gpu_resource_id().0.to_le_bytes())
            .collect();
        let held = buffer_with(&lane.device, &ids).unwrap();
        let got = lane.buffer_dispatch(
            "a8r_held",
            [SIDE, SIDE],
            &[[0.8, 0.6, 0.4, 0.2], COLOR],
            None,
            Some((&held, &[&rw, &w])),
        );
        let alpha = [0.0, 0.0, 0.0, 153.0 / 255.0];
        assert!(
            got.iter().all(|&v| v == alpha),
            "{name} {form}: the read_write member read {got:?}"
        );
        every(
            &format!("{name} {form}: the read_write member"),
            &lane.bytes(&rw),
            51,
        );
        every(
            &format!("{name} {form}: the write member"),
            &lane.bytes(&w),
            STORED,
        );
        println!("A8-HELD {name} {form}: as Metal's");
        drop(lane);
        layer_reported_nothing_at_all(&messages);
    }
}
