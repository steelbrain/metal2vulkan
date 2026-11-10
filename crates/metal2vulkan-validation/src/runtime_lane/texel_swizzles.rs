//! Texture buffers made with a swizzle (work-pile 0p) on the device the cfg selects, under the
//! validation layer: every read takes the swizzle, out of bounds too.
//!
//! Metal applies a texture buffer's swizzle to every read (an M3 Max under macOS 26.7 and an M4
//! under macOS 27.0.1, plain and validated alike). Each channel reads the texel's channel its
//! swizzle names, `Zero` 0 and `One` 1 (`1.0` for a float format, `1` for an integer one); a
//! format's missing green and blue read 0 and its missing alpha 1. A read past the texture buffer
//! reads the texel (0, 0, 0, 0), alpha 1 where the format has no alpha, swizzled alike. A view made
//! with a swizzle reads its own, replacing its texture's. A swizzle beside `ShaderWrite` usage is
//! refused when the texture is made, by Metal and by the runtime, so only reads see one.
//!
//! The kernels are public fixtures' (`validation/fixtures/public/engine`):
//!
//! - `texture_ops`'s `texel_buffers` writes twice the `float` texel of its slot-0 texture buffer
//!   at each thread's index into its slot-1 texture buffer. The source holds [`TEXELS`] texels
//!   and the grid [`GRID`] threads, so half the reads lie past it.
//! - `ob1_texel_buffer` writes the red channel of the `uint` texel of its slot-0 texture buffer
//!   at the index each thread is given, past the texture buffer for half of them.

use super::degrade_notices::counts;
use super::pixel_samplers::metal_sampler;
use super::{
    at_plan, buffer_with, converted_holds, layer_reported_nothing_at_all, read_buffer,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ComputePipelineState, Degradation, Device, PipelineOptions, ResourceOptions, Size,
    Texture, TextureDescriptor,
};

/// The texels of each source texture buffer.
const TEXELS: usize = 4;

/// The threads of each grid, one threadgroup: indices `TEXELS..GRID` read past the source.
const GRID: usize = 8;

/// The four swizzles the probes read, `bgr1`, `01ar`, `gggg` and `abgr`, and the identity.
fn swizzles() -> [(&'static str, TextureSwizzleChannels); 5] {
    use TextureSwizzle::{Alpha, Blue, Green, One, Red, Zero};
    let s = |red, green, blue, alpha| TextureSwizzleChannels {
        red,
        green,
        blue,
        alpha,
    };
    [
        ("rgba", TextureSwizzleChannels::DEFAULT),
        ("bgr1", s(Blue, Green, Red, One)),
        ("01ar", s(Zero, One, Alpha, Red)),
        ("gggg", s(Green, Green, Green, Green)),
        ("abgr", s(Alpha, Blue, Green, Red)),
    ]
}

/// A source texture buffer's format, its texels' bytes, and each texel's four channels as Metal
/// reads them unswizzled (a float format's as `f32` bits).
struct Source {
    pixel_format: PixelFormat,
    bytes: Vec<u8>,
    texels: Vec<[u32; 4]>,
    /// The format has an alpha channel: a read past the texture buffer reads alpha 0, else 1.
    alpha: bool,
    /// `One`'s value, and a missing alpha's: `1.0`'s bits for a float format, else 1.
    one: u32,
}

/// The float sources: 8-bit normalized RGBA, one and two 32-bit float channels.
fn float_sources() -> Vec<Source> {
    let one = 1.0f32.to_bits();
    let codes: Vec<u8> = (0..4 * TEXELS)
        .map(|i| ((7 * i + 3) & 0xff) as u8)
        .collect();
    let unorm = codes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| [0, 1, 2, 3].map(|i| (f32::from(c[i]) / 255.0).to_bits()))
        .collect();
    let floats = [1.5f32, -2.25, 3.0e5, 0.125, 7.0, -0.5, 1024.0, 3.75];
    let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    vec![
        Source {
            pixel_format: PixelFormat::RGBA8Unorm,
            bytes: codes.clone(),
            texels: unorm,
            alpha: true,
            one,
        },
        Source {
            pixel_format: PixelFormat::R32Float,
            bytes: bytes(&floats[..TEXELS]),
            texels: floats[..TEXELS]
                .iter()
                .map(|f| [f.to_bits(), 0, 0, one])
                .collect(),
            alpha: false,
            one,
        },
        Source {
            pixel_format: PixelFormat::RG32Float,
            bytes: bytes(&floats),
            texels: floats
                .as_chunks::<2>()
                .0
                .iter()
                .map(|f| [f[0].to_bits(), f[1].to_bits(), 0, one])
                .collect(),
            alpha: false,
            one,
        },
    ]
}

/// The unsigned sources: one 32-bit channel and four 16-bit ones.
fn uint_sources() -> Vec<Source> {
    let words = [0x1122_3344u32, 7, 0xffff_ffff, 0x8000_0000];
    let halves: Vec<u16> = (0..4 * TEXELS as u16)
        .map(|i| i.wrapping_mul(0x1357).wrapping_add(0x0203))
        .collect();
    vec![
        Source {
            pixel_format: PixelFormat::R32Uint,
            bytes: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            texels: words.iter().map(|&w| [w, 0, 0, 1]).collect(),
            alpha: false,
            one: 1,
        },
        Source {
            pixel_format: PixelFormat::RGBA16Uint,
            bytes: halves.iter().flat_map(|h| h.to_le_bytes()).collect(),
            texels: halves
                .as_chunks::<4>()
                .0
                .iter()
                .map(|h| h.iter().map(|&h| u32::from(h)).collect::<Vec<u32>>())
                .map(|h| [h[0], h[1], h[2], h[3]])
                .collect(),
            alpha: true,
            one: 1,
        },
    ]
}

impl Source {
    /// The channels Metal reads at `index` through `swizzle`, past the texels too.
    fn read(&self, index: usize, swizzle: TextureSwizzleChannels) -> [u32; 4] {
        let texel = match self.texels.get(index) {
            Some(&t) => t,
            None => [0, 0, 0, if self.alpha { 0 } else { self.one }],
        };
        [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha].map(|s| match s.as_raw() {
            0 => 0,
            1 => self.one,
            raw => texel[(raw - 2) as usize],
        })
    }

    /// Whether `got`, the `float4` `texel_buffers` writes for index `index` through `swizzle`, is
    /// twice Metal's read, bit for bit: an 8-bit normalized code's `code / 255` correctly
    /// rounded too (so no channel may sit an ulp from it).
    fn doubled_matches(
        &self,
        index: usize,
        swizzle: TextureSwizzleChannels,
        got: [u32; 4],
        exact: bool,
    ) -> bool {
        let want = twice(self.read(index, swizzle));
        (0..4).all(|c| converted_holds(f32::from_bits(got[c]), f32::from_bits(want[c]), exact))
    }

    /// The source as a read-only texture buffer made with `swizzle`, over a new buffer, with
    /// `usage` beside `ShaderRead`.
    fn texture(
        &self,
        device: &Device,
        swizzle: TextureSwizzleChannels,
        usage: TextureUsage,
    ) -> Texture {
        let buffer = buffer_with(device, &self.bytes).unwrap();
        let mut d = TextureDescriptor::texture_buffer(
            self.pixel_format,
            TEXELS as u64,
            ResourceOptions::shared(),
            TextureUsage::SHADER_READ | usage,
        );
        d.swizzle = swizzle;
        buffer.new_texture(&d, 0, self.bytes.len() as u64).unwrap()
    }
}

/// The kernel `name` of public fixture `fixture`'s library on `device`.
fn pipeline(device: &Device, fixture: &str, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{fixture}.ll"))).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// Dispatch `state` over [`GRID`] threads with `textures` at slots 0 and 1 and `buffers` at
/// buffers 0 and up, and wait for it.
fn dispatch(
    device: &Device,
    state: &ComputePipelineState,
    textures: [&Texture; 2],
    buffers: &[&Buffer],
) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for (slot, texture) in textures.into_iter().enumerate() {
        encoder.set_texture(Some(texture), slot as u32).unwrap();
    }
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(GRID as u64, 1, 1), Size::new(GRID as u64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The `float4`s `texel_buffers` writes reading `source` (its texels doubled), into an
/// `RGBA32Float` texture buffer of [`GRID`] texels.
fn doubled(device: &Device, state: &ComputePipelineState, source: &Texture) -> Vec<[u32; 4]> {
    let length = (GRID * 16) as u64;
    let target = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let d = TextureDescriptor::texture_buffer(
        PixelFormat::RGBA32Float,
        GRID as u64,
        ResourceOptions::shared(),
        TextureUsage::SHADER_WRITE,
    );
    let written = target.new_texture(&d, 0, length).unwrap();
    dispatch(device, state, [source, &written], &[]);
    read_buffer(&target, 0, length)
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let w = |i: usize| u32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
            [w(0), w(1), w(2), w(3)]
        })
        .collect()
}

/// `texel`'s float channels doubled, as `texel_buffers` writes them.
fn twice(texel: [u32; 4]) -> [u32; 4] {
    texel.map(|c| (2.0 * f32::from_bits(c)).to_bits())
}

/// Every read of a swizzled texture buffer takes its swizzle, out of bounds too: the float
/// sources through each swizzle at every index of the grid, and the unsigned sources' red
/// channel through a swizzle that takes it from each source (`Zero`, `One`, and each channel) at
/// indices in and past the texture buffer. Every channel is exact, a converted 8-bit normalized
/// one's too, where the device samples through Metal's own texture unit; elsewhere a converted
/// one is the device's conversion of its code, announced `device-texel-conversion`, within its
/// bound (`converted_holds`). Prints the reads checked.
#[test]
fn swizzled_texture_buffers_read_metals_swizzled_texels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let exact = metal_sampler(&device);
        let mut wrong = Vec::new();
        let mut checked = 0usize;
        let floats = pipeline(&device, "texture_ops", "texel_buffers");
        let announced = counts(&messages.take_notices());
        assert_eq!(
            announced.contains_key(&Degradation::DeviceTexelConversion),
            !exact,
            "the float reads' conversion of the codes: {announced:?}"
        );
        for source in float_sources() {
            for (label, swizzle) in swizzles() {
                let texture = source.texture(&device, swizzle, TextureUsage::empty());
                let got = doubled(&device, &floats, &texture);
                for (index, &got) in got.iter().enumerate() {
                    if !source.doubled_matches(index, swizzle, got, exact) {
                        wrong.push(format!(
                            "{:?} {label} at {index}: {got:08x?}, want {:08x?}",
                            source.pixel_format,
                            twice(source.read(index, swizzle))
                        ));
                    }
                    checked += 1;
                }
            }
        }
        // Each thread `t` reads texel `idx[t]` and writes texel `idx[8 + t]` of an unswizzled
        // `R32Uint` texture buffer of `GRID` texels.
        let uints = pipeline(&device, "ob1_texel_buffer", "ob1_texel_buffer");
        let at: [u32; GRID] = [0, 1, 2, 3, 4, 7, 64, 1 << 20];
        let idx: Vec<u8> = at
            .iter()
            .copied()
            .chain(0..GRID as u32)
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let idx = buffer_with(&device, &idx).unwrap();
        use TextureSwizzle::{Alpha, Blue, Green, One, Red, Zero};
        for source in uint_sources() {
            for red in [Zero, One, Red, Green, Blue, Alpha] {
                let swizzle = TextureSwizzleChannels {
                    red,
                    ..TextureSwizzleChannels::DEFAULT
                };
                let texture = source.texture(&device, swizzle, TextureUsage::empty());
                let length = (GRID * 4) as u64;
                let target = device
                    .new_buffer(length, ResourceOptions::shared())
                    .unwrap();
                let d = TextureDescriptor::texture_buffer(
                    PixelFormat::R32Uint,
                    GRID as u64,
                    ResourceOptions::shared(),
                    TextureUsage::SHADER_WRITE,
                );
                let written = target.new_texture(&d, 0, length).unwrap();
                let out = device
                    .new_buffer(length, ResourceOptions::shared())
                    .unwrap();
                dispatch(&device, &uints, [&texture, &written], &[&idx, &out]);
                let got = read_buffer(&out, 0, length).unwrap();
                for (t, &index) in at.iter().enumerate() {
                    let got = u32::from_le_bytes(got[4 * t..4 * t + 4].try_into().unwrap());
                    let want = source.read(index as usize, swizzle)[0];
                    if got != want {
                        wrong.push(format!(
                            "{:?} red from {red:?} at {index}: {got:#x}, want {want:#x}",
                            source.pixel_format
                        ));
                    }
                    checked += 1;
                }
            }
        }
        println!(
            "SWIZZLES {name}: {checked} reads checked, {} differ from Metal",
            wrong.len()
        );
        for line in wrong.iter().take(60) {
            println!("SWIZZLES-DIFFERS {name}: {line}");
        }
        assert!(wrong.is_empty(), "{name}: {} reads differ", wrong.len());
    }
    layer_reported_nothing_at_all(&messages);
}

/// A view of a texture buffer made with a swizzle reads its own swizzle, replacing its
/// texture's, whether the texture was made with one or not, as the probes' views read.
#[test]
fn a_swizzled_view_of_a_texture_buffer_reads_its_own_swizzle() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let floats = pipeline(&device, "texture_ops", "texel_buffers");
        let source = float_sources().remove(0);
        assert_eq!(source.pixel_format, PixelFormat::RGBA8Unorm);
        let [plain, bgr1, zero_one, _, abgr] = swizzles();
        for (parent_label, parent) in [plain, bgr1] {
            let texture = source.texture(&device, parent, TextureUsage::PIXEL_FORMAT_VIEW);
            for (label, swizzle) in [zero_one, abgr] {
                let view = texture
                    .new_texture_view_with_swizzle(
                        PixelFormat::RGBA8Unorm,
                        TextureType::TextureBuffer,
                        0..1,
                        0..1,
                        swizzle,
                    )
                    .unwrap();
                assert_eq!(view.swizzle(), swizzle, "{parent_label} viewed {label}");
                let got = doubled(&device, &floats, &view);
                for (index, &got) in got.iter().enumerate() {
                    assert!(
                        source.doubled_matches(index, swizzle, got, metal_sampler(&device)),
                        "{parent_label} viewed {label} at {index}: {got:08x?}, want {:08x?}",
                        twice(source.read(index, swizzle))
                    );
                }
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}
