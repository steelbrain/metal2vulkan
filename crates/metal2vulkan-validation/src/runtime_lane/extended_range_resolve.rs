//! Multisample resolves of the extended-range formats Metal makes multisampled textures of
//! (`BGR10_XR`, `BGRA10_XR`, their `_sRGB` twins and `RGB10A2Unorm_sRGB`), on the device the cfg
//! selects, under the validation layer, over the owned fixture `xr_resolve`.
//!
//! Each pass draws an 8 x 4 target at 2 or 4 samples, each sample its own color: our probe `xr`'s
//! values (its generator, seed 7, its first 32 pixels), whose resolves Metal wrote on an Apple M3
//! Max under macOS 27.0.1 ([`METAL`]). A linear form's color resolves by Metal's rule, the
//! average of the samples' codes rounded half up, so each such channel is Metal's code;
//! `BGRA10_XR`'s alpha, an `_sRGB` twin's color and `RGB10A2Unorm_sRGB`'s channels average values
//! the codes do not keep, so the runtime writes a near-equivalent, announced as
//! `extended-range-resolve`, each such channel here within one code of Metal's (the probe's 32
//! pixels; `bind::ColorResolveFormat::extended_range` states the bounds over its 4096). A second
//! pass draws every sample of a pixel alike, whose resolve is the code a single-sampled target of
//! the same draw stores, as Metal's resolve of samples of one value is. A sample count the device
//! lacks (2 is optional in Vulkan: lavapipe offers 1 and 4), refused as `Unsupported` with
//! `Feature::RasterSampleCount`, is skipped with a printed line, and the other counts are checked.

use super::degrade_notices::{assert_once, forced, shared_in_system_memory};
use super::{buffer_with, read_buffer, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandQueue, Degradation, Device, Error, Library, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The kinds this lane's resolves announce, for `degrade_notices`' census.
pub(super) const XR_RESOLVE: [Degradation; 1] = [Degradation::ExtendedRangeResolve];

/// The target's width and height.
const WIDTH: u64 = 8;
const HEIGHT: u64 = 4;
/// Its pixels.
const PIXELS: usize = (WIDTH * HEIGHT) as usize;

/// Metal's resolved codes of each pixel (red, green, blue and alpha, 16 bits each from the top;
/// `BGR10_XR`'s alpha 0), by format and sample count: probe `xr` on an Apple M3 Max under macOS
/// 27.0.1.
#[rustfmt::skip]
const METAL: [(PixelFormat, u32, [u64; PIXELS]); 10] = [
    (PixelFormat::BGRA10XR, 2, [
        0x03ce_00c1_02a0_0278, 0x00b8_002c_0040_021d, 0x0343_0327_00df_0336, 0x01bf_017e_02a3_02f2,
        0x0117_01c3_002a_0308, 0x03bd_01be_0197_0280, 0x0141_011a_002b_0180, 0x03d1_02f8_025f_0360,
        0x00a0_02cd_00d5_01c6, 0x020d_0162_0150_0180, 0x01ba_016e_021a_019c, 0x0056_00ec_0362_0180,
        0x0280_01ac_01f5_025c, 0x0301_01e0_019f_0252, 0x0391_022c_0158_0257, 0x0241_02ad_0000_0180,
        0x030e_022d_01d8_0180, 0x0009_032e_0113_030f, 0x03fc_004a_0252_0291, 0x029d_0386_033d_037e,
        0x0144_038f_0033_0318, 0x0146_0134_0279_01c4, 0x028c_01ad_0129_0247, 0x024a_02fa_005c_02df,
        0x01f4_01c8_02ee_01cc, 0x002d_01ee_01a9_01e8, 0x0066_02e3_005d_02fa, 0x0088_0342_026e_0180,
        0x01e3_0270_0114_0219, 0x0008_0183_016c_0250, 0x0102_004f_01b5_037e, 0x006d_026e_0122_01eb,
    ]),
    (PixelFormat::BGRA10XR, 4, [
        0x026b_026e_0199_0281, 0x03ec_033f_00e0_0227, 0x004b_02ac_01a5_0180, 0x03bd_0174_00e5_01e6,
        0x0284_02a6_02ed_01d6, 0x01ed_036f_03d7_0205, 0x01b9_01d7_025f_01c7, 0x03b9_0204_036b_0294,
        0x01ab_033f_01b1_0229, 0x0241_03ff_0000_0180, 0x00ff_03c2_0113_01d8, 0x0348_021b_0081_02be,
        0x01e7_018b_01ff_0286, 0x007a_008f_0128_032a, 0x0108_028f_0002_037e, 0x002c_00b5_0024_0316,
        0x01a9_02af_0198_0257, 0x0382_02b8_03f5_0180, 0x0102_02c1_00e3_02c9, 0x02a5_027d_0179_01a7,
        0x01e1_018a_00f9_0278, 0x00f1_0201_0196_0212, 0x00af_03d4_03ff_02f4, 0x0001_019b_01fd_037e,
        0x017b_0265_0164_025b, 0x017d_00ed_031d_037e, 0x0252_016c_0381_02ac, 0x0376_0007_02a4_0368,
        0x0299_0272_018a_0231, 0x00d5_0296_03c1_0308, 0x0138_012f_00f4_01a7, 0x00a5_009e_032f_01fa,
    ]),
    (PixelFormat::BGRA10XRSRGB, 2, [
        0x039f_0037_030d_0278, 0x002f_0000_0000_021d, 0x0363_0356_004f_0336, 0x0245_0167_030e_02f2,
        0x0085_024b_0000_0308, 0x0399_0244_01f8_0280, 0x00bb_0088_0000_0180, 0x03a1_033d_02e1_0360,
        0x001e_0327_007c_01c6, 0x029f_00f7_00d2_0180, 0x023c_0115_02aa_019c, 0x0000_005a_0371_0180,
        0x02f7_0226_0287_025c, 0x0343_0270_020b_0252, 0x0386_02ba_00e1_0257, 0x02cb_0314_0000_0180,
        0x0349_02ba_0267_0180, 0x0000_0359_0081_030f, 0x03b2_0000_02d8_0291, 0x030a_0381_0360_037e,
        0x00be_0385_0000_0318, 0x00c3_00a9_02f3_01c4, 0x02ff_0227_009a_0247, 0x02d2_033e_0000_02df,
        0x02bb_0252_0339_01cc, 0x0000_027f_021f_01e8, 0x0000_0333_0000_02fa, 0x000d_0363_02eb_0180,
        0x0274_02ed_0082_0219, 0x0000_019f_0111_0250, 0x006f_0000_0235_037e, 0x0000_02eb_0092_01eb,
    ]),
    (PixelFormat::BGRA10XRSRGB, 4, [
        0x02ea_02ec_01fc_0281, 0x03ac_0361_0050_0227, 0x0000_0313_0217_0180, 0x0399_012d_0055_01e6,
        0x0307_0310_0338_01d6, 0x027f_0377_03a3_0205, 0x023c_0265_02e1_01c7, 0x0397_0296_0375_0294,
        0x022e_0361_0238_0229, 0x02cb_03ba_0000_0180, 0x006d_039b_0081_01d8, 0x0365_02ac_0009_02be,
        0x0278_0225_0291_0286, 0x0004_0012_009a_032a, 0x0076_0302_0000_037e, 0x0000_002d_0000_0316,
        0x0220_0315_0236_0257, 0x0380_031a_03af_0180, 0x006f_0320_0053_02c9, 0x030f_02f5_0141_01a7,
        0x0271_01cc_0082_0278, 0x0060_0293_01f5_0212, 0x0029_03a2_03b6_02f4, 0x0000_0202_028f_037e,
        0x01d7_02e5_0130_025b, 0x015d_005c_0350_037e, 0x02d8_0110_037f_02ac, 0x037a_0000_030e_0368,
        0x030b_02ee_021c_0231, 0x0047_0305_039a_0308, 0x00ad_00a1_0062_01a7, 0x0022_001d_035a_01fa,
    ]),
    (PixelFormat::BGR10XR, 2, [
        0x03ce_00c1_02a0_0000, 0x00b8_002c_0040_0000, 0x0343_0327_00df_0000, 0x01bf_017e_02a3_0000,
        0x0117_01c3_002a_0000, 0x03bd_01be_0197_0000, 0x0141_011a_002b_0000, 0x03d1_02f8_025f_0000,
        0x00a0_02cd_00d5_0000, 0x020d_0162_0150_0000, 0x01ba_016e_021a_0000, 0x0056_00ec_0362_0000,
        0x0280_01ac_01f5_0000, 0x0301_01e0_019f_0000, 0x0391_022c_0158_0000, 0x0241_02ad_0000_0000,
        0x030e_022d_01d8_0000, 0x0009_032e_0113_0000, 0x03fc_004a_0252_0000, 0x029d_0386_033d_0000,
        0x0144_038f_0033_0000, 0x0146_0134_0279_0000, 0x028c_01ad_0129_0000, 0x024a_02fa_005c_0000,
        0x01f4_01c8_02ee_0000, 0x002d_01ee_01a9_0000, 0x0066_02e3_005d_0000, 0x0088_0342_026e_0000,
        0x01e3_0270_0114_0000, 0x0008_0183_016c_0000, 0x0102_004f_01b5_0000, 0x006d_026e_0122_0000,
    ]),
    (PixelFormat::BGR10XR, 4, [
        0x026b_026e_0199_0000, 0x03ec_033f_00e0_0000, 0x004b_02ac_01a5_0000, 0x03bd_0174_00e5_0000,
        0x0284_02a6_02ed_0000, 0x01ed_036f_03d7_0000, 0x01b9_01d7_025f_0000, 0x03b9_0204_036b_0000,
        0x01ab_033f_01b1_0000, 0x0241_03ff_0000_0000, 0x00ff_03c2_0113_0000, 0x0348_021b_0081_0000,
        0x01e7_018b_01ff_0000, 0x007a_008f_0128_0000, 0x0108_028f_0002_0000, 0x002c_00b5_0024_0000,
        0x01a9_02af_0198_0000, 0x0382_02b8_03f5_0000, 0x0102_02c1_00e3_0000, 0x02a5_027d_0179_0000,
        0x01e1_018a_00f9_0000, 0x00f1_0201_0196_0000, 0x00af_03d4_03ff_0000, 0x0001_019b_01fd_0000,
        0x017b_0265_0164_0000, 0x017d_00ed_031d_0000, 0x0252_016c_0381_0000, 0x0376_0007_02a4_0000,
        0x0299_0272_018a_0000, 0x00d5_0296_03c1_0000, 0x0138_012f_00f4_0000, 0x00a5_009e_032f_0000,
    ]),
    (PixelFormat::BGR10XRSRGB, 2, [
        0x039f_0037_030d_0000, 0x002f_0000_0000_0000, 0x0363_0356_004f_0000, 0x0245_0167_030e_0000,
        0x0085_024b_0000_0000, 0x0399_0244_01f8_0000, 0x00bb_0088_0000_0000, 0x03a1_033d_02e1_0000,
        0x001e_0327_007c_0000, 0x029f_00f7_00d2_0000, 0x023c_0115_02aa_0000, 0x0000_005a_0371_0000,
        0x02f7_0226_0287_0000, 0x0343_0270_020b_0000, 0x0386_02ba_00e1_0000, 0x02cb_0314_0000_0000,
        0x0349_02ba_0267_0000, 0x0000_0359_0081_0000, 0x03b2_0000_02d8_0000, 0x030a_0381_0360_0000,
        0x00be_0385_0000_0000, 0x00c3_00a9_02f3_0000, 0x02ff_0227_009a_0000, 0x02d2_033e_0000_0000,
        0x02bb_0252_0339_0000, 0x0000_027f_021f_0000, 0x0000_0333_0000_0000, 0x000d_0363_02eb_0000,
        0x0274_02ed_0082_0000, 0x0000_019f_0111_0000, 0x006f_0000_0235_0000, 0x0000_02eb_0092_0000,
    ]),
    (PixelFormat::BGR10XRSRGB, 4, [
        0x02ea_02ec_01fc_0000, 0x03ac_0361_0050_0000, 0x0000_0313_0217_0000, 0x0399_012d_0055_0000,
        0x0307_0310_0338_0000, 0x027f_0377_03a3_0000, 0x023c_0265_02e1_0000, 0x0397_0296_0375_0000,
        0x022e_0361_0238_0000, 0x02cb_03ba_0000_0000, 0x006d_039b_0081_0000, 0x0365_02ac_0009_0000,
        0x0278_0225_0291_0000, 0x0004_0012_009a_0000, 0x0076_0302_0000_0000, 0x0000_002d_0000_0000,
        0x0220_0315_0236_0000, 0x0380_031a_03af_0000, 0x006f_0320_0053_0000, 0x030f_02f5_0141_0000,
        0x0271_01cc_0082_0000, 0x0060_0293_01f5_0000, 0x0029_03a2_03b6_0000, 0x0000_0202_028f_0000,
        0x01d7_02e5_0130_0000, 0x015d_005c_0350_0000, 0x02d8_0110_037f_0000, 0x037a_0000_030e_0000,
        0x030b_02ee_021c_0000, 0x0047_0305_039a_0000, 0x00ad_00a1_0062_0000, 0x0022_001d_035a_0000,
    ]),
    (PixelFormat::RGB10A2UnormSRGB, 2, [
        0x03fc_01a8_034a_0001, 0x0198_0000_0000_0001, 0x03be_03af_01d4_0002, 0x02bd_0284_0361_0002,
        0x021a_02c0_0000_0002, 0x03fe_02bc_029b_0002, 0x0248_0220_0000_0000, 0x03ff_0393_0335_0003,
        0x0170_037b_01f1_0000, 0x02fb_026a_0258_0000, 0x02b8_0275_0303_0000, 0x00aa_01e4_03cf_0000,
        0x034a_02ad_02e8_0001, 0x0399_02d8_02a2_0001, 0x03e8_0310_0261_0001, 0x031f_0366_0000_0000,
        0x0396_0311_02d2_0000, 0x0000_03b3_0215_0002, 0x03ff_0065_032b_0002, 0x035c_03e2_03bb_0003,
        0x024b_03e2_0087_0002, 0x024e_023c_0345_0001, 0x0351_02ae_022f_0001, 0x0325_0394_00c0_0002,
        0x02f0_02c5_037d_0000, 0x0000_02e2_02ab_0001, 0x00e3_0388_00c5_0002, 0x0140_03bd_033d_0000,
        0x02da_033f_0217_0001, 0x0000_028a_0274_0001, 0x0201_0087_02b5_0003, 0x00f8_033d_0227_0001,
    ]),
    (PixelFormat::RGB10A2UnormSRGB, 4, [
        0x0335_0333_029c_0002, 0x03ff_03bc_01d6_0001, 0x006b_0366_02a8_0000, 0x03fe_027b_01dc_0001,
        0x034e_0362_038d_0001, 0x02e1_03d6_03ff_0001, 0x02b7_02d0_0335_0001, 0x03fc_02f4_03d4_0002,
        0x02ad_03bb_02b2_0001, 0x031f_03ff_0000_0000, 0x01fd_03ff_0217_0001, 0x03c1_0305_012e_0002,
        0x02da_0298_02ef_0002, 0x011c_014d_022f_0002, 0x020a_0354_0000_0003, 0x0000_0194_0000_0002,
        0x02ab_0368_02ab_0001, 0x03e0_036d_03ff_0000, 0x0200_0374_01d9_0002, 0x0361_0348_027f_0000,
        0x02d8_0283_0208_0001, 0x01ee_02f1_029b_0001, 0x018a_03ff_03ff_0002, 0x0000_029f_02ed_0003,
        0x0286_0338_0271_0001, 0x0283_01e7_03a9_0003, 0x032c_0274_03df_0002, 0x03d9_0000_0360_0003,
        0x0350_0341_029e_0001, 0x01c5_0358_03ff_0002, 0x023f_0234_01f1_0000, 0x0177_016c_03b4_0001,
    ]),
];

/// The probe's generator: a 64-bit LCG, each draw its top 53 bits over 2^53.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / 9007199254740992.0
    }
}

/// The probe's per-sample colors (red, green, blue, alpha) of its first [`PIXELS`] pixels at
/// `samples` for `pf`, each pixel's samples in turn, in the probe's draw order: each channel
/// anything in its range, or by the pixel's kind near one code (centers, off center, or half
/// codes).
fn values(pf: PixelFormat, samples: u32) -> Vec<[f32; 4]> {
    let mut rng = Lcg(7);
    let n = samples as usize;
    let mut out = vec![[0.0f32; 4]; PIXELS * n];
    let padded = pf == PixelFormat::RGB10A2UnormSRGB;
    for px in 0..PIXELS {
        let kind = px & 3;
        for ch in 0..4 {
            let (lo, hi) = match (padded, ch) {
                (true, _) => (-0.1, 1.1),
                (false, 3) => (-0.2, 1.2),
                (false, _) => (-0.8, 1.3),
            };
            let base = lo + (hi - lo) * rng.next();
            let code = (base * 510.0 + 384.5) as i32;
            for sample in &mut out[px * n..(px + 1) * n] {
                let x = match kind {
                    0 => lo + (hi - lo) * rng.next(),
                    1 => f64::from(code + (rng.next() * 3.0) as i32 - 1 - 384) / 510.0,
                    2 => (f64::from(code) + (rng.next() - 0.5) * 3.0 - 384.0) / 510.0,
                    _ => {
                        let halves = f64::from((rng.next() * 4.0) as i32);
                        (f64::from(code - 384) + 0.5 * halves - 1.0) / 510.0
                    }
                };
                sample[ch] = x as f32;
            }
        }
    }
    out
}

/// The bytes of a texel of `pf`.
fn texel_bytes(pf: PixelFormat) -> u64 {
    match pf {
        PixelFormat::BGRA10XR | PixelFormat::BGRA10XRSRGB => 8,
        _ => 4,
    }
}

/// The codes (red, green, blue, alpha; `BGR10_XR`'s alpha 0) of a raw texel of `pf`.
fn codes(pf: PixelFormat, texel: &[u8]) -> [u32; 4] {
    match pf {
        PixelFormat::BGRA10XR | PixelFormat::BGRA10XRSRGB => {
            let (fields, _) = texel.as_chunks::<2>();
            let field = |i: usize| u32::from(u16::from_le_bytes(fields[i])) >> 6;
            [field(2), field(1), field(0), field(3)]
        }
        PixelFormat::BGR10XR | PixelFormat::BGR10XRSRGB => {
            let w = u32::from_le_bytes(texel.try_into().unwrap());
            [(w >> 20) & 1023, (w >> 10) & 1023, w & 1023, 0]
        }
        _ => {
            let w = u32::from_le_bytes(texel.try_into().unwrap());
            [w & 1023, (w >> 10) & 1023, (w >> 20) & 1023, w >> 30]
        }
    }
}

/// The channels `pf` has.
fn channels(pf: PixelFormat) -> usize {
    match pf {
        PixelFormat::BGR10XR | PixelFormat::BGR10XRSRGB => 3,
        _ => 4,
    }
}

/// Whether `pf`'s channel `ch` resolves by Metal's rule (a linear form's color), not by the
/// near-equivalent.
fn by_rule(pf: PixelFormat, ch: usize) -> bool {
    matches!(pf, PixelFormat::BGRA10XR | PixelFormat::BGR10XR) && ch < 3
}

/// Metal's codes of `word`, red first.
fn unpacked(word: u64) -> [u32; 4] {
    [48, 32, 16, 0].map(|shift| ((word >> shift) & 0xffff) as u32)
}

struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(messages: &Messages, cache: &ScratchCache) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |_| {}).unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("xr_resolve.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        }
    }

    fn function(&self, name: &str) -> StageSource {
        StageSource::Function(
            self.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    }

    /// A private render target of `pf`, `samples` deep.
    fn texture(&self, pf: PixelFormat, samples: u32) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, WIDTH, HEIGHT, false);
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = u64::from(samples);
        }
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        self.device.new_texture(&d).unwrap()
    }

    /// Each pixel's codes after a pass drawing `colors` (each pixel's samples in turn) into a
    /// target of `pf` at `samples`: stored and resolved into a single-sampled texture past one
    /// sample, else stored; `None`, with a printed line, where the device refuses the count.
    fn draw(&self, pf: PixelFormat, samples: u32, colors: &[[f32; 4]]) -> Option<Vec<[u32; 4]>> {
        let mut d = RenderPipelineDescriptor::new(self.function("xrr_vertex"));
        d.fragment = Some(self.function("xrr_samples"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
        d.raster_sample_count = samples;
        let pipeline = match self
            .device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
        {
            Ok(pipeline) => pipeline,
            Err(Error::Unsupported(u)) if u.feature == Feature::RasterSampleCount => {
                println!("{}: {pf:?} at {samples} refused: {u:?}", self.device.name());
                return None;
            }
            Err(e) => panic!("{pf:?} at {samples}: {e}"),
        };
        let target = self.texture(pf, samples);
        let resolved = (samples > 1).then(|| self.texture(pf, 1));
        let bytes: Vec<u8> = colors
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let values = buffer_with(&self.device, &bytes).unwrap();
        let shape = [WIDTH as u32, samples].map(u32::to_le_bytes).concat();
        let shape = buffer_with(&self.device, &shape).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(target.clone());
        a.load_action = LoadAction::Clear;
        a.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        a.store_action = match &resolved {
            Some(resolved) => {
                a.resolve_texture = Some(resolved.clone());
                StoreAction::StoreAndMultisampleResolve
            }
            None => StoreAction::Store,
        };
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        e.end_encoding().unwrap();
        let texel = texel_bytes(pf);
        let length = texel * WIDTH * HEIGHT;
        let out = buffer_with(&self.device, &vec![0; length as usize]).unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            resolved.as_ref().unwrap_or(&target),
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, HEIGHT, 1),
            &out,
            0,
            texel * WIDTH,
            length,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{pf:?} at {samples}");
        let texels = read_buffer(&out, 0, length)
            .unwrap()
            .chunks(texel as usize)
            .map(|t| codes(pf, t))
            .collect();
        Some(texels)
    }
}

/// Each form's resolve of the probe's samples: a linear form's colors Metal's codes, the rest
/// within one code of Metal's; and of samples drawn alike, the code a single-sampled target of
/// the same draw stores. The near-equivalent is announced once, as is the extra pass.
#[test]
fn extended_range_resolves_are_metals_or_within_a_code() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::new(&messages, &cache);
    let name = lane.device.name().to_owned();
    let mut wrong = Vec::new();
    let mut drawn = 0;
    for &(pf, samples, metal) in &METAL {
        let colors = values(pf, samples);
        let Some(got) = lane.draw(pf, samples, &colors) else {
            continue;
        };
        drawn += 1;
        let (mut same, mut near) = (0, 0);
        for (px, (got, &metal)) in got.iter().zip(&metal).enumerate() {
            let metal = unpacked(metal);
            for ch in 0..channels(pf) {
                match (got[ch].abs_diff(metal[ch]), by_rule(pf, ch)) {
                    (0, _) => same += 1,
                    (1, false) => near += 1,
                    _ => wrong.push(format!(
                        "{pf:?} at {samples}, pixel {px} channel {ch}: {} against Metal's {}",
                        got[ch], metal[ch]
                    )),
                }
            }
        }
        println!("{name}: {pf:?} at {samples}: {same} channels Metal's, {near} one code off");
        // Every sample of a pixel alike: the code a single-sampled draw of that color stores.
        let firsts: Vec<[f32; 4]> = colors.iter().step_by(samples as usize).copied().collect();
        let alike: Vec<[f32; 4]> = firsts
            .iter()
            .flat_map(|&c| std::iter::repeat_n(c, samples as usize))
            .collect();
        let stored = lane.draw(pf, 1, &firsts).expect("one sample");
        let resolved = lane
            .draw(pf, samples, &alike)
            .expect("the count drawn above");
        if resolved != stored {
            wrong.push(format!(
                "{pf:?} at {samples}, samples alike: resolved {resolved:x?} against stored \
                 {stored:x?}"
            ));
        }
    }
    assert!(drawn > 0, "{name}: no sample count drawn");
    assert!(
        wrong.is_empty(),
        "{name}: {} channels:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    // The `Shared` textures and buffers the draws read and write announce their memory on a
    // device without unified memory.
    let mut kinds = vec![XR_RESOLVE[0], Degradation::ShaderResolve];
    kinds.extend(forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&lane.device),
    )]));
    assert_once(&messages.take_notices(), &kinds);
    drop(lane);
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
