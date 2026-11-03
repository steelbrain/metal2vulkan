//! Render targets of Apple's extended-range formats on the device the cfg selects, under the
//! validation layer: what a draw and a clear store, against Metal's measured texels.
//!
//! Metal's answers: our probes (an Apple M3 Max on macOS 26.7 and an Apple M4 on
//! macOS 27.0.1, byte-identical, plain and under Metal's validation layer alike), whose probe was
//! written for this repository:
//!
//! - **Draws.** A 128 x 64 target of each format, one full-screen draw of the owned fixture
//!   `rt6_samples`'s fills, `float4` (`rt6_fill_rgba32float`) and `half4`
//!   (`rt6_fill_rgba16float`), of 8,192 outputs from the probe's generator (rebuilt here; its
//!   inputs' digests are checked first): values across and past each format's range, code
//!   boundaries a few `f32` steps either side, NaN, the infinities and zeros. Metal's texels: the
//!   SHA-256 of the target's bytes (`getBytes`), and for the two-plane formats of plane 0's words
//!   and of the alpha bytes. The model (`ExtendedRange::clear_texel`: the output rounded to
//!   `half`, then a kernel write's code) gives the same digests, checked here too. The device's
//!   bytes are compared with the model's texel by texel; a two-plane target, whose image holds
//!   both planes side by side and is not copied, is compared through kernel reads with a texture
//!   the fixture `xr_two_planes` writes from the `half` values (a kernel write is Metal's,
//!   `extended_range_two_planes`).
//! - **Clears.** 484 clear colors per format (Metal's specials, every sixteenth code's value, and
//!   code and alpha-byte boundaries with their `f64` and `f32` neighbours), each the color's four
//!   channels, cleared into a 1 x 1 target through a `Clear` load action: Metal's texel bytes
//!   (but the two-plane formats') and its `float` reads (`xr2p_read_float`), as digests over the
//!   list; the model's bytes give the same digest.
//!
//! - **Clear routes**. A `Clear` of a `BGR10_XR` or `BGRA10_XR` target and their `_sRGB` twins
//!   stores the texel of Metal's route for its texture and pass (`ClearRoute`, measured by our
//!   probes, whose model the product's `clears_store_metal_s_texels_on_each_route` checks):
//!   straight from the `float` for a 16 x 16 private target of plain usage in a pass with no draw
//!   call, and through `half` for an 8 x 8 one, one of `ShaderWrite` usage, and one whose pass
//!   draws into another attachment.
//!
//! A difference is printed on `XR-TARGETS-DIFFERS` lines, the first texels of each output with
//! their inputs, the stored texel and the model's.

use super::resolves::{lane, quiet, Lane};
use super::xr_codes::{
    assert_announced, codes, metal_read, nearest_code, read_within, texel_within,
};
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_samples::{library, shape};
use metal2vulkan::metal::format::{ClearRoute, ExtendedRange};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandBufferStatus, ComputePipelineState, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use sha2::{Digest, Sha256};

/// The draws' target: 128 x 64 texels, one output each.
pub(super) const WIDTH: usize = 128;
pub(super) const HEIGHT: usize = 64;
pub(super) const COUNT: usize = WIDTH * HEIGHT;

/// The SHA-256 of the probe's `float4` and `half4` outputs (`rt.src.f`, `rt.src.h`).
const SOURCE_FLOAT: &str = "aa40c5c5e485af466c71b3a8877a9be51f2a8e3005a19fc950c104feede45238";
pub(super) const SOURCE_HALF: &str =
    "e1e257f0dca558bae1d82053b821eea9a78ad59af1379462600590815d0f366a";

/// One format and Metal's digests: the draws' target bytes (plane 0's words for a two-plane
/// format), the alpha bytes of a two-plane one, the clears' bytes (none for a two-plane one, which
/// Metal copies a plane of), and the clears' `float` reads.
struct Format {
    pf: PixelFormat,
    form: ExtendedRange,
    drawn: &'static str,
    alpha: Option<&'static str>,
    cleared: Option<&'static str>,
    cleared_reads: &'static str,
}

const FORMATS: [Format; 6] = [
    Format {
        pf: PixelFormat::BGRA10XR,
        form: ExtendedRange::Bgra10 { srgb: false },
        drawn: "986030cab9c1f7923198bf430d0795ad790ceb0dccb9d19ea65c1335e67f251e",
        alpha: None,
        cleared: Some("595f84a2bee0b90bd9cb389158597b06bc4ffc6c1fcfb2569cbbf7c5bda29b27"),
        cleared_reads: "8d0d429aed5e9af06fc5859247760db1ce1144c064285fa36e50e417390fae88",
    },
    Format {
        pf: PixelFormat::BGRA10XRSRGB,
        form: ExtendedRange::Bgra10 { srgb: true },
        drawn: "9a9782967835f0e6e7f20de236df35874853520d9930671f4c518cb08cb0c566",
        alpha: None,
        cleared: Some("ce43cb3f3da2e6e21b051f7136f4eb70e1790ca464218c93cfcb09e2d6f50de1"),
        cleared_reads: "1be003396edef52c0437a7aa83cdff12018a2cf0d3e69028f264aa31dc35291c",
    },
    Format {
        pf: PixelFormat::BGR10XR,
        form: ExtendedRange::Bgr10 { srgb: false },
        drawn: "97716aae7c825382cf06d8fedc2276f223ed71c652c1da2f18669efaeeba3d51",
        alpha: None,
        cleared: Some("e2d9b0aae4b8e2d86b5c4a7bfeabec307fa8182c0e21ab928c7a3b4b12840dfe"),
        cleared_reads: "4bdd52191959806c079f0d43f6c081b75babe4d36c25063bea2c14b80e070f02",
    },
    Format {
        pf: PixelFormat::BGR10XRSRGB,
        form: ExtendedRange::Bgr10 { srgb: true },
        drawn: "80f7d569272f4df4f80f1f7710f03e2c86fd42f1e938c945d7aa84fa5e9b0bbf",
        alpha: None,
        cleared: Some("24c1b2ed9442318d565e07c0a88b7d7b550d9d80c6191b025ee05af4ab58b4ed"),
        cleared_reads: "9567ee343aaa05bea4c15d0f24d4add69b4014c5f7217e5db5cc6c6a0a9bd26e",
    },
    Format {
        pf: PixelFormat::RGB10A82PXR10,
        form: ExtendedRange::Planes { srgb: false },
        drawn: "42669033462c97584fa60c6f0e1495600df3a4f19f10d4251922413f5aee7f23",
        alpha: Some("da175edfc5cbcb3d7428f49d8441ac42d4237a208f1b5b13fcde98f0b3b1384a"),
        cleared: None,
        cleared_reads: "7ac4c7ca5bf6bb551b467e08cb268a161492024529b29650a541432408055b3f",
    },
    Format {
        pf: PixelFormat::RGB10A82PXR10SRGB,
        form: ExtendedRange::Planes { srgb: true },
        drawn: "4f567cfd3684f195a0a50aabca00083623c0f4587d7f29d1fe2f1dc3012e014d",
        alpha: Some("da175edfc5cbcb3d7428f49d8441ac42d4237a208f1b5b13fcde98f0b3b1384a"),
        cleared: None,
        cleared_reads: "7bc0a7cd0e9845cdb769aadfc189695738bf791392dc1b4555622362fefbcf2f",
    },
];

// ---- The probe's generator ---------------------------------------------------------------------

/// The probe's xorshift64* generator.
struct Rng(u64);

impl Rng {
    fn seeded(seed: u64) -> Rng {
        let state = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x632B_E59B_D9B4_E019);
        let mut r = Rng(if state == 0 { 1 } else { state });
        for _ in 0..8 {
            r.next();
        }
        r
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(2_685_821_657_736_338_717)
    }

    /// A uniform `f64` in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// A uniform integer below `n`.
    fn below(&mut self, n: u32) -> i32 {
        (self.next() % u64::from(n)) as i32
    }
}

/// `v` moved `k` `f32` steps (up for a positive `k`).
fn step(mut v: f32, k: i32) -> f32 {
    for _ in 0..k.max(0) {
        v = v.next_up();
    }
    for _ in 0..(-k).max(0) {
        v = v.next_down();
    }
    v
}

/// A color channel's output.
fn color(r: &mut Rng) -> f32 {
    let k = r.below(1000);
    if k < 650 {
        return (-0.8 + 2.1 * r.unit()) as f32;
    }
    if k < 750 {
        let c = r.below(1024) - 384;
        let s = r.below(5) - 2;
        return step((f64::from(c) / 510.0) as f32, s);
    }
    if k < 820 {
        let c = r.below(1023);
        let s = r.below(5) - 2;
        return step(((f64::from(c) + 0.5 - 384.0) / 510.0) as f32, s);
    }
    if k < 900 {
        let s = r.below(22552) - 6168;
        let h = if r.below(2) != 0 { 0.5 } else { 0.0 };
        let t = r.below(3) - 1;
        return step(((f64::from(s) + h) / 8192.0) as f32, t);
    }
    if k < 995 {
        return (-3.0 + 6.0 * r.unit()) as f32;
    }
    match k {
        995 => f32::NAN,
        996 => f32::INFINITY,
        997 => f32::NEG_INFINITY,
        998 => -0.0,
        _ => 0.0,
    }
}

/// An alpha channel's output.
fn alpha(r: &mut Rng) -> f32 {
    let k = r.below(1000);
    if k < 650 {
        return r.unit() as f32;
    }
    if k < 700 {
        return 0.0;
    }
    if k < 750 {
        return 1.0;
    }
    if k < 800 {
        let b = r.below(256);
        let t = r.below(3) - 1;
        return step((f64::from(b) / 255.0) as f32, t);
    }
    if k < 850 {
        let b = r.below(255);
        let t = r.below(3) - 1;
        return step(((f64::from(b) + 0.5) / 255.0) as f32, t);
    }
    if k < 900 {
        let c = r.below(511);
        let t = r.below(3) - 1;
        return step((f64::from(c) / 510.0) as f32, t);
    }
    if k < 995 {
        return (-0.5 + 2.0 * r.unit()) as f32;
    }
    match k {
        995 => f32::NAN,
        996 => f32::INFINITY,
        997 => -0.0,
        _ => 2.0,
    }
}

/// The draws' outputs: the probe's seed 77, red, green, blue, alpha per texel.
pub(super) fn outputs() -> Vec<[f32; 4]> {
    let mut r = Rng::seeded(77);
    (0..COUNT)
        .map(|_| {
            let rgb = [color(&mut r), color(&mut r), color(&mut r)];
            [rgb[0], rgb[1], rgb[2], alpha(&mut r)]
        })
        .collect()
}

/// `x` rounded to `half` to nearest, ties to even, as the probe's `(_Float16)` conversion.
pub(super) fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return sign | 0x7e00 | ((a >> 13) & 0x1ff) as u16;
    }
    if a < 0x3880_0000 {
        // Below 2^-14: a `half` denormal, in units of 2^-24.
        if a < 0x3300_0000 {
            return sign;
        }
        let m = (a & 0x7f_ffff) | 0x80_0000;
        let shift = 126 - (a >> 23);
        let (q, r, half) = (m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1));
        let q = q + u32::from(r > half || (r == half && q & 1 == 1));
        return sign | q as u16;
    }
    let mut h = (a >> 13) - ((127 - 15) << 10);
    let r = a & 0x1fff;
    if r > 0x1000 || (r == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    if h >= 0x7c00 {
        return sign | 0x7c00;
    }
    sign | h as u16
}

/// The `f32` value of `half` bits `h`.
pub(super) fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (u32::from((h >> 10) & 0x1f), u32::from(h & 0x3ff));
    match e {
        0 => sign * m as f32 * 2f32.powi(-24),
        31 if m == 0 => sign * f32::INFINITY,
        31 => f32::from_bits(u32::from(h & 0x8000) << 16 | 0x7f80_0000 | m << 13),
        _ => sign * (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    }
}

// ---- The model and the comparisons ---------------------------------------------------------------

/// The bytes of raw texel `texel` of `form` as Metal's texture holds plane 0, and a two-plane
/// texel's alpha byte.
fn metal_bytes(form: ExtendedRange, texel: [u32; 4]) -> (Vec<u8>, Option<u8>) {
    match form {
        ExtendedRange::Bgra10 { .. } => (
            texel
                .iter()
                .flat_map(|&f| (f as u16).to_le_bytes())
                .collect(),
            None,
        ),
        ExtendedRange::Bgr10 { .. } => (texel[0].to_le_bytes().to_vec(), None),
        _ => (texel[0].to_le_bytes().to_vec(), Some(texel[1] as u8)),
    }
}

/// The bytes of a texel of `form`'s raw format.
fn texel_size(form: ExtendedRange) -> usize {
    match form {
        ExtendedRange::Bgra10 { .. } => 8,
        _ => 4,
    }
}

pub(super) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The fixture `xr_two_planes`'s kernel `name`.
pub(super) fn kernel(lane: &Lane, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("xr_two_planes.ll")).unwrap();
    let library = lane.device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    lane.device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Runs the kernel `state` over the `width` x `height` texels of `texture` with `buffer` at 0
/// (from `offset`) and the row width at 1.
pub(super) fn dispatch(
    lane: &Lane,
    state: &ComputePipelineState,
    texture: &Texture,
    (buffer, offset): (&metal2vulkan_runtime::Buffer, u64),
    (width, height): (u64, u64),
) {
    let w = buffer_with(&lane.device, &(width as u32).to_le_bytes()).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(buffer), offset, 0).unwrap();
    encoder.set_buffer(Some(&w), 0, 1).unwrap();
    encoder
        .dispatch_threads(
            Size::new(width, height, 1),
            Size::new(width.min(32), height.min(32), 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// The `float4` reads of every texel of a `width` x `height` `texture`.
pub(super) fn reads(
    lane: &Lane,
    read: &ComputePipelineState,
    texture: &Texture,
    size: (u64, u64),
) -> Vec<u8> {
    let length = size.0 * size.1 * 16;
    let out = buffer_with(&lane.device, &vec![0xab; length as usize]).unwrap();
    dispatch(lane, read, texture, (&out, 0), size);
    read_buffer(&out, 0, length).unwrap()
}

/// A texture of `pf`, `width` x `height`, of `usage`.
fn texture(
    lane: &Lane,
    pf: PixelFormat,
    (width, height): (u64, u64),
    usage: TextureUsage,
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::private();
    lane.device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {pf:?} texture: {e}"))
}

/// The target `f`'s draw of `values` (`half4`s where `half`, else `float4`s) stored.
fn draw(lane: &Lane, f: &Format, values: &[u8], half: bool) -> Texture {
    let library = library(&lane.device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut pipeline = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
    pipeline.fragment = Some(function(if half {
        "rt6_fill_rgba16float"
    } else {
        "rt6_fill_rgba32float"
    }));
    pipeline.color_attachments[0] = Some(ColorAttachmentDescriptor::new(f.pf));
    let pipeline = lane
        .device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
        .unwrap_or_else(|e| panic!("{:?}: the pipeline: {e}", f.pf));
    let values = buffer_with(&lane.device, values).unwrap();
    let shape = shape(&lane.device, WIDTH as u32, 1).unwrap();
    let target = texture(
        lane,
        f.pf,
        (WIDTH as u64, HEIGHT as u64),
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    );
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    lane.run(&pass, |_, e, _| {
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap_or_else(|e| panic!("{:?}: the pass: {e}", f.pf));
    target
}

/// The first differing texels of `got` and `want` (`size` bytes each), with `inputs`' lanes.
fn differences(got: &[u8], want: &[u8], size: usize, inputs: &[[f32; 4]]) -> Vec<String> {
    got.chunks(size)
        .zip(want.chunks(size))
        .enumerate()
        .filter(|(_, (g, w))| g != w)
        .take(6)
        .map(|(i, (g, w))| {
            let bits = inputs.get(i).map(|v| v.map(f32::to_bits));
            format!("texel {i} of {bits:08x?}: {g:02x?}, Metal's {w:02x?}")
        })
        .collect()
}

/// Each format's draws of the probe's outputs, `float` and `half`, store Metal's texels: each code
/// within one of Metal's and the bits Metal does not read Metal's, a two-plane target's codes
/// (through its kernel reads, each the code whose Metal read is nearest it) within one of Metal's
/// and each read within its bound of Metal's read of that code, where the conversion is the
/// device's (`PrecisionNote::DevicePackConversion`, announced).
#[test]
fn draws_store_metals_extended_range_texels() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let inputs = outputs();
    let floats: Vec<u8> = inputs
        .iter()
        .flat_map(|v| v.iter().flat_map(|x| x.to_le_bytes()))
        .collect();
    let halves: Vec<u16> = inputs.iter().flat_map(|v| v.map(half_bits)).collect();
    let half_bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    assert_eq!(digest(&floats), SOURCE_FLOAT, "the probe's float outputs");
    assert_eq!(digest(&half_bytes), SOURCE_HALF, "the probe's half outputs");
    let read = kernel(&lane, "xr2p_read_float");
    let (mut wrong, mut moved) = (Vec::new(), 0);
    for f in &FORMATS {
        // The model's texels, of the `half` each output rounds to.
        let (mut plane, mut bytes) = (Vec::new(), Vec::new());
        for v in halves.chunks(4) {
            let color = [0, 1, 2, 3].map(|c| f64::from(half_value(v[c])));
            let (p, a) = metal_bytes(f.form, f.form.clear_texel(color, ClearRoute::ThroughHalf));
            plane.extend(p);
            bytes.extend(a);
        }
        assert_eq!(digest(&plane), f.drawn, "{:?}: the model's texels", f.pf);
        if let Some(alpha) = f.alpha {
            assert_eq!(digest(&bytes), alpha, "{:?}: the model's alpha bytes", f.pf);
        }
        for half in [false, true] {
            let what = format!("{:?} from {}", f.pf, if half { "half" } else { "float" });
            let values = if half { &half_bytes } else { &floats };
            let target = draw(&lane, f, values, half);
            let size = texel_size(f.form);
            if f.alpha.is_none() {
                let got = lane.read(&target, (0, 0), size as u64, BlitOption::None);
                let mut differ = Vec::new();
                for (i, (g, w)) in got.chunks(size).zip(plane.chunks(size)).enumerate() {
                    match (g == w, texel_within(f.form, g, w)) {
                        (true, _) => {}
                        (false, true) => moved += 1,
                        (false, false) => differ.push(i),
                    }
                }
                for &i in differ.iter().take(6) {
                    let at = i * size..(i + 1) * size;
                    let bits = inputs[i].map(f32::to_bits);
                    wrong.push(format!(
                        "{what}: texel {i} of {bits:08x?}: {:02x?}, Metal's {:02x?}",
                        &got[at.clone()],
                        &plane[at]
                    ));
                }
                continue;
            }
            // A two-plane target through its kernel reads: the codes they came from, and the reads.
            let got = reads(&lane, &read, &target, (WIDTH as u64, HEIGHT as u64));
            let mut differ = 0;
            for (i, texel) in got.as_chunks::<16>().0.iter().enumerate() {
                let (want, _) = codes(f.form, &plane[i * size..(i + 1) * size]);
                let want = [want[0], want[1], want[2], u32::from(bytes[i])];
                let lanes = texel
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b));
                let (mut same, mut held) = (true, true);
                for (c, (v, code)) in lanes.zip(want).enumerate() {
                    let alpha = c == 3;
                    let read_code = nearest_code(f.form, v, alpha);
                    same &= read_code == code && v == metal_read(f.form, code, alpha);
                    held &=
                        read_code.abs_diff(code) <= 1 && read_within(f.form, v, read_code, alpha);
                }
                match (same, held) {
                    (true, _) => {}
                    (false, true) => moved += 1,
                    (false, false) => {
                        differ += 1;
                        if differ <= 6 {
                            let bits = inputs[i].map(f32::to_bits);
                            wrong.push(format!(
                                "{what} (reads): texel {i} of {bits:08x?}: {:02x?}, Metal's codes \
                                 {want:?}",
                                &texel[..]
                            ));
                        }
                    }
                }
            }
        }
    }
    for line in &wrong {
        println!("XR-TARGETS-DIFFERS {name}: {line}");
    }
    println!("XR-TARGETS {name}: {moved} drawn texels within the device's conversion of Metal's");
    assert!(
        wrong.is_empty(),
        "{} texels differ from Metal's",
        wrong.len()
    );
    assert_announced(&messages, moved, "the draws");
    quiet(lane, &messages);
}

/// The clear colors: Metal's specials, every sixteenth code's value, and code and alpha-byte
/// boundaries with their `f64` and `f32` neighbours (a subset of the probe's, in this order).
fn clear_values() -> Vec<f64> {
    let mut out = vec![
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0,
        -2.0,
        1e-40,
        3.0,
        -1.0,
        1.0,
        0.5,
        1.252_941_176_470_588_2,
        1.3,
        1e30,
        -0.752_941_176_470_588_2,
        0.1,
        0.2,
        0.25,
        0.75,
    ];
    out.extend((0..1024).step_by(16).map(|c| f64::from(c - 384) / 510.0));
    let boundaries = (8..1023)
        .step_by(16)
        .map(|c| (f64::from(c) + 0.5 - 384.0) / 510.0)
        .chain((4..255).step_by(16).map(|b| (f64::from(b) + 0.5) / 255.0));
    for t in boundaries {
        let tf = t as f32;
        out.extend([
            t,
            t.next_down(),
            t.next_up(),
            f64::from(tf.next_down()),
            f64::from(tf.next_up()),
        ]);
    }
    out
}

/// Each format's `Clear` load action stores Metal's texel for every clear color, which reads
/// Metal's values.
#[test]
fn clears_store_metals_extended_range_texels() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let values = clear_values();
    assert_eq!(values.len(), 484, "the clear colors");
    let read = kernel(&lane, "xr2p_read_float");
    let (mut wrong, mut moved) = (Vec::new(), 0);
    for f in &FORMATS {
        let size = texel_size(f.form) as u64;
        let n = values.len() as u64;
        let bytes = buffer_with(&lane.device, &vec![0xab; (size * n) as usize]).unwrap();
        let floats = buffer_with(&lane.device, &vec![0xab; (16 * n) as usize]).unwrap();
        let w = buffer_with(&lane.device, &1u32.to_le_bytes()).unwrap();
        let target = texture(
            &lane,
            f.pf,
            (1, 1),
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        );
        for batch in (0..n).step_by(64) {
            let command_buffer = lane.queue.command_buffer().unwrap();
            for i in batch..(batch + 64).min(n) {
                let v = values[i as usize];
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(target.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(v, v, v, v);
                command_buffer
                    .render_command_encoder(&pass)
                    .unwrap()
                    .end_encoding()
                    .unwrap();
                if f.cleared.is_some() {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.copy_from_texture_to_buffer(
                        &target,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        Size::new(1, 1, 1),
                        &bytes,
                        size * i,
                        size,
                        size,
                        BlitOption::None,
                    )
                    .unwrap();
                    blit.end_encoding().unwrap();
                }
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encoder.set_compute_pipeline_state(&read).unwrap();
                encoder.set_texture(Some(&target), 0).unwrap();
                encoder.set_buffer(Some(&floats), 16 * i, 0).unwrap();
                encoder.set_buffer(Some(&w), 0, 1).unwrap();
                encoder
                    .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
                    .unwrap();
                encoder.end_encoding().unwrap();
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        }
        if let Some(cleared) = f.cleared {
            let model: Vec<u8> = values
                .iter()
                .flat_map(|&v| {
                    metal_bytes(f.form, f.form.clear_texel([v; 4], ClearRoute::ThroughHalf)).0
                })
                .collect();
            assert_eq!(digest(&model), cleared, "{:?}: the model's clears", f.pf);
            let got = read_buffer(&bytes, 0, size * n).unwrap();
            if got != model {
                let colors: Vec<[f32; 4]> = values.iter().map(|&v| [v as f32; 4]).collect();
                for line in differences(&got, &model, size as usize, &colors) {
                    wrong.push(format!("{:?} clears: {line}", f.pf));
                }
            }
        }
        // Metal's reads of the clears' codes, which give Metal's digest; the device's are its
        // conversion of the same codes.
        let (codes, model) = cleared_reads(f.form, &values);
        assert_eq!(
            digest(&model),
            f.cleared_reads,
            "{:?}: the model's clears' reads",
            f.pf
        );
        let got = read_buffer(&floats, 0, 16 * n).unwrap();
        let lanes = got
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b));
        let mut differ = Vec::new();
        for (i, (v, want)) in lanes.zip(model.as_chunks::<4>().0).enumerate() {
            let (texel, c) = (i / 4, i % 4);
            let held = match codes[texel][c] {
                Some(code) => read_within(f.form, v, code, c == 3),
                None => v == 1.0,
            };
            match (v.to_bits() == u32::from_le_bytes(*want), held) {
                (true, _) => {}
                (false, true) => moved += 1,
                (false, false) => differ.push(i),
            }
        }
        for &i in differ.iter().take(6) {
            let want = f32::from_le_bytes(model.as_chunks::<4>().0[i]);
            let got = f32::from_le_bytes(got.as_chunks::<4>().0[i]);
            wrong.push(format!(
                "{:?} clear {} of {}: channel {} reads {got:e}, Metal's {want:e}",
                f.pf,
                i / 4,
                values[i / 4],
                i % 4
            ));
        }
    }
    for line in &wrong {
        println!("XR-TARGETS-DIFFERS {name}: {line}");
    }
    println!("XR-TARGETS {name}: {moved} clears' reads within the device's conversion of Metal's");
    assert!(
        wrong.is_empty(),
        "{} clears differ from Metal's",
        wrong.len()
    );
    assert_announced(&messages, moved, "the clears' reads");
    quiet(lane, &messages);
}

/// The codes of each clear of `values` into a texel of `form` (Metal's, `clear_texel`'s), and the
/// bytes of Metal's `float4` reads of them: its read of each code ([`metal_read`]), `BGR10_XR`'s
/// alpha 1.
fn cleared_reads(form: ExtendedRange, values: &[f64]) -> (Vec<[Option<u32>; 4]>, Vec<u8>) {
    let codes: Vec<[Option<u32>; 4]> = values
        .iter()
        .map(|&v| rgba_codes(form, form.clear_texel([v; 4], ClearRoute::ThroughHalf)))
        .collect();
    let reads = codes
        .iter()
        .flat_map(|texel| {
            (0..4).map(|c| texel[c].map_or(1.0, |code| metal_read(form, code, c == 3)))
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    (codes, reads)
}

/// The models give Metal's digests: each format's draws of the probe's `half` outputs, a two-plane
/// format's alpha bytes, its clears' bytes and its `float` reads of them.
#[test]
fn the_models_give_metal_s_digests() {
    let halves: Vec<u16> = outputs().iter().flat_map(|v| v.map(half_bits)).collect();
    let values = clear_values();
    for f in &FORMATS {
        let (mut plane, mut bytes) = (Vec::new(), Vec::new());
        for v in halves.chunks(4) {
            let color = [0, 1, 2, 3].map(|c| f64::from(half_value(v[c])));
            let (p, a) = metal_bytes(f.form, f.form.clear_texel(color, ClearRoute::ThroughHalf));
            plane.extend(p);
            bytes.extend(a);
        }
        assert_eq!(digest(&plane), f.drawn, "{:?}: the drawn texels", f.pf);
        if let Some(alpha) = f.alpha {
            assert_eq!(digest(&bytes), alpha, "{:?}: the drawn alpha bytes", f.pf);
        }
        if let Some(cleared) = f.cleared {
            let model: Vec<u8> = values
                .iter()
                .flat_map(|&v| {
                    metal_bytes(f.form, f.form.clear_texel([v; 4], ClearRoute::ThroughHalf)).0
                })
                .collect();
            assert_eq!(digest(&model), cleared, "{:?}: the clears' bytes", f.pf);
        }
        let (_, reads) = cleared_reads(f.form, &values);
        assert_eq!(
            digest(&reads),
            f.cleared_reads,
            "{:?}: the clears' reads",
            f.pf
        );
    }
}

/// The red, green, blue and alpha codes of the raw texel `raw` of `form` (`clear_texel`'s); none
/// for `BGR10_XR`'s alpha, which reads 1.
fn rgba_codes(form: ExtendedRange, raw: [u32; 4]) -> [Option<u32>; 4] {
    match form {
        ExtendedRange::Bgra10 { .. } => [2, 1, 0, 3].map(|f| Some(raw[f] >> 6)),
        _ => {
            let w = raw[0];
            let alpha = matches!(form, ExtendedRange::Planes { .. }).then_some(raw[1]);
            [
                Some(w >> 20 & 0x3ff),
                Some(w >> 10 & 0x3ff),
                Some(w & 0x3ff),
                alpha,
            ]
        }
    }
}

/// Colors whose texels differ between Metal's two clear routes: ties between codes, which the
/// straight route takes up (`0.25` there `float` steps below a tie, down), NaN (code 0 straight,
/// 384 through `half`; alpha 384 on both), and `_sRGB` values each side of an encode threshold
/// whose route rounds them apart (`0.0254486893...` straight below its threshold, where an exact
/// sRGB curve puts it above).
const ROUTE_COLORS: [[f64; 4]; 4] = [
    [0.25, -0.25, f64::NAN, 0.75],
    [
        0.000_980_392_156_862_745,
        0.500_980_392_156_862_7,
        0.999,
        f64::NAN,
    ],
    [
        0.035_294_117_647_058_82,
        -0.522_549_019_607_843_2,
        0.025_448_689_342_018_27,
        0.25,
    ],
    [
        -0.343_137_254_901_960_8,
        1.175_490_196_078_431_4,
        f64::INFINITY,
        -1.0,
    ],
];

/// A `Clear` of each single-plane extended-range format stores Metal's route's texel in every
/// texel of the target (the module docs).
#[test]
fn clears_take_metal_s_route() {
    let (messages, _cache, lane) = lane();
    let plain = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let library = library(&lane.device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut wrong = Vec::new();
    for f in FORMATS.iter().filter(|f| f.alpha.is_none()) {
        let mut state = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
        state.fragment = Some(function("rt6_fill_rgba32float"));
        state.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        state.color_attachments[1] = Some(ColorAttachmentDescriptor::new(f.pf));
        let state = lane
            .device
            .new_render_pipeline_state(&state, &PipelineOptions::default())
            .unwrap_or_else(|e| panic!("{:?}: the pipeline: {e}", f.pf));
        let size = texel_size(f.form);
        for (side, usage, draws, route) in [
            (16, plain, false, ClearRoute::Straight),
            (8, plain, false, ClearRoute::ThroughHalf),
            (
                16,
                plain | TextureUsage::SHADER_WRITE,
                false,
                ClearRoute::ThroughHalf,
            ),
            (16, plain, true, ClearRoute::ThroughHalf),
        ] {
            let case = format!("{:?} {side}x{side} {usage:?} draws {draws}", f.pf);
            let values: Vec<u8> = (0..side * side)
                .flat_map(|i| [(i % 7) as f32 / 7.0, 0.25, 0.5, 1.0])
                .flat_map(f32::to_le_bytes)
                .collect();
            let values = buffer_with(&lane.device, &values).unwrap();
            let fill_shape = shape(&lane.device, side as u32, 1).unwrap();
            for (i, &c) in ROUTE_COLORS.iter().enumerate() {
                let target = texture(&lane, f.pf, (side, side), usage);
                let drawn = texture(&lane, PixelFormat::RGBA32Float, (side, side), plain);
                let mut descriptor = RenderPassDescriptor::new();
                let other = &mut descriptor.color_attachments[0];
                other.texture = Some(drawn);
                other.load_action = LoadAction::DontCare;
                other.store_action = StoreAction::Store;
                let color = &mut descriptor.color_attachments[1];
                color.texture = Some(target.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(c[0], c[1], c[2], c[3]);
                lane.run(&descriptor, |_, e, _| {
                    if draws {
                        e.set_render_pipeline_state(&state).unwrap();
                        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
                        e.set_fragment_buffer(Some(&fill_shape), 0, 1).unwrap();
                        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                            .unwrap();
                    }
                })
                .unwrap_or_else(|e| panic!("{case}: the pass: {e}"));
                let got = lane.read(&target, (0, 0), size as u64, BlitOption::None);
                let (model, _) = metal_bytes(f.form, f.form.clear_texel(c, route));
                if let Some((at, t)) = got.chunks(size).enumerate().find(|(_, t)| *t != model) {
                    wrong.push(format!(
                        "{case} color {i} {c:?}: texel {at} stored {t:02x?}, Metal's {model:02x?}"
                    ));
                }
            }
        }
    }
    for line in &wrong {
        println!("XR-TARGETS-DIFFERS {}: {line}", lane.device.name());
    }
    assert!(wrong.is_empty(), "{} clears differ", wrong.len());
    quiet(lane, &messages);
}
