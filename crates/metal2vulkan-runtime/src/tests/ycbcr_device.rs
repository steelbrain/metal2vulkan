//! The device's own Y′CbCr conversion of two-plane YCbCr textures (`ycbcr_device.rs`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Each surface is a `Shared` buffer of
//! random codes standing for a CoreVideo surface of its OSType: the luma plane at byte 0 and the
//! interleaved chroma plane after it, rows 64-byte aligned. A kernel reads every texel.
//!
//! Where the device converts the texture's format (`YCbCrSampler::of`), every texel is the
//! conversion Vulkan states for the sampler's chroma location and filter, the format's range
//! expansion and its model's matrix ([`spec_rgb`]), within [`SPEC_BOUND`] (a code-reading
//! format's, a nearest code of it, either at a tie); the texture is
//! counted (`WorkCounts::ycbcr_device_conversions`, `ycbcr_device_samples`) and announced
//! (`ycbcr-filter-not-exact`). Its distance from Metal's reads (`metal2vulkan::bind::ycbcr_read`
//! after Metal's chroma filter) is printed, the note's measured bound. Everywhere else every
//! texel is Metal's read exactly, and nothing is counted or announced.
//!
//! A 4:2:2 texture (`GBGR422`, `BGRG422`) likewise: where the device converts its pairs' format
//! (`YCbCrSampler::pairs`), its decoded twin holds the device's conversion of the pairs, each
//! channel the nearest code to Vulkan's midpoint reconstruction through a linear chroma filter
//! (`3 c + n` over 4 of the texel's pair `c` and the pair beside it `n`), counted
//! (`WorkCounts::chroma_device_twins`, `chroma_device_decodes`) and announced
//! (`decoded-texel-precision`); elsewhere it holds Metal's decode exactly.

use super::blit_holds::{Scratch, WAIT};
use crate::shadow::ShadowSource;
use crate::ycbcr_device::YCbCrSampler;
use crate::{
    Buffer, ComputePipelineState, Degradation, DegradeNotice, Device, DeviceKind, DeviceOptions,
    IoSurfacePlane, IoSurfaceYCbCr, MessageSink, PipelineOptions, Region, ResourceOptions,
    ShaderCacheOptions, Size, SurfaceYCbCrMatrix, Texture, TextureDescriptor,
};
use metal2vulkan::bind::{
    ycbcr_read, ChromaDecode, YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrRange,
};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::format::{ycbcr_format, ChromaSubsampling};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The largest distance of a texel's channel, in units of `1/2048`, from Vulkan's conversion
/// ([`spec_rgb`]; before the sRGB curve, for the formats read through it) the device's may take,
/// past the `half` a converted format's image stores: at most 0.025 measured on lavapipe (Mesa
/// 26.0.8, LLVM 21.1.8), whose 8-bit 4:4:4 layout is the one it converts (0.038 over the
/// validation lanes' surfaces), and 0.034 on NVIDIA's 615.71.09 (RTX 5080 Laptop GPU), which
/// converts the 8-bit 4:2:0, 4:2:2 and 4:4:4 layouts. A code-reading format's channel is a code:
/// the nearest to Vulkan's reconstruction within the bound, so at a tie either (NVIDIA's are
/// Metal's, rounded up).
const SPEC_BOUND: f64 = 0.04;

/// A kernel that reads every texel of texture 0 into buffer 0, `float4`s in rows of the grid.
const READ: &str = r#"; Owned synthetic kernel.
source_filename = "ycbcr_device.metal"

define void @read_texels(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.version = !{!30}
!air.language_version = !{!31}
!0 = !{ptr @read_texels, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!30 = !{i32 2, i32 8, i32 0}
!31 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects, its notices into `heard`.
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
        announce_each: [
            Degradation::YCbCrFilterNotExact,
            Degradation::DecodedTexelPrecision,
        ]
        .into_iter()
        .collect(),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The read kernel's state on `device`.
fn read_state(device: &Device) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(READ).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("read_texels")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// One texture: its format over a surface of `ostype` and `size`, with `matrix`.
#[derive(Clone, Copy, Debug)]
struct Case {
    pf: PixelFormat,
    ostype: &'static [u8; 4],
    size: [u64; 2],
    matrix: Option<SurfaceYCbCrMatrix>,
}

impl Case {
    fn codes(&self) -> YCbCrCodes {
        ycbcr_format(self.pf).and_then(YCbCrCodes::of).unwrap()
    }

    fn range(&self) -> YCbCrRange {
        match self.ostype[3] == b'f' || self.ostype[1] == b'f' {
            true => YCbCrRange::Full,
            false => YCbCrRange::Video,
        }
    }

    fn matrix(&self) -> YCbCrMatrix {
        match self.matrix {
            Some(SurfaceYCbCrMatrix::Itu709) => YCbCrMatrix::Bt709,
            Some(SurfaceYCbCrMatrix::Itu2020) => YCbCrMatrix::Bt2020,
            _ => YCbCrMatrix::Bt601,
        }
    }

    /// The chroma plane's pairs across and rows.
    fn chroma(&self) -> [u64; 2] {
        let halved = self.codes().subsampling.halved();
        [0, 1].map(|axis| match halved[axis] {
            true => self.size[axis].div_ceil(2),
            false => self.size[axis],
        })
    }

    /// The luma plane's row bytes and the chroma plane's, each 64-byte aligned, and where the
    /// chroma plane starts.
    fn layout(&self) -> (u64, u64, u64) {
        let unit = u64::from(self.codes().bytes);
        let luma = (self.size[0] * unit).next_multiple_of(64);
        let chroma = (2 * self.chroma()[0] * unit).next_multiple_of(64);
        (luma, chroma, luma * self.size[1])
    }

    fn length(&self) -> u64 {
        let (_, chroma, at) = self.layout();
        at + chroma * self.chroma()[1]
    }
}

/// The surface's bytes: random codes from `seed`, every unit's every bit random.
fn surface_bytes(case: &Case, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 56) as u8
    };
    (0..case.length()).map(|_| next()).collect()
}

/// Code `i` of the row at byte `at` of `bytes`, of `codes`.
fn code(bytes: &[u8], codes: YCbCrCodes, at: u64, i: u64) -> u16 {
    match codes.bytes {
        1 => u16::from(bytes[(at + i) as usize]),
        _ => {
            let at = (at + 2 * i) as usize;
            u16::from_le_bytes([bytes[at], bytes[at + 1]]) >> (16 - codes.bits)
        }
    }
}

/// The luma code at `(x, y)`, and the chroma pair `(i, j)`'s Cb and Cr.
fn codes_at(
    case: &Case,
    bytes: &[u8],
) -> (impl Fn(u64, u64) -> u16, impl Fn(u64, u64) -> [u16; 2]) {
    let codes = case.codes();
    let (luma_row, chroma_row, chroma_at) = case.layout();
    let luma = {
        let bytes = bytes.to_vec();
        move |x: u64, y: u64| code(&bytes, codes, y * luma_row, x)
    };
    let chroma = {
        let bytes = bytes.to_vec();
        move |i: u64, j: u64| {
            let at = chroma_at + j * chroma_row;
            [
                code(&bytes, codes, at, 2 * i),
                code(&bytes, codes, at, 2 * i + 1),
            ]
        }
    };
    (luma, chroma)
}

/// Metal's read of texel `(x, y)` (`bind::ycbcr_read` after Metal's chroma filter: 9/3/3/1 of
/// 4:2:0, 3/1 of 4:2:2, each sum rounded), as values.
fn metal_rgb(case: &Case, bytes: &[u8], (x, y): (u64, u64)) -> [f64; 3] {
    let codes = case.codes();
    let (luma, chroma) = codes_at(case, bytes);
    let [cw, ch] = case.chroma();
    let halved = codes.subsampling.halved();
    // The taps of one axis: the nearer pair weighted 3, the farther 1, clamped.
    let taps = |v: u64, n: u64, halved: bool| -> Vec<(u64, u64)> {
        match halved {
            false => vec![(v, 4)],
            true => {
                let near = v / 2;
                let far = match v % 2 {
                    1 => (near + 1).min(n - 1),
                    _ => near.saturating_sub(1),
                };
                vec![(near, 3), (far, 1)]
            }
        }
    };
    let across = taps(x, cw, halved[0]);
    let down = taps(y, ch, halved[1]);
    let mut sums = [0u64; 2];
    for &(i, wi) in &across {
        for &(j, wj) in &down {
            let pair = chroma(i, j);
            for c in 0..2 {
                sums[c] += wi * wj * u64::from(pair[c]);
            }
        }
    }
    let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
    let read = ycbcr_read(codes, (case.range(), case.matrix()), [luma(x, y), cb, cr]);
    let grid = match codes.output {
        YCbCrOutput::Srgb => 8192.0,
        YCbCrOutput::Codes => f64::from((1u32 << codes.bits) - 1),
        _ => 2048.0,
    };
    read.map(|v| f64::from(v) / grid)
}

/// The sRGB curve's inverse: the value a channel `l` of a texel read through it was decoded
/// from, which [`spec_rgb`] states.
fn unsrgb(l: f64) -> f64 {
    match l <= 0.04045 / 12.92 {
        true => l * 12.92,
        false => 1.055 * l.powf(1.0 / 2.4) - 0.055,
    }
}

/// Vulkan's conversion of texel `(x, y)` sampled at its centre (implicit reconstruction at the
/// midpoint through a linear chroma filter, the only one the runtime takes of a subsampled
/// format; the range expansion; the model's matrix), as the sample module writes it, unclamped;
/// a code-reading format's clamped into `[0, 1]`, which the sample module rounds to a code; before
/// the sRGB curve, for a format read through it.
fn spec_rgb(case: &Case, bytes: &[u8], (x, y): (u64, u64)) -> [f64; 3] {
    let codes = case.codes();
    let max = f64::from((1u32 << codes.bits) - 1);
    let (luma, chroma) = codes_at(case, bytes);
    let [cw, ch] = case.chroma();
    let halved = codes.subsampling.halved();
    // The taps of one axis at luma texel `v`'s centre, in the chroma plane's texel space.
    let taps = |v: u64, n: u64, halved: bool| -> Vec<(u64, f64)> {
        if !halved {
            return vec![(v, 1.0)];
        }
        let clamp = |i: f64| (i.max(0.0) as u64).min(n - 1);
        let q = (v as f64 + 0.5) / 2.0 - 0.5;
        let i0 = q.floor();
        let f = q - i0;
        vec![(clamp(i0), 1.0 - f), (clamp(i0 + 1.0), f)]
    };
    let across = taps(x, cw, halved[0]);
    let down = taps(y, ch, halved[1]);
    let mut c = [0.0f64; 2];
    for &(i, wi) in &across {
        for &(j, wj) in &down {
            let pair = chroma(i, j);
            for k in 0..2 {
                c[k] += wi * wj * f64::from(pair[k]) / max;
            }
        }
    }
    let y_code = f64::from(luma(x, y));
    if codes.output == YCbCrOutput::Codes {
        return [c[1], y_code / max, c[0]].map(|v| v.clamp(0.0, 1.0));
    }
    let n = codes.bits as i32;
    let scale = f64::from(1u32 << (n - 8));
    let (luma, cb, cr) = match case.range() {
        YCbCrRange::Video => (
            (y_code - 16.0 * scale) / (219.0 * scale),
            (c[0] * max - 128.0 * scale) / (224.0 * scale),
            (c[1] * max - 128.0 * scale) / (224.0 * scale),
        ),
        _ => {
            let mid = f64::from(1u32 << (n - 1)) / max;
            (y_code / max, c[0] - mid, c[1] - mid)
        }
    };
    let (kr, kb) = match case.matrix() {
        YCbCrMatrix::Bt709 => (0.2126, 0.0722),
        YCbCrMatrix::Bt2020 => (0.2627, 0.0593),
        _ => (0.299, 0.114),
    };
    let kg = 1.0 - kr - kb;
    [
        luma + 2.0 * (1.0 - kr) * cr,
        luma - 2.0 * kb * (1.0 - kb) / kg * cb - 2.0 * kr * (1.0 - kr) / kg * cr,
        luma + 2.0 * (1.0 - kb) * cb,
    ]
}

/// A texture of `case` over `surface`'s plane 0.
fn texture(case: &Case, surface: &Buffer) -> Texture {
    let [w, h] = case.size;
    let mut d = TextureDescriptor::texture_2d(case.pf, w, h, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let (luma_row, chroma_row, chroma_at) = case.layout();
    let mut facts = IoSurfaceYCbCr::new(
        u32::from_be_bytes(*case.ostype),
        0,
        case.size,
        chroma_at,
        chroma_row,
    );
    if let Some(matrix) = case.matrix {
        facts = facts.with_matrix(matrix);
    }
    surface
        .new_texture_with_iosurface_plane(
            &d,
            IoSurfacePlane::new(0, luma_row, true).with_ycbcr(facts),
        )
        .unwrap()
}

/// Every texel `state` reads of `texture`, `[r, g, b]` in rows.
fn read(
    device: &Device,
    state: &ComputePipelineState,
    case: &Case,
    texture: &Texture,
) -> Vec<[f32; 3]> {
    let [w, h] = case.size;
    let out = device
        .new_buffer(16 * w * h, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(w, h, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), (16 * w * h) as usize) };
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            [0, 1, 2]
                .map(|c| f32::from_le_bytes([t[4 * c], t[4 * c + 1], t[4 * c + 2], t[4 * c + 3]]))
        })
        .collect()
}

/// Every case: each layout the device may convert, both ranges, three matrices, an even and an
/// odd size, the code-reading and the sRGB-reading formats.
fn cases() -> Vec<Case> {
    let even = [40, 24];
    let odd = [41, 23];
    let mut out = Vec::new();
    for (pf, ostype, size) in [
        (PixelFormat::YCBCR84202P, b"420v", even),
        (PixelFormat::YCBCR84202P, b"420f", odd),
        (PixelFormat::YCBCR84222P, b"422v", even),
        (PixelFormat::YCBCR84442P, b"444f", even),
        (PixelFormat::YCBCR104202P, b"x420", even),
        (PixelFormat::YCBCR104202P, b"xf20", odd),
        (PixelFormat::YCBCR104222P, b"x422", even),
        (PixelFormat::YCBCR104442P, b"x444", even),
        (PixelFormat::RGB84202P, b"420v", even),
        (PixelFormat::RGB104442P, b"x444", even),
        (PixelFormat::YCBCR84202PSRGB, b"420v", even),
        (PixelFormat::YCBCR104202PSRGB, b"xf20", even),
    ] {
        for matrix in [
            None,
            Some(SurfaceYCbCrMatrix::Itu709),
            Some(SurfaceYCbCrMatrix::Itu2020),
        ] {
            out.push(Case {
                pf,
                ostype,
                size,
                matrix,
            });
        }
    }
    out
}

/// Each texture whose format the device converts reads Vulkan's conversion of its planes at
/// every texel, within [`SPEC_BOUND`], counted and announced; every other reads Metal's
/// exactly. A 4:2:0 or 4:2:2 texture takes it only at Metal's midpoint chroma. On MoltenVK, whose
/// two-plane chroma reconstruction is misplaced (`RuntimeDriverDefect::TwoPlaneChromaMisplaced`),
/// no 4:2:0 or 4:2:2 texture takes it; on lavapipe, RADV and NVIDIA, whose conversions read a
/// 10-bit code's unused low bits (`RuntimeDriverDefect::TenBitCodesReadAsWords`), no 10-bit one.
#[test]
fn a_ycbcr_texture_reads_the_devices_conversion_where_it_converts_the_format() {
    let scratch = Scratch::new("ycbcr-device");
    let heard: Heard = Arc::default();
    let device = device(&scratch.0, &heard);
    let state = read_state(&device);
    let defects = &device.properties().driver_defects;
    let misplaced = defects.contains(&RuntimeDriverDefect::TwoPlaneChromaMisplaced);
    let words = defects.contains(&RuntimeDriverDefect::TenBitCodesReadAsWords);
    let mut taken = 0;
    for (n, case) in cases().into_iter().enumerate() {
        let bytes = surface_bytes(&case, 17 + n as u64);
        let surface = device
            .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
            .unwrap();
        heard.lock().unwrap().clear();
        let before = device.work_counts();
        let texture = texture(&case, &surface);
        let texels = read(&device, &state, &case, &texture);
        let after = device.work_counts();
        let copies = texture.shadow().expect("a converted texture's shadow");
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            panic!("a converted texture's source");
        };
        let planes = conversion.source().0;
        let form = YCbCrSampler::of(&device, &planes, false).map(|sampler| sampler.chroma());
        let conversions = after.ycbcr_device_conversions - before.ycbcr_device_conversions;
        let samples = after.ycbcr_device_samples - before.ycbcr_device_samples;
        let announced = heard
            .lock()
            .unwrap()
            .iter()
            .filter(|notice| notice.degradation == Degradation::YCbCrFilterNotExact)
            .count();
        let subsampled = case.codes().subsampling != ChromaSubsampling::Chroma444;
        if (misplaced && subsampled) || (words && case.codes().bits == 10) {
            assert_eq!(form, None, "{case:?}: the defect keeps Metal's arithmetic");
        }
        if let Some((midpoint, linear)) = form {
            assert!(
                (midpoint && linear) || !subsampled,
                "{case:?}: Metal's reconstruction, the midpoint through a linear chroma filter"
            );
        }
        let [w, h] = case.size;
        let (mut spec_far, mut metal_far, mut metal_far_in, mut outside) =
            (0.0f64, 0.0f64, 0.0f64, 0);
        for y in 0..h {
            for x in 0..w {
                let got = texels[(y * w + x) as usize].map(f64::from);
                let metal = metal_rgb(&case, &bytes, (x, y));
                match form {
                    Some(_) => {
                        let spec = spec_rgb(&case, &bytes, (x, y));
                        let decoded = match case.codes().output {
                            YCbCrOutput::Srgb => got.map(unsrgb),
                            _ => got,
                        };
                        let inside = spec.iter().all(|v| (0.0..=1.0).contains(v));
                        outside += usize::from(!inside);
                        for c in 0..3 {
                            // A converted format's image holds `half`s: the device's value lies
                            // within one unit in the last place of the stored one
                            // (`PrecisionNote::NativeHalfConversion`). A code-reading format's
                            // holds codes: Vulkan's value lies within half a code of the nearest.
                            let ulp = match case.codes().output {
                                YCbCrOutput::Converted => 2f64.powi(
                                    got[c].abs().max(2f64.powi(-14)).log2().floor() as i32 - 10,
                                ),
                                YCbCrOutput::Codes => {
                                    0.5 / f64::from((1u32 << case.codes().bits) - 1)
                                }
                                _ => 0.0,
                            };
                            let far = ((decoded[c] - spec[c]).abs() - ulp).max(0.0);
                            spec_far = spec_far.max(far * 2048.0);
                            let far = (got[c] - metal[c]).abs() * 2048.0;
                            metal_far = metal_far.max(far);
                            if inside {
                                metal_far_in = metal_far_in.max(far);
                            }
                        }
                    }
                    None => {
                        let (got, metal) = (got.map(|v| v as f32), metal.map(|v| v as f32));
                        // A code's value is the device's conversion of the texture's format's
                        // normalized code, within one unit in the last place
                        // (`PrecisionNote::DeviceTexelConversion`); every other output exact.
                        let ulps = match case.codes().output {
                            YCbCrOutput::Codes => 1,
                            _ => 0,
                        };
                        assert!(
                            (0..3).all(|c| got[c].to_bits().abs_diff(metal[c].to_bits()) <= ulps),
                            "{case:?} texel ({x}, {y}): {got:?}, Metal's read {metal:?}"
                        );
                    }
                }
            }
        }
        match form {
            Some(form) => {
                taken += 1;
                println!(
                    "{:?} {} {:?} {w}x{h} midpoint/linear {form:?}: from Vulkan's {spec_far:.4}/2048, \
                     from Metal's {metal_far:.2}/2048 ({metal_far_in:.4} in [0, 1]), {outside} \
                     texels outside [0, 1]",
                    case.pf,
                    String::from_utf8_lossy(case.ostype),
                    case.matrix,
                );
                assert!(
                    spec_far <= SPEC_BOUND,
                    "{case:?}: {spec_far} from Vulkan's conversion"
                );
                assert_eq!((conversions, samples), (1, 1), "{case:?}: counted");
                assert_eq!(announced, 1, "{case:?}: announced");
            }
            None => {
                assert_eq!(
                    (conversions, samples),
                    (0, 0),
                    "{case:?}: Metal's arithmetic"
                );
                assert_eq!(announced, 0, "{case:?}: exact, unannounced");
            }
        }
    }
    println!("{taken} textures took the device's conversion");
}

/// The largest distance of a 4:2:2 twin's chroma channel, in codes, from Vulkan's
/// reconstruction `(3 c + n) / 4` the device's conversion may take: half a code (the twin holds
/// the nearest code) and a quarter for the device's precision.
const PAIRS_BOUND: f64 = 0.75;

/// Every texel `state` reads of a 4:2:2 `texture` of `w x h`, `[r, g, b]` in rows.
fn read_pairs(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    (w, h): (u64, u64),
) -> Vec<[f32; 3]> {
    let case = Case {
        pf: PixelFormat::YCBCR84222P,
        ostype: b"422v",
        size: [w, h],
        matrix: None,
    };
    read(device, state, &case, texture)
}

/// Each 4:2:2 texture whose format the device converts holds the device's conversion of its pairs
/// in its twin at every texel: its green its own code, its red and blue within [`PAIRS_BOUND`] of
/// Vulkan's midpoint reconstruction, counted and announced; every other reads Metal's decode
/// exactly, unannounced. The texture is written on the CPU, whole and in part of two rows.
#[test]
fn a_four_two_two_twin_holds_the_devices_conversion_where_it_converts_the_format() {
    let scratch = Scratch::new("ycbcr-device-pairs");
    let heard: Heard = Arc::default();
    let device = device(&scratch.0, &heard);
    let state = read_state(&device);
    let (w, h) = (40u64, 24u64);
    let mut taken = 0;
    for (k, pf) in [PixelFormat::GBGR422, PixelFormat::BGRG422]
        .into_iter()
        .enumerate()
    {
        let decode = ChromaDecode::of(pf).unwrap();
        let form = YCbCrSampler::pairs(&device, decode, [w as u32, h as u32]);
        heard.lock().unwrap().clear();
        let before = device.work_counts();
        let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = device.new_texture(&descriptor).unwrap();
        let mut random = 0x422_u64 + k as u64;
        let mut bytes: Vec<u8> = (0..2 * w * h)
            .map(|_| {
                random = random
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (random >> 56) as u8
            })
            .collect();
        texture
            .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &bytes, 2 * w, 0)
            .unwrap();
        // 6 texels from x 3 on rows 5 and 6: their pairs' red and blue change beside them.
        let patch: Vec<u8> = (0..24u32).map(|i| (i * 37 + 11) as u8).collect();
        texture
            .replace_region(Region::new_2d(3, 5, 6, 2), 0, 0, &patch, 12, 0)
            .unwrap();
        for row in 0..2u64 {
            let at = ((5 + row) * 2 * w + 2 * 3) as usize;
            bytes[at..at + 12].copy_from_slice(&patch[12 * row as usize..][..12]);
        }
        let texels = read_pairs(&device, &state, &texture, (w, h));
        let after = device.work_counts();
        let metal = decode.decode(&bytes, [w as u32, h as u32, 1]);
        let pair =
            |x: u64, y: u64, byte: u64| f64::from(bytes[(y * 2 * w + (x / 2) * 4 + byte) as usize]);
        let (red, blue) = match decode {
            ChromaDecode::Gbgr => (3, 1),
            _ => (2, 0),
        };
        let (mut far, mut off_metal) = (0.0f64, 0);
        for y in 0..h {
            for x in 0..w {
                let got = texels[(y * w + x) as usize].map(|v| f64::from(v) * 255.0);
                let want = &metal[((y * w + x) * 4) as usize..][..4];
                match form {
                    Some(_) => {
                        assert!(
                            (got[1] - f64::from(want[1])).abs() < 1e-3,
                            "{pf:?} texel ({x}, {y}): green {} against its code {}",
                            got[1],
                            want[1]
                        );
                        // The pair beside: before an even texel, after an odd one, its own at
                        // the row's ends.
                        let beside = match x % 2 {
                            0 => x.saturating_sub(2),
                            _ => (x + 2).min(w - 1),
                        };
                        for (c, byte) in [(0, red), (2, blue)] {
                            let spec = (3.0 * pair(x, y, byte) + pair(beside, y, byte)) / 4.0;
                            far = far.max((got[c] - spec).abs());
                            off_metal += usize::from((got[c] - f64::from(want[c])).abs() > 1e-3);
                        }
                    }
                    None => {
                        for c in 0..3 {
                            assert!(
                                (got[c] - f64::from(want[c])).abs() < 1e-3,
                                "{pf:?} texel ({x}, {y}) channel {c}: {} against Metal's {}",
                                got[c],
                                want[c]
                            );
                        }
                    }
                }
            }
        }
        let twins = after.chroma_device_twins - before.chroma_device_twins;
        let decodes = after.chroma_device_decodes - before.chroma_device_decodes;
        let announced = heard
            .lock()
            .unwrap()
            .iter()
            .filter(|notice| notice.degradation == Degradation::DecodedTexelPrecision)
            .count();
        match form {
            Some(_) => {
                taken += 1;
                println!(
                    "{pf:?} {w}x{h}: chroma within {far:.4} codes of Vulkan's; {off_metal} of {} \
                     chroma channels a code off Metal's",
                    2 * w * h
                );
                assert!(far <= PAIRS_BOUND, "{pf:?}: {far} codes from Vulkan's");
                assert_eq!(twins, 1, "{pf:?}: counted");
                assert!(decodes >= 2, "{pf:?}: each write a pass, {decodes}");
                assert_eq!(
                    after.cpu_block_decodes, before.cpu_block_decodes,
                    "{pf:?}: no CPU decode"
                );
                assert_eq!(announced, 1, "{pf:?}: announced");
            }
            None => {
                assert_eq!((twins, decodes), (0, 0), "{pf:?}: Metal's decode");
                assert_eq!(announced, 0, "{pf:?}: exact, unannounced");
            }
        }
    }
    println!("{taken} 4:2:2 textures took the device's conversion");
}
