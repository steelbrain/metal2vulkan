//! Device tests of a two-plane YCbCr texture's planes read in place
//! (`metal2vulkan::bind::YCbCrInPlace`, the runtime's `texture/ycbcr_image.rs`), on the device the
//! cfg selects, under the validation layer.
//!
//! A device whose memory is the host's (an integrated GPU or a CPU device: `hasUnifiedMemory`)
//! reads the planes of a `YCBCR8_420_2P` or `YCBCR10_420_2P` texture in place: the texture counts
//! `WorkCounts::ycbcr_textures_in_place` when it is made, and a dispatch whose kernel only samples
//! it binds a uniform texel buffer over the planes' bytes (`ycbcr_planes_bound`) and converts
//! nothing (`ycbcr_refreshes`, `ycbcr_whole_conversions`). Any other device keeps the texture's
//! image alone (`ycbcr_images_kept_device`) and converts it through the refresh. On either, a
//! nearest sample at a texel's centre reads the texel: Metal's, or where the device converts the
//! texture's planes itself (`ycbcr_device_reads`), the device's conversion of them. Where the planes
//! are read in place a linear sample at a texel's centre reads Metal's texel and one between
//! texels Metal's filter of them; a texture that keeps its image is filtered by the device's own
//! filter, which this module does not sample. A `read()` takes the image on every device.
//!
//! Each surface is a buffer standing for the IOSurface: the luma plane at byte 0, rows padded past
//! their codes, the interleaved chroma plane after it; a 10-bit code in the high 10 bits of a
//! little-endian word whose low 6 bits are noise, which Metal does not read.

use super::degrade_notices::{counts, on_device, on_device_with};
use super::ycbcr_device_reads::{assert_device_reads, device_conversions, vulkan_read};
use super::{buffer_with, read_buffer, ycbcr_textures};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{
    ycbcr_read, SamplerDescriptor, YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrRange,
};
use metal2vulkan::device::DevicePolicy;
use metal2vulkan::metal::format::ycbcr_format;
use metal2vulkan::metal::{PixelFormat, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, ComputePipelineState, Degradation, Device, Error, IoSurfacePlane,
    IoSurfaceYCbCr, PipelineOptions, ResourceOptions, SamplerState, Size, SurfaceYCbCrMatrix,
    Texture, TextureDescriptor, WorkCounts,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// A two-plane 4:2:0 surface of one format: its OSType and matrix attachment, the conversion Metal
/// reads them as, its size, the bytes between both planes' rows, and the chroma plane's offset.
#[derive(Clone, Copy, Debug)]
pub(super) struct Surface {
    pub(super) format: PixelFormat,
    pub(super) ostype: [u8; 4],
    pub(super) matrix: Option<SurfaceYCbCrMatrix>,
    pub(super) conversion: (YCbCrRange, YCbCrMatrix),
    pub(super) size: [u64; 2],
    pub(super) row: u64,
    pub(super) chroma_at: u64,
}

/// The lane's surfaces: 40 x 24, 8-bit `420v` without a matrix (BT.601) and `420f` of BT.709 with
/// rows of 64 bytes, 10-bit `x420` of BT.2020 and `xf20` without a matrix with rows of 96.
const SURFACES: [Surface; 4] = [
    Surface {
        format: PixelFormat::YCBCR84202P,
        ostype: *b"420v",
        matrix: None,
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        size: [40, 24],
        row: 64,
        chroma_at: 64 * 24,
    },
    Surface {
        format: PixelFormat::YCBCR84202P,
        ostype: *b"420f",
        matrix: Some(SurfaceYCbCrMatrix::Itu709),
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt709),
        size: [40, 24],
        row: 64,
        chroma_at: 64 * 24,
    },
    Surface {
        format: PixelFormat::YCBCR104202P,
        ostype: *b"x420",
        matrix: Some(SurfaceYCbCrMatrix::Itu2020),
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        size: [40, 24],
        row: 96,
        chroma_at: 96 * 24,
    },
    Surface {
        format: PixelFormat::YCBCR104202P,
        ostype: *b"xf20",
        matrix: None,
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt601),
        size: [40, 24],
        row: 96,
        chroma_at: 96 * 24,
    },
];

/// Surfaces whose rows lie 256 bytes apart, the pitch a driver's linear image of their width
/// takes (`texture/ycbcr_planes.rs` takes the image only where its planes lie as the surface's):
/// 8-bit `420v` and 10-bit `x420`, 40 x 24.
const PLANES_SURFACES: [Surface; 2] = [
    Surface {
        format: PixelFormat::YCBCR84202P,
        ostype: *b"420v",
        matrix: None,
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        size: [40, 24],
        row: 256,
        chroma_at: 256 * 24,
    },
    Surface {
        format: PixelFormat::YCBCR104202P,
        ostype: *b"x420",
        matrix: Some(SurfaceYCbCrMatrix::Itu2020),
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        size: [40, 24],
        row: 256,
        chroma_at: 256 * 24,
    },
];

impl Surface {
    /// The format's codes.
    fn codes(&self) -> YCbCrCodes {
        ycbcr_format(self.format)
            .and_then(YCbCrCodes::of)
            .expect("a converted two-plane format")
    }

    /// The bytes of a code: 1 or 2.
    fn unit(&self) -> u64 {
        u64::from(self.codes().bytes)
    }

    /// The chroma plane's pairs across and rows.
    fn chroma_size(&self) -> [u64; 2] {
        self.size.map(|n| n.div_ceil(2))
    }

    /// The surface's bytes: through the chroma plane's last row.
    pub(super) fn length(&self) -> u64 {
        self.chroma_at + self.row * self.chroma_size()[1]
    }

    /// The surface's bytes of codes from generator `seed`, every other byte `0xa5`.
    pub(super) fn bytes(&self, seed: u64) -> Vec<u8> {
        let mut random = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = move || {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (random >> 33) as u16
        };
        let mut bytes = vec![0xa5; self.length() as usize];
        let unit = self.unit() as usize;
        let mut put = |at: u64, word: u16| {
            let at = at as usize;
            match unit {
                1 => bytes[at] = word as u8,
                _ => bytes[at..at + 2].copy_from_slice(&word.to_le_bytes()),
            }
        };
        let [w, h] = self.size;
        for y in 0..h {
            for x in 0..w {
                put(y * self.row + x * self.unit(), next());
            }
        }
        let [cw, ch] = self.chroma_size();
        for y in 0..ch {
            for x in 0..2 * cw {
                put(self.chroma_at + y * self.row + x * self.unit(), next());
            }
        }
        bytes
    }

    /// The code at byte `at` of `bytes`: a byte, or a word's high 10 bits.
    fn code(&self, bytes: &[u8], at: u64) -> u16 {
        let at = at as usize;
        match self.unit() {
            1 => u16::from(bytes[at]),
            _ => u16::from_le_bytes([bytes[at], bytes[at + 1]]) >> 6,
        }
    }

    /// Each texel's red, green and blue `v` of `v / 2048` Metal reads of `bytes`, row by row: the
    /// luma code at its place, the four nearest chroma pairs weighted 9/3/3/1 and rounded to a
    /// code, converted (`metal2vulkan::bind::ycbcr_read`).
    pub(super) fn model(&self, bytes: &[u8]) -> Vec<[u16; 3]> {
        self.sums(bytes)
            .into_iter()
            .map(|(luma, sums)| {
                let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
                ycbcr_read(self.codes(), self.conversion, [luma, cb, cr])
            })
            .collect()
    }

    /// Each texel's red, green and blue through the device's conversion of `bytes`, row by row:
    /// Vulkan's of the luma code and the chroma sums over 16, unrounded
    /// (`ycbcr_device_reads::vulkan_read`), which is Vulkan's midpoint reconstruction through a
    /// linear chroma filter at a luma texel's centre.
    pub(super) fn device_model(&self, bytes: &[u8]) -> Vec<[f64; 3]> {
        self.sums(bytes)
            .into_iter()
            .map(|(luma, sums)| {
                let [cb, cr] = sums.map(|s| f64::from(s) / 16.0);
                vulkan_read(self.codes(), self.conversion, [f64::from(luma), cb, cr])
            })
            .collect()
    }

    /// Asserts that the `float4`s `raw` (a `read()` of every texel, or a nearest sample at each
    /// centre) of a texture of the surface over `bytes` are its texels: Metal's exactly
    /// ([`Surface::model`]), or where the device converted the texture itself (`by_device`), the
    /// device's conversion within its bound ([`Surface::device_model`], `ycbcr_device_reads`).
    pub(super) fn assert_texels(&self, raw: &[u8], bytes: &[u8], by_device: bool, case: &str) {
        match by_device {
            true => {
                let want = self.device_model(bytes);
                assert_device_reads(raw, &want, (YCbCrOutput::Converted, false), case);
            }
            false => assert!(on_grid(raw) == self.model(bytes), "{case}: Metal's texels"),
        }
    }

    /// Each texel's luma code and its four nearest chroma pairs' Cb and Cr codes weighted
    /// 9/3/3/1, in sixteenths, row by row.
    fn sums(&self, bytes: &[u8]) -> Vec<(u16, [u32; 2])> {
        let taps = |q: u64, n: u64| -> [(u64, u32); 2] {
            let c = q / 2;
            if q % 2 == 1 {
                [(c, 3), ((c + 1).min(n - 1), 1)]
            } else {
                [(c.saturating_sub(1), 1), (c, 3)]
            }
        };
        let [w, h] = self.size;
        let [cw, ch] = self.chroma_size();
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let mut sums = [0u32; 2];
                for (cy, wy) in taps(y, ch) {
                    for (cx, wx) in taps(x, cw) {
                        let at = self.chroma_at + cy * self.row + 2 * cx * self.unit();
                        sums[0] += wx * wy * u32::from(self.code(bytes, at));
                        sums[1] += wx * wy * u32::from(self.code(bytes, at + self.unit()));
                    }
                }
                let luma = self.code(bytes, y * self.row + x * self.unit());
                out.push((luma, sums));
            }
        }
        out
    }

    /// The texture of the surface's format over `buffer`'s plane 0, its luma plane at
    /// `luma_at`, with the surface's facts (its chroma plane at `luma_at + chroma_at`).
    pub(super) fn texture(&self, buffer: &Buffer, luma_at: u64) -> Result<Texture, Error> {
        let [w, h] = self.size;
        let mut d = TextureDescriptor::texture_2d(self.format, w, h, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let ostype = u32::from_be_bytes(self.ostype);
        let facts = IoSurfaceYCbCr::new(ostype, 0, self.size, luma_at + self.chroma_at, self.row);
        let facts = match self.matrix {
            Some(matrix) => facts.with_matrix(matrix),
            None => facts,
        };
        let plane = IoSurfacePlane::new(luma_at, self.row, true).with_ycbcr(facts);
        buffer.new_texture_with_iosurface_plane(&d, plane)
    }
}

/// Metal's step of a linear sample at fractions `[a, b]` (in `1/256`) past texel `(x, y)` of
/// texels `texels` of `size`, channel `c`: the taps `(x, y)` to `(x + 1, y + 1)` clamped to the
/// edge, an axis whose two taps address one texel weighed `(256, 0)`, each term `k · w` truncated by
/// `2^7`, then `(T + 2^8) >> 9` (the YCbCr lane's `metal_step`).
fn filtered(texels: &[[u16; 3]], [w, h]: [u64; 2], (x, y): (u64, u64), c: usize) -> u16 {
    let [a, b] = ycbcr_textures::BETWEEN;
    let (x1, y1) = ((x + 1).min(w - 1), (y + 1).min(h - 1));
    let a = if x1 == x { 0 } else { a };
    let b = if y1 == y { 0 } else { b };
    let k = |x: u64, y: u64| u32::from(texels[(y * w + x) as usize][c]);
    let taps = [
        (k(x, y), 256 - a, 256 - b),
        (k(x1, y), a, 256 - b),
        (k(x, y1), 256 - a, b),
        (k(x1, y1), a, b),
    ];
    let t: u32 = taps.into_iter().map(|(k, wx, wy)| (k * wx * wy) >> 7).sum();
    ((t + 256) >> 9) as u16
}

/// The lane's kernels (`ycbcr_textures::KERNELS`) this module dispatches: `read()` of each texel,
/// a sample at its centre with a `float` and a `half` result, and a sample between it and the
/// next ([`ycbcr_textures::BETWEEN`]).
pub(super) struct Kernels {
    read: ComputePipelineState,
    pub(super) sample: ComputePipelineState,
    sample_half: ComputePipelineState,
    between: ComputePipelineState,
}

impl Kernels {
    pub(super) fn new(device: &Device) -> Kernels {
        let library = device
            .new_library_with_air(&llvm_assemble_text(ycbcr_textures::KERNELS).unwrap())
            .unwrap();
        let state = |name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(StageSource::Function(
                        function.function_descriptor(),
                    )),
                    &PipelineOptions::default(),
                )
                .unwrap()
        };
        Kernels {
            read: state("read_texels"),
            sample: state("sample_texels"),
            sample_half: state("sample_half"),
            between: state("sample_between"),
        }
    }
}

/// A sampler of `filter` for both min and mag.
pub(super) fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    device.new_sampler_state(&d).unwrap()
}

/// The `v` of `v / 2048` of each channel `state` writes of `texture` of `size` through `sampler`,
/// one thread a texel, in a command buffer of its own; checked to be on the grid, alpha 1.
pub(super) fn texels(
    device: &Device,
    state: &ComputePipelineState,
    (texture, size): (&Texture, [u64; 2]),
    sampler: Option<&SamplerState>,
) -> Vec<[u16; 3]> {
    on_grid(&raw_texels(device, state, (texture, size), sampler))
}

/// The `v` of `v / 2048` of each channel of the `float4`s `raw`, checked to be on the grid, alpha
/// 1.
fn on_grid(raw: &[u8]) -> Vec<[u16; 3]> {
    raw.as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
            assert_eq!(c(3), 1.0, "alpha 1");
            std::array::from_fn(|i| {
                let v = c(i) * 2048.0;
                assert!(
                    v == v.round() && (0.0..=2048.0).contains(&v),
                    "{v} off the grid"
                );
                v as u16
            })
        })
        .collect()
}

/// The `float4`s `state` writes of `texture` of `size` through `sampler`, one thread a texel, in a
/// command buffer of its own, as their bytes.
pub(super) fn raw_texels(
    device: &Device,
    state: &ComputePipelineState,
    (texture, size): (&Texture, [u64; 2]),
    sampler: Option<&SamplerState>,
) -> Vec<u8> {
    let [w, h] = size;
    let out = device
        .new_buffer(16 * w * h, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    if let Some(sampler) = sampler {
        encoder.set_sampler_state(Some(sampler), 0).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(w, h, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, 16 * w * h).unwrap()
}

/// The counts of the YCbCr paths between two readings: textures made reading their planes in
/// place, kept on their image for the device, planes bound, refreshes and whole conversions.
pub(super) fn ycbcr_work(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    [
        after.ycbcr_textures_in_place - before.ycbcr_textures_in_place,
        after.ycbcr_images_kept_device - before.ycbcr_images_kept_device,
        after.ycbcr_planes_bound - before.ycbcr_planes_bound,
        after.ycbcr_refreshes - before.ycbcr_refreshes,
        after.ycbcr_whole_conversions - before.ycbcr_whole_conversions,
    ]
}

/// Whether `device` reads two-plane YCbCr textures' planes in place: its profile's answer, which
/// is the device's kind (an integrated GPU or a CPU device, `hasUnifiedMemory`).
pub(super) fn reads_in_place(device: &Device) -> bool {
    let in_place = device
        .compile_device()
        .profile()
        .reads_ycbcr_planes_in_place();
    assert_eq!(
        in_place,
        device.properties().has_unified_memory,
        "{}: the profile's rule is the device's kind ({:?})",
        device.name(),
        device.properties().kind
    );
    in_place
}

/// The samples of each surface, then of new codes the CPU wrote into its planes, and a `read()`:
/// each texel Metal's (or, where the device converted the texture itself, the device's conversion
/// of the planes, [`Surface::assert_texels`]), and on a device whose memory is the host's the
/// samples read the planes in place, converting nothing, where any other device converts the
/// image. Where the planes are read in place a linear sample at a texel's centre and between
/// texels is Metal's texel and Metal's filter of them; a device that keeps the image filters it as
/// the device does under every policy (`PrecisionNote::DeviceSampling`), which this module does not
/// sample. The textures the device converted itself are counted.
fn samples_read_the_planes(device: &Device) -> u64 {
    let in_place = reads_in_place(device);
    eprintln!(
        "YCBCR-IN-PLACE {} ({:?}): in place {in_place}",
        device.name(),
        device.properties().kind
    );
    let kernels = Kernels::new(device);
    let nearest = sampler(device, SamplerMinMagFilter::Nearest);
    let linear = sampler(device, SamplerMinMagFilter::Linear);
    let mut by_devices = 0;
    for (seed, surface) in (1..).zip(SURFACES) {
        let what = format!("{:?} {:?}", surface.format, surface.conversion);
        let mut bytes = surface.bytes(seed);
        let buffer = buffer_with(device, &bytes).unwrap();
        let before = device.work_counts();
        let converted = device_conversions(device);
        let texture = surface.texture(&buffer, 0).unwrap();
        let by_device = device_conversions(device) > converted;
        by_devices += u64::from(by_device);
        let made = ycbcr_work(&before, &device.work_counts());
        let want = if in_place { [1, 0] } else { [0, 1] };
        assert_eq!(made[..2], want, "{what}: in place, or kept for the device");
        let size = surface.size;
        for round in 0..2 {
            let model = surface.model(&bytes);
            let before = device.work_counts();
            let centres = raw_texels(device, &kernels.sample, (&texture, size), Some(&nearest));
            // Read in place, through the texel buffer (these surfaces' rows are no driver's
            // linear image's), the samples are Metal's; elsewhere the image's.
            let case = format!("{what} round {round}: nearest samples");
            surface.assert_texels(&centres, &bytes, by_device && !in_place, &case);
            // A device that keeps the image filters it as the device does, off Metal's grid: its
            // linear samples are another row's (docs/FAST-PATHS.md).
            if in_place {
                let half = texels(
                    device,
                    &kernels.sample_half,
                    (&texture, size),
                    Some(&linear),
                );
                assert!(half == model, "{what} round {round}: half linear samples");
                let between = texels(device, &kernels.between, (&texture, size), Some(&linear));
                for (i, got) in (0u64..).zip(&between) {
                    let at = (i % size[0], i / size[0]);
                    let want: [u16; 3] = std::array::from_fn(|c| filtered(&model, size, at, c));
                    assert_eq!(*got, want, "{what} round {round}: texel {at:?}");
                }
            }
            let work = ycbcr_work(&before, &device.work_counts());
            eprintln!("YCBCR-IN-PLACE {what} round {round}: work {work:?}");
            match in_place {
                // Three dispatches, each binding the planes, none converting.
                true => assert_eq!(work[2..], [3, 0, 0], "{what} round {round}"),
                false => {
                    assert_eq!(work[2], 0, "{what} round {round}: planes bound");
                    assert!(work[3] >= 1, "{what} round {round}: the image's refresh");
                }
            }
            // The CPU writes new codes into the planes, which no command uses now.
            bytes = surface.bytes(seed + 100);
            let contents = buffer.contents().unwrap();
            // SAFETY: the buffer's bytes, which no command uses (each dispatch was waited for).
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
        }
        let before = device.work_counts();
        let read = raw_texels(device, &kernels.read, (&texture, size), None);
        let work = ycbcr_work(&before, &device.work_counts());
        surface.assert_texels(&read, &bytes, by_device, &format!("{what}: read()"));
        assert_eq!(work[2..], [0, 1, 0], "{what}: a read() converts the image");
    }
    by_devices
}

/// Under the lane's policy (`DevicePolicy::exact_answers`), a texture's samples read its planes in
/// place on a device whose memory is the host's, Metal's texels and Metal's filter of them there
/// ([`samples_read_the_planes`]); no generic filter notice is announced, and `ycbcr-filter-not-exact`
/// only where the device converted a texture itself (`ycbcr_device_reads`).
#[test]
fn a_ycbcr_texture_s_samples_read_its_planes_in_place_where_the_memory_is_the_host_s() {
    on_device(|device, messages| {
        let by_device = samples_read_the_planes(device);
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::HalfSampleFilter,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
        assert_eq!(
            notices.get(&Degradation::YCbCrFilterNotExact),
            (by_device > 0).then_some(&1),
            "{notices:?}"
        );
    });
}

/// Under the runtime's defaults the device takes the same form: in place on a device whose memory
/// is the host's, where the samples are Metal's exactly, unannounced; elsewhere the image, sampled
/// through the device's filter under the defaults.
#[test]
fn a_ycbcr_texture_reads_its_planes_in_place_under_the_defaults() {
    let defaults = |options: &mut metal2vulkan_runtime::DeviceOptions| {
        options.policy = DevicePolicy::default();
    };
    on_device_with(defaults, |device, messages| {
        samples_read_the_planes(device);
        let notices = counts(&messages.take_notices());
        if reads_in_place(device) {
            for kind in [
                Degradation::FloatSampleFilter,
                Degradation::HalfSampleFilter,
                Degradation::YCbCrFilterNotExact,
            ] {
                assert_eq!(notices.get(&kind), None, "{notices:?}");
            }
        }
    });
}

/// A texture whose planes reach past their buffer (Metal reads zeros there) and one made with a
/// swizzle keep their image on a device whose memory is the host's, counted by the reason
/// (`ycbcr_images_kept_reach`, `ycbcr_images_kept_swizzle`), and the first still samples Metal's
/// texels of the bytes and zeros past them.
#[test]
fn a_ycbcr_texture_the_form_does_not_read_keeps_its_image() {
    on_device(|device, _| {
        if !reads_in_place(device) {
            eprintln!("{}: reads no planes in place", device.name());
            return;
        }
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let surface = SURFACES[0];
        let bytes = surface.bytes(7);
        // The buffer ends 100 bytes before the chroma plane's last row does.
        let short = surface.length() as usize - 100;
        let buffer = buffer_with(device, &bytes[..short]).unwrap();
        let before = device.work_counts();
        let texture = surface.texture(&buffer, 0).unwrap();
        let after = device.work_counts();
        assert_eq!(
            (
                after.ycbcr_images_kept_reach - before.ycbcr_images_kept_reach,
                after.ycbcr_textures_in_place - before.ycbcr_textures_in_place,
            ),
            (1, 0),
            "planes past the buffer"
        );
        let mut zeroed = bytes.clone();
        zeroed[short..].fill(0);
        let got = texels(
            device,
            &kernels.sample,
            (&texture, surface.size),
            Some(&nearest),
        );
        assert!(got == surface.model(&zeroed), "the bytes, then zeros");
        let whole = buffer_with(device, &bytes).unwrap();
        let [w, h] = surface.size;
        let mut d = TextureDescriptor::texture_2d(surface.format, w, h, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        d.swizzle = metal2vulkan::metal::TextureSwizzleChannels {
            red: metal2vulkan::metal::TextureSwizzle::Blue,
            green: metal2vulkan::metal::TextureSwizzle::Green,
            blue: metal2vulkan::metal::TextureSwizzle::Red,
            alpha: metal2vulkan::metal::TextureSwizzle::Alpha,
        };
        let facts = IoSurfaceYCbCr::new(
            u32::from_be_bytes(surface.ostype),
            0,
            surface.size,
            surface.chroma_at,
            surface.row,
        );
        let plane = IoSurfacePlane::new(0, surface.row, true).with_ycbcr(facts);
        let before = device.work_counts();
        whole.new_texture_with_iosurface_plane(&d, plane).unwrap();
        let after = device.work_counts();
        assert_eq!(
            after.ycbcr_images_kept_swizzle - before.ycbcr_images_kept_swizzle,
            1,
            "a swizzle"
        );
    });
}

/// The planes images made between two readings, then the textures kept from one by each reason:
/// the device's conversion, the memory, the layout (`WorkCounts::ycbcr_planes_images*`).
fn planes_images(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.ycbcr_planes_images - before.ycbcr_planes_images,
        after.ycbcr_planes_images_kept_conversion - before.ycbcr_planes_images_kept_conversion,
        after.ycbcr_planes_images_kept_memory - before.ycbcr_planes_images_kept_memory,
        after.ycbcr_planes_images_kept_layout - before.ycbcr_planes_images_kept_layout,
    ]
}

/// A texture whose planes are read in place is also an image the device's own sampler Y′CbCr
/// conversion samples where the device converts the 4:2:0 format as a linear image with Metal's
/// reconstruction (`WorkCounts::ycbcr_planes_images`, the runtime's `texture/ycbcr_planes.rs`),
/// and is counted by why where it is not: each such texture once. Through the image a nearest
/// sample of each texel is the device's conversion, Vulkan's of the luma code and Metal's
/// chroma sums unrounded within the device's bound, announced (`ycbcr-filter-not-exact`);
/// without it, Metal's texel through the texel buffer, unannounced.
#[test]
fn a_ycbcr_texture_s_planes_image_samples_through_the_device_s_conversion() {
    on_device(|device, messages| {
        if !reads_in_place(device) {
            eprintln!("{}: reads no planes in place", device.name());
            return;
        }
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let mut images = 0;
        for (seed, surface) in (11..).zip(PLANES_SURFACES) {
            let what = format!("{:?} {:?}", surface.format, surface.conversion);
            let bytes = surface.bytes(seed);
            let buffer = buffer_with(device, &bytes).unwrap();
            let before = device.work_counts();
            let texture = surface.texture(&buffer, 0).unwrap();
            let after = device.work_counts();
            let made = planes_images(&before, &after);
            eprintln!(
                "YCBCR-PLANES-IMAGE {} {what}: image, or kept for the conversion, the memory, \
                 the layout: {made:?}",
                device.name()
            );
            assert_eq!(
                after.ycbcr_textures_in_place - before.ycbcr_textures_in_place,
                1,
                "{what}: in place"
            );
            assert_eq!(made.iter().sum::<u64>(), 1, "{what}: one answer, {made:?}");
            images += made[0];
            let before = device.work_counts();
            let raw = raw_texels(
                device,
                &kernels.sample,
                (&texture, surface.size),
                Some(&nearest),
            );
            let after = device.work_counts();
            let bound = (
                after.ycbcr_planes_images_bound - before.ycbcr_planes_images_bound,
                after.ycbcr_planes_bound - before.ycbcr_planes_bound,
            );
            match made[0] {
                1 => {
                    assert_eq!(bound, (1, 0), "{what}: the planes image bound");
                    let want = surface.device_model(&bytes);
                    assert_device_reads(&raw, &want, (YCbCrOutput::Converted, false), &what);
                }
                _ => {
                    assert_eq!(bound, (0, 1), "{what}: the texel buffer bound");
                    assert!(
                        on_grid(&raw) == surface.model(&bytes),
                        "{what}: Metal's texels"
                    );
                }
            }
        }
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.contains_key(&Degradation::YCbCrFilterNotExact),
            images > 0,
            "{notices:?}"
        );
    });
}
