//! The color resolve modules: Metal's measured tables as the module docs state them, valid on
//! every committed profile (L3, `G-val`), needing nothing of a device, and exact on the CPU
//! interpreter: the fragment stage's arithmetic, run in a kernel over every pair of 8-bit codes and
//! over random and edge texels, writes the float the attachment stores as Metal's code.

use super::*;
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::interp::{run, Buffer, Config, Order, Outcome, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};
use std::path::PathBuf;

/// Every format the module docs list: the twin resolve's.
const CLOSED: [Format; 8] = [
    Format::R8_UNORM,
    Format::R8G8_UNORM,
    Format::R8G8B8A8_UNORM,
    Format::B8G8R8A8_UNORM,
    Format::R8_SRGB,
    Format::R8G8_SRGB,
    Format::R8G8B8A8_SRGB,
    Format::B8G8R8A8_SRGB,
];

/// Every format the module docs list, and none of the formats the device's `AVERAGE` resolves
/// where they are no twin resolve's: the rules of those are the device's own.
#[test]
fn the_formats_are_the_twin_resolve_s() {
    for format in CLOSED {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        assert_eq!(resolve.format(), format);
        // Each channel lies inside the texel, apart from every other.
        let size = format.block().expect("block").size * 8;
        let mut used = 0u32;
        for channel in resolve.channels().iter().flatten() {
            let offset = u32::from(channel.offset);
            assert!(offset + 8 <= size, "{format:?}");
            let mask = 0xff << offset;
            assert_eq!(used & mask, 0, "{format:?}");
            used |= mask;
        }
    }
    for format in [
        Format::A8_UNORM,
        Format::R8_SNORM,
        Format::R16G16B16A16_UNORM,
        Format::A2B10G10R10_UNORM_PACK32,
        Format::R32_SFLOAT,
        Format::R16G16B16A16_SFLOAT,
        Format::R5G6B5_UNORM_PACK16,
        Format::B10G11R11_UFLOAT_PACK32,
        Format::E5B9G9R9_UFLOAT_PACK32,
        Format::A4R4G4B4_UNORM_PACK16,
        Format::R8_UINT,
        Format::D32_SFLOAT,
    ] {
        assert_eq!(ColorResolveFormat::of(format), None, "{format:?}");
    }
}

/// The sRGB tables are what the module docs say: the 12-bit linear value of each code rounded,
/// and Metal's encode, ascending, the exact encode rounded half up but at the four entries
/// measured one code higher.
#[test]
fn the_srgb_tables_are_metals_measured_ones() {
    for (code, &linear) in SRGB_RESOLVE_LINEAR.iter().enumerate() {
        assert_eq!(
            f64::from(linear),
            (srgb_decode(code as u32) * 4095.0).round(),
            "{code}"
        );
    }
    assert!(SRGB_RESOLVE_THRESHOLDS.windows(2).all(|w| w[0] < w[1]));
    let encode = |linear: u16| SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= linear);
    let mut higher = Vec::new();
    for linear in 0..4096u16 {
        let l = f64::from(linear) / 4095.0;
        let exact = if l <= 0.003_130_8 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        let rounded = (exact * 255.0 + 0.5).floor() as usize;
        match encode(linear) {
            code if code == rounded => {}
            code if code == rounded + 1 => higher.push(linear),
            code => panic!("{linear}: {code} against the exact encode's {rounded}"),
        }
    }
    assert_eq!(higher, [804, 1371, 2121, 2569]);
}

/// The sources a test reads `format` from: the input attachment, in its own format and, where that
/// is another, its raw twin's.
fn sources(format: ColorResolveFormat) -> Vec<ResolveSource> {
    let mut sources = vec![ResolveSource::InputAttachment];
    if format.raw_format() != format.format() {
        sources.push(ResolveSource::RawInputAttachment);
    }
    sources
}

/// The resolves a test runs of `format`: its result written in its own format and, where its raw
/// twin is another, through the twin ([`ColorResolveFormat::written_raw`]), each from every source
/// ([`sources`]).
fn cases(format: ColorResolveFormat) -> Vec<(ColorResolveFormat, ResolveSource)> {
    let mut written = vec![format];
    if format.written_raw() != format {
        written.push(format.written_raw());
    }
    written
        .into_iter()
        .flat_map(|format| sources(format).into_iter().map(move |s| (format, s)))
        .collect()
}

/// The requirements of `m` as serialized.
fn needs(m: TypedModule) -> Requirements {
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .requirements
}

/// A format of each rule.
fn representatives() -> Vec<ColorResolveFormat> {
    [Format::R8G8B8A8_UNORM, Format::B8G8R8A8_SRGB]
        .into_iter()
        .map(|format| ColorResolveFormat::of(format).expect("closed"))
        .collect()
}

/// The fragment stage needs nothing of a device beyond Vulkan 1.2's shaders, at 2, 4 and 8
/// samples and from every source.
#[test]
fn the_fragment_stage_needs_nothing_of_a_device() {
    for format in representatives() {
        for samples in [2, 4, 8] {
            for (format, source) in cases(format) {
                let r = needs(fragment_typed_module(format, samples, source).expect("module"));
                assert!(
                    r.features.is_empty(),
                    "{format:?} x{samples} {source:?}: {r:?}"
                );
            }
        }
    }
}

/// Every committed profile validates each representative's modules.
#[test]
fn the_modules_are_valid_on_every_committed_profile() {
    for (name, profile) in committed_profiles().expect("profiles") {
        for format in representatives() {
            let device = (format.device_srgb(), ResolveSource::InputAttachment);
            for (format, source) in cases(format).into_iter().chain([device]) {
                for samples in [2, 4] {
                    let m = fragment_typed_module(format, samples, source).expect("module");
                    let _ = assert_valid(m, &profile);
                }
            }
            let _ = name;
        }
    }
}

/// On a committed device the modules come numbered with the fixed binding; a sample count that is
/// no resolve's is refused, and so are sample planes of a twin resolve's format, which reads the
/// attachment itself.
#[test]
fn the_modules_come_numbered_and_refuse_another_count_or_planes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(&path).expect("read");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device = Device::new(desc, DevicePolicy::default()).expect("device");
    let format = representatives()[0];
    for (format, source) in cases(format) {
        let modules =
            color_resolve_modules(&device, format, 4, source, Feature::Format).expect("modules");
        assert!(!modules.vertex.is_empty() && !modules.fragment.is_empty());
    }
    for samples in [0, 1, 3, 128] {
        let refused = color_resolve_modules(
            &device,
            format,
            samples,
            ResolveSource::InputAttachment,
            Feature::Format,
        );
        assert!(matches!(refused, Err(Error::Internal(_))), "{samples}");
    }
    for format in representatives() {
        let copy = PlanesCopy::for_planes(format.format()).expect("copy");
        let planes = ResolveSource::Planes(copy);
        assert!(
            fragment_typed_module(format, 4, planes).is_err(),
            "{format:?}"
        );
        let refused = color_resolve_modules(&device, format, 4, planes, Feature::Format);
        assert!(refused.is_err(), "{format:?}");
    }
    // The device's sRGB form reads the attachment in the sRGB format alone.
    let srgb = ColorResolveFormat::of(Format::B8G8R8A8_SRGB)
        .expect("closed")
        .device_srgb();
    assert!(color_resolve_modules(
        &device,
        srgb,
        4,
        ResolveSource::InputAttachment,
        Feature::Format
    )
    .is_ok());
    assert!(fragment_typed_module(srgb, 4, ResolveSource::RawInputAttachment).is_err());
}

// ---- The arithmetic on the interpreter ----

/// The kernel's buffers: each invocation's samples (four words a texel), and its four output
/// words.
const SAMPLES_IN: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
const RESOLVED_OUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// A kernel running the fragment stage's arithmetic: invocation `i` reads its `samples` texels as
/// four words each from `SAMPLES_IN` (the attachment's floats' bits for
/// [`ResolveSource::InputAttachment`], its raw twin's for [`ResolveSource::RawInputAttachment`]),
/// and writes the output floats' bits to `RESOLVED_OUT`.
fn arithmetic_module(
    format: ColorResolveFormat,
    samples: u32,
    source: ResolveSource,
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b).expect("types");
    let srgb = (format.srgb() && !format.decodes_on_device())
        .then(|| SrgbTables::declare(&mut b, t, format, source))
        .transpose()
        .expect("tables");
    let words = super::super::visibility_combine::words_buffer;
    let input = words(&mut b, SAMPLES_IN, t.u32_).expect("in");
    let output = words(&mut b, RESOLVED_OUT, t.u32_).expect("out");
    let uvec3 = b
        .ty(Type::Vector {
            elem: t.u32_,
            len: 3,
        })
        .expect("uvec3");
    let uvec4 = b
        .ty(Type::Vector {
            elem: t.u32_,
            len: 4,
        })
        .expect("uvec4");
    let invocation = b
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)
        .expect("id");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let build = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let zero = uint(f, t, 0)?;
        let invocation = f.module().global_value(invocation)?;
        let invocation = f.load(invocation, MemAccess::default())?;
        let i = f.composite_extract(invocation, &[0])?;
        let input = f.module().global_value(input)?;
        let output = f.module().global_value(output)?;
        let per = uint(f, t, samples * 4)?;
        let base = f.binary(BinOp::IMul, i, per)?;
        let mut texels = Vec::new();
        for s in 0..samples {
            let mut lanes = Vec::new();
            for w in 0..4 {
                let k = uint(f, t, s * 4 + w)?;
                let at = f.binary(BinOp::IAdd, base, k)?;
                let p = f.access_chain(input, &[zero, at])?;
                lanes.push(f.load(p, MemAccess::default())?);
            }
            let texel = f.composite_construct(uvec4, &lanes)?;
            texels.push(f.convert(ConvOp::Bitcast, texel, t.vec4)?);
        }
        let four = uint(f, t, 4)?;
        let out_base = f.binary(BinOp::IMul, i, four)?;
        let lanes = resolve_texel(f, t, srgb, format, &texels, source)?;
        for (component, channel) in format.channels.iter().enumerate() {
            if channel.is_none() {
                continue;
            }
            let bits = f.convert(ConvOp::Bitcast, lanes[component], t.u32_)?;
            let c = uint(f, t, component as u32)?;
            let at = f.binary(BinOp::IAdd, out_base, c)?;
            let p = f.access_chain(output, &[zero, at])?;
            f.store(p, bits, MemAccess::default())?;
        }
        Ok(())
    };
    build(&mut f).expect("body");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// SplitMix64.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Metal's resolve of one channel's codes (the module docs), written independently of the
/// modules: a `u32` code.
fn metal(rule: ResolveRule, codes: &[u32]) -> u32 {
    let n = codes.len() as i64;
    match rule {
        ResolveRule::Unorm => {
            let sum: i64 = codes.iter().map(|&c| i64::from(c)).sum();
            ((sum + n / 2) / n) as u32
        }
        ResolveRule::Srgb => {
            let sum: i64 = codes
                .iter()
                .map(|&c| i64::from(SRGB_RESOLVE_LINEAR[c as usize]))
                .sum();
            let linear = ((sum + n / 2) / n) as u16;
            SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= linear) as u32
        }
    }
}

/// The float an attachment of `rule` reads as `code`.
fn read_as(rule: ResolveRule, code: u32) -> f32 {
    match rule {
        ResolveRule::Unorm => (f64::from(code) / 255.0) as f32,
        ResolveRule::Srgb => srgb_decode(code) as f32,
    }
}

/// The code an attachment of `rule` stores for the output float `x`, converting to nearest.
fn stored_as(rule: ResolveRule, x: f32) -> u32 {
    let x = f64::from(x);
    match rule {
        ResolveRule::Unorm => (x * 255.0).round() as u32,
        ResolveRule::Srgb => (0..256u32)
            .min_by(|&a, &b| {
                let da = (srgb_decode(a) - x).abs();
                let db = (srgb_decode(b) - x).abs();
                da.total_cmp(&db)
            })
            .expect("a code"),
    }
}

/// The kernel's input words of `texels` (each sample's codes by component, one texel a row) of
/// `format` as read from `source`.
fn sample_words(
    format: ColorResolveFormat,
    source: ResolveSource,
    texels: &[Vec<[u32; 4]>],
) -> Vec<u8> {
    let mut input = Vec::new();
    for texel in texels {
        for sample in texel {
            let mut words = [0u32; 4];
            for (component, channel) in format.channels.iter().enumerate() {
                let Some(channel) = channel else { continue };
                let code = sample[component];
                // The raw twin reads an sRGB code as an 8-bit unorm one.
                let read = match source {
                    ResolveSource::RawInputAttachment => ResolveRule::Unorm,
                    _ => channel.rule,
                };
                words[component] = read_as(read, code).to_bits();
            }
            input.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        }
    }
    input
}

/// Run `format`'s arithmetic at `samples` from `source` over the kernel's `input` words of
/// `texels` texels.
fn dispatch(
    format: ColorResolveFormat,
    samples: usize,
    source: ResolveSource,
    input: Vec<u8>,
    texels: usize,
) -> Outcome {
    let m = arithmetic_module(format, samples as u32, source);
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(SAMPLES_IN),
                address: None,
                bytes: input,
            },
            Buffer {
                slot: Some(RESOLVED_OUT),
                address: None,
                bytes: vec![0; texels * 16],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let cfg = Config {
        order: Order::Forward,
        workgroups: [texels as u32, 1, 1],
        ..Config::default()
    };
    run(&m, &cfg, res).expect("run")
}

/// Run `format`'s arithmetic at `samples` from `source` over `texels` (each sample's codes by
/// component, one texel a row), and check every channel against [`metal`].
fn check(
    format: ColorResolveFormat,
    samples: usize,
    source: ResolveSource,
    texels: &[Vec<[u32; 4]>],
) {
    let input = sample_words(format, source, texels);
    let out = dispatch(format, samples, source, input, texels.len())
        .buffers
        .remove(1);
    for (i, texel) in texels.iter().enumerate() {
        for (component, channel) in format.channels.iter().enumerate() {
            let Some(channel) = channel else { continue };
            let at = (i * 4 + component) * 4;
            let word = u32::from_le_bytes(out[at..at + 4].try_into().expect("word"));
            let codes: Vec<u32> = texel.iter().map(|s| s[component]).collect();
            let want = metal(channel.rule, &codes);
            // Written through the raw twin, an sRGB channel's float is stored as an 8-bit unorm
            // one.
            let stored = match channel.rule {
                ResolveRule::Srgb if format.written_raw => ResolveRule::Unorm,
                rule => rule,
            };
            let got = stored_as(stored, f32::from_bits(word));
            assert_eq!(
                got,
                want,
                "{:?} written as {:?} x{samples} {source:?} component {component}: {codes:x?} \
                 wrote {word:#x}",
                format.format(),
                format.written_format()
            );
        }
    }
}

/// The codes a test draws: random, and the edges of an 8-bit code.
fn draw(r: &mut SplitMix) -> u32 {
    if r.next().is_multiple_of(4) {
        let edges = [0, 1, 127, 128, 129, 254, 255];
        edges[(r.next() % edges.len() as u64) as usize]
    } else {
        (r.next() as u32) & 0xff
    }
}

/// Every pair of 8-bit codes of `format`'s one channel at 2 samples, through every source
/// ([`sources`]).
fn every_pair_of_8_bit_codes(format: Format) {
    let resolve = ColorResolveFormat::of(format).expect("closed");
    let component = resolve
        .channels()
        .iter()
        .position(Option::is_some)
        .expect("a channel");
    let sample = |code: u32| {
        let mut codes = [0; 4];
        codes[component] = code;
        codes
    };
    let texels: Vec<Vec<[u32; 4]>> = (0..65536u32)
        .map(|p| vec![sample(p & 0xff), sample(p >> 8)])
        .collect();
    for (resolve, source) in cases(resolve) {
        check(resolve, 2, source, &texels);
    }
}

#[test]
fn every_pair_of_unorm_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::R8_UNORM);
}

#[test]
fn every_pair_of_srgb_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::R8_SRGB);
}

/// Random and edge texels of the twin resolve's formats at 2, 4 and 8 samples, through every
/// source ([`sources`]).
#[test]
fn random_texels_of_every_format_resolve_as_metal() {
    let mut r = SplitMix(0x0052_504c);
    for format in [
        Format::R8G8_UNORM,
        Format::B8G8R8A8_UNORM,
        Format::R8G8B8A8_SRGB,
        Format::B8G8R8A8_SRGB,
    ] {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        for samples in [2usize, 4, 8] {
            let texels: Vec<Vec<[u32; 4]>> = (0..512)
                .map(|_| {
                    (0..samples)
                        .map(|_| {
                            let mut codes = [0u32; 4];
                            for (component, channel) in resolve.channels().iter().enumerate() {
                                if channel.is_some() {
                                    codes[component] = draw(&mut r);
                                }
                            }
                            codes
                        })
                        .collect()
                })
                .collect();
            for (resolve, source) in cases(resolve) {
                check(resolve, samples, source, &texels);
            }
        }
    }
}

#[test]
fn the_srgb_encode_of_each_code_s_linear_value_is_the_code() {
    for (code, &linear) in SRGB_RESOLVE_LINEAR.iter().enumerate() {
        assert_eq!(
            SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= linear),
            code,
            "{linear}"
        );
    }
}

/// Texels of `format` whose `samples` samples all hold the same codes: component `k` of texel `i`
/// the code `(i + 7k) mod 256`, every code of every channel.
fn alike_texels(format: ColorResolveFormat, samples: usize) -> Vec<Vec<[u32; 4]>> {
    (0..256)
        .map(|i| {
            let mut codes = [0u32; 4];
            for (k, channel) in format.channels().iter().enumerate() {
                if channel.is_some() {
                    codes[k] = ((i + 7 * k) % 256) as u32;
                }
            }
            vec![codes; samples]
        })
        .collect()
}

/// A pixel whose samples all hold the same codes resolves as the model of Metal's rules
/// ([`metal`]) says, which is the code (`resolve_alike`): every closed format at each of 2, 4 and
/// 8 samples, through every source ([`sources`]).
#[test]
fn alike_samples_resolve_as_metal() {
    for format in CLOSED {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        for samples in [2usize, 4, 8] {
            let texels = alike_texels(resolve, samples);
            for (resolve, source) in cases(resolve) {
                check(resolve, samples, source, &texels);
            }
        }
    }
}

/// A pixel of an sRGB format whose samples all hold the same bits skips the average: on the
/// interpreter, 64 such texels of each representative format at 4 samples take fewer lane-steps,
/// through every source, than the same texels with the last sample's first channel changed; a
/// format of unorm channels alone, which always averages, takes as many.
#[test]
fn alike_samples_skip_the_average() {
    let mut r = SplitMix(0x0041_4c53);
    for format in representatives() {
        for source in sources(format) {
            let first = format
                .channels()
                .iter()
                .position(Option::is_some)
                .expect("a channel");
            let codes: Vec<[u32; 4]> = (0..64)
                .map(|_| {
                    let mut codes = [0u32; 4];
                    for (k, channel) in format.channels().iter().enumerate() {
                        if channel.is_some() {
                            codes[k] = draw(&mut r);
                        }
                    }
                    codes
                })
                .collect();
            let inputs = [false, true].map(|change| {
                let texels: Vec<Vec<[u32; 4]>> = codes
                    .iter()
                    .map(|&c| {
                        let mut last = c;
                        if change {
                            last[first] ^= 2;
                        }
                        vec![c, c, c, last]
                    })
                    .collect();
                sample_words(format, source, &texels)
            });
            let [alike, changed] = inputs.map(|input| dispatch(format, 4, source, input, 64).steps);
            let what = format!(
                "{:?} {source:?}: {alike} against {changed}",
                format.format()
            );
            if format.srgb() {
                assert!(alike < changed, "{what}");
            } else {
                assert_eq!(alike, changed, "{what}");
            }
        }
    }
}

/// The raw twin reads an sRGB format's codes with no table lookup: on the interpreter, 64 random
/// `BGRA8Unorm_sRGB` texels at 4 samples take fewer lane-steps from
/// [`ResolveSource::RawInputAttachment`] than from [`ResolveSource::InputAttachment`], each sample
/// alike and with the last sample's red changed.
#[test]
fn the_raw_twin_reads_srgb_codes_in_fewer_steps() {
    let format = ColorResolveFormat::of(Format::B8G8R8A8_SRGB).expect("closed");
    let mut r = SplitMix(0x0052_4157);
    let codes: Vec<[u32; 4]> = (0..64)
        .map(|_| [0; 4].map(|_: u32| (r.next() as u32) & 0xff))
        .collect();
    for change in [false, true] {
        let texels: Vec<Vec<[u32; 4]>> = codes
            .iter()
            .map(|&c| {
                let mut last = c;
                if change {
                    last[0] ^= 2;
                }
                vec![c, c, c, last]
            })
            .collect();
        let [read, raw] = [
            ResolveSource::InputAttachment,
            ResolveSource::RawInputAttachment,
        ]
        .map(|source| {
            let input = sample_words(format, source, &texels);
            dispatch(format, 4, source, input, texels.len()).steps
        });
        assert!(raw < read, "changed {change}: {raw} against {read}");
    }
}

/// Written through the raw twin, an sRGB format's codes go out with no table: the fragment module
/// of `BGRA8Unorm_sRGB` at 4 samples, read through the raw twin, holds no 256-entry table of
/// linear values written raw ([`ColorResolveFormat::written_raw`]), and so is at least that many
/// words smaller than the one written in the format. Lane-steps do not show the saving: the
/// lookup takes as many instructions as the conversion and scale in its place, where the GPU pays
/// for a dynamically indexed `Private` array.
#[test]
fn the_raw_twin_writes_srgb_codes_with_no_table() {
    let format = ColorResolveFormat::of(Format::B8G8R8A8_SRGB).expect("closed");
    let source = ResolveSource::RawInputAttachment;
    let [own, raw] = [format, format.written_raw()].map(|format| {
        let m = fragment_typed_module(format, 4, source).expect("module");
        crate::spirv::serialize::serialize(m)
            .expect("serialize")
            .words
            .len()
    });
    assert!(raw + 256 <= own, "{raw} words against {own}");
}

// ---- The device's sRGB decode and encode ----

/// The code a device's sRGB store keeps for the linear `x`, its conversion modelled correctly
/// rounded: the encode (IEC 61966-2-1) of `x` in codes, to nearest.
fn device_encoded(x: f32) -> u32 {
    let x = f64::from(x).clamp(0.0, 1.0);
    let encoded = if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5).floor() as u32
}

/// The device's sRGB form ([`ColorResolveFormat::device_srgb`]) declares none of Metal's tables
/// and writes, for each sRGB channel, the mean of the decoded samples it read (their `f32` sum in
/// sample order times `1 / n`), which the sRGB view it writes through encodes; the alpha keeps
/// Metal's unorm rule: every format with an sRGB channel, random and edge texels at 2, 4 and 8
/// samples.
#[test]
fn the_device_srgb_form_writes_the_mean_of_the_decoded_samples() {
    let mut r = SplitMix(0x0044_5652);
    for format in [
        Format::R8_SRGB,
        Format::R8G8_SRGB,
        Format::R8G8B8A8_SRGB,
        Format::B8G8R8A8_SRGB,
    ] {
        let resolve = ColorResolveFormat::of(format)
            .expect("closed")
            .device_srgb();
        assert!(resolve.decodes_on_device(), "{format:?}");
        assert_eq!(resolve.written_format(), format, "{format:?}");
        let source = ResolveSource::InputAttachment;
        let tables = fragment_typed_module(resolve, 4, source).expect("module");
        let table =
            fragment_typed_module(ColorResolveFormat::of(format).expect("closed"), 4, source)
                .expect("module");
        let words = |m| {
            crate::spirv::serialize::serialize(m)
                .expect("serialize")
                .words
                .len()
        };
        // Metal's tables hold 256 entries and more: the device's form holds none of them.
        assert!(
            words(tables) + 256 <= words(table),
            "{format:?}: the device's form keeps a table"
        );
        for samples in [2usize, 4, 8] {
            let texels: Vec<Vec<[u32; 4]>> = (0..256)
                .map(|_| {
                    (0..samples)
                        .map(|_| {
                            let mut codes = [0u32; 4];
                            for (component, channel) in resolve.channels().iter().enumerate() {
                                if channel.is_some() {
                                    codes[component] = draw(&mut r);
                                }
                            }
                            codes
                        })
                        .collect()
                })
                .collect();
            let input = sample_words(resolve, source, &texels);
            let out = dispatch(resolve, samples, source, input, texels.len())
                .buffers
                .remove(1);
            for (i, texel) in texels.iter().enumerate() {
                for (component, channel) in resolve.channels().iter().enumerate() {
                    let Some(channel) = channel else { continue };
                    let at = (i * 4 + component) * 4;
                    let word = u32::from_le_bytes(out[at..at + 4].try_into().expect("word"));
                    let codes: Vec<u32> = texel.iter().map(|s| s[component]).collect();
                    let what = format!("{format:?} x{samples} component {component}: {codes:?}");
                    match channel.rule {
                        ResolveRule::Srgb => {
                            let mut sum = read_as(ResolveRule::Srgb, codes[0]);
                            for &c in &codes[1..] {
                                sum += read_as(ResolveRule::Srgb, c);
                            }
                            let mean = sum * (1.0 / samples as f32);
                            assert_eq!(word, mean.to_bits(), "{what}");
                        }
                        ResolveRule::Unorm => assert_eq!(
                            stored_as(ResolveRule::Unorm, f32::from_bits(word)),
                            metal(ResolveRule::Unorm, &codes),
                            "{what}"
                        ),
                    }
                }
            }
        }
    }
}

/// The device's sRGB form is a near-equivalent of Metal's rule, as the module docs state it: with
/// the device's conversions correctly rounded, of every pair of 8-bit codes at 2 samples, 1,758
/// resolve one code from Metal's and none further.
#[test]
fn the_device_srgb_form_stands_within_a_code_of_metal_s() {
    let resolve = ColorResolveFormat::of(Format::R8_SRGB)
        .expect("closed")
        .device_srgb();
    let source = ResolveSource::InputAttachment;
    let texels: Vec<Vec<[u32; 4]>> = (0..65536u32)
        .map(|p| vec![[p & 0xff, 0, 0, 0], [p >> 8, 0, 0, 0]])
        .collect();
    let input = sample_words(resolve, source, &texels);
    let out = dispatch(resolve, 2, source, input, texels.len())
        .buffers
        .remove(1);
    let mut apart = 0;
    for (i, texel) in texels.iter().enumerate() {
        let word = u32::from_le_bytes(out[i * 16..i * 16 + 4].try_into().expect("word"));
        let got = device_encoded(f32::from_bits(word));
        let want = metal(ResolveRule::Srgb, &[texel[0][0], texel[1][0]]);
        assert!(got.abs_diff(want) <= 1, "{texel:?}: {got} against {want}");
        apart += usize::from(got != want);
    }
    assert_eq!(apart, 1758);
}

// ---- The sRGB codes by estimate and correction ----

/// A kernel whose invocation `i` reads `words` words from `SAMPLES_IN` at `i * words` and writes
/// what `body` makes of them (with the module's sRGB tables) to word `i` of `RESOLVED_OUT`.
fn word_kernel(
    words: u32,
    body: &dyn Fn(&mut FunctionBuilder<'_>, Types, SrgbTables, &[Value]) -> BResult<Value>,
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b).expect("types");
    // Every table: an sRGB format's, read as floats and written in its own format.
    let srgb = ColorResolveFormat::of(Format::R8G8B8A8_SRGB).expect("closed");
    let tables =
        SrgbTables::declare(&mut b, t, srgb, ResolveSource::InputAttachment).expect("tables");
    let buffer = super::super::visibility_combine::words_buffer;
    let input = buffer(&mut b, SAMPLES_IN, t.u32_).expect("in");
    let output = buffer(&mut b, RESOLVED_OUT, t.u32_).expect("out");
    let uvec3 = b
        .ty(Type::Vector {
            elem: t.u32_,
            len: 3,
        })
        .expect("uvec3");
    let invocation = b
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)
        .expect("id");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let build = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let zero = uint(f, t, 0)?;
        let invocation = f.module().global_value(invocation)?;
        let invocation = f.load(invocation, MemAccess::default())?;
        let i = f.composite_extract(invocation, &[0])?;
        let input = f.module().global_value(input)?;
        let output = f.module().global_value(output)?;
        let per = uint(f, t, words)?;
        let base = f.binary(BinOp::IMul, i, per)?;
        let mut read = Vec::new();
        for w in 0..words {
            let k = uint(f, t, w)?;
            let at = f.binary(BinOp::IAdd, base, k)?;
            let p = f.access_chain(input, &[zero, at])?;
            read.push(f.load(p, MemAccess::default())?);
        }
        let value = body(f, t, tables, &read)?;
        let p = f.access_chain(output, &[zero, i])?;
        f.store(p, value, MemAccess::default())
    };
    build(&mut f).expect("body");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// Run `m` (a [`word_kernel`]) over `rows`, each its words, and return each invocation's word.
fn run_words(m: &TypedModule, rows: &[Vec<u32>]) -> Vec<u32> {
    let input = rows
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(SAMPLES_IN),
                address: None,
                bytes: input,
            },
            Buffer {
                slot: Some(RESOLVED_OUT),
                address: None,
                bytes: vec![0; rows.len() * 4],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let cfg = Config {
        order: Order::Forward,
        workgroups: [rows.len() as u32, 1, 1],
        ..Config::default()
    };
    let out = run(m, &cfg, res).expect("run").buffers.remove(1);
    out.chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().expect("word")))
        .collect()
}

/// The sRGB encode of the linear `x` in codes, `255 encode(x)` (IEC 61966-2-1), in `f64`.
fn srgb_encode_codes(x: f64) -> f64 {
    let encoded = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    255.0 * encoded
}

/// The midpoints the module's table holds: the `f32` midway between each code's linear value and
/// the next's.
fn srgb_midpoints() -> Vec<f32> {
    (0..255)
        .map(|k| ((srgb_decode(k) + srgb_decode(k + 1)) / 2.0) as f32)
        .collect()
}

/// The floats whose sRGB code a test asks: each code's linear value, and the lowest and highest
/// float of the code (the midpoint below it, and the float under the midpoint above it), all in
/// `[0, 1]`; each with its code.
fn srgb_floats() -> Vec<(f32, u32)> {
    let mid = srgb_midpoints();
    let mut out = Vec::new();
    for c in 0..256u32 {
        out.push((srgb_decode(c) as f32, c));
        if c > 0 {
            out.push((mid[c as usize - 1], c));
        }
        if c < 255 {
            out.push((f32::from_bits(mid[c as usize].to_bits() - 1), c));
        }
    }
    out
}

/// The estimate is within 0.12 codes of the sRGB encode over every float a test asks and every
/// 12-bit value's linear value, on the interpreter (its square root rounded correctly).
#[test]
fn the_srgb_estimate_is_within_0_12_codes_of_the_encode() {
    let m = word_kernel(1, &|f, t, _, w| {
        let x = f.convert(ConvOp::Bitcast, w[0], t.f32_)?;
        let e = srgb_estimate(f, t, x)?;
        f.convert(ConvOp::Bitcast, e, t.u32_)
    });
    let mut xs: Vec<f32> = srgb_floats().into_iter().map(|(x, _)| x).collect();
    xs.extend((0..4096u32).map(|l| l as f32 * (1.0f32 / 4095.0)));
    xs.extend((0..=4096u32).map(|k| k as f32 / 4096.0));
    let rows: Vec<Vec<u32>> = xs.iter().map(|x| vec![x.to_bits()]).collect();
    for (x, got) in xs.iter().zip(run_words(&m, &rows)) {
        let want = srgb_encode_codes(f64::from(*x));
        let error = (f64::from(f32::from_bits(got)) - want).abs();
        assert!(
            error < 0.12,
            "{x}: estimate {} against {want}",
            f32::from_bits(got)
        );
    }
}

/// The offsets [`the_srgb_codes_are_exact_from_any_estimate_near_the_encode`] adds to the
/// estimate: an estimate this far off still rounds to within one code, so the correction is exact
/// for any device whose square roots err by far less.
const ESTIMATE_OFFSETS: [f32; 5] = [-0.75, -0.4, 0.0, 0.4, 0.75];

/// The corrected rank is exact from any estimate up to 0.75 codes off the module's own: an input
/// attachment's float of every code, and its lowest and highest float, give the code (the count of
/// midpoints it reaches); and every 12-bit value gives Metal's encode of it (the count of
/// thresholds it reaches), on the interpreter.
#[test]
fn the_srgb_codes_are_exact_from_any_estimate_near_the_encode() {
    let floats = word_kernel(2, &|f, t, tables, w| {
        let x = f.convert(ConvOp::Bitcast, w[0], t.f32_)?;
        let offset = f.convert(ConvOp::Bitcast, w[1], t.f32_)?;
        let e = srgb_estimate(f, t, x)?;
        let e = f.binary(BinOp::FAdd, e, offset)?;
        let midpoints = tables.midpoints.expect("midpoints");
        corrected_rank(f, t, midpoints, x, e, BinOp::FOrdGreaterThanEqual)
    });
    let cases: Vec<(f32, u32, f32)> = srgb_floats()
        .into_iter()
        .flat_map(|(x, c)| ESTIMATE_OFFSETS.map(|o| (x, c, o)))
        .collect();
    let mid = srgb_midpoints();
    let rows: Vec<Vec<u32>> = cases
        .iter()
        .map(|&(x, _, o)| vec![x.to_bits(), o.to_bits()])
        .collect();
    for (&(x, c, o), got) in cases.iter().zip(run_words(&floats, &rows)) {
        assert_eq!(mid.partition_point(|&m| m <= x) as u32, c, "{x}");
        assert_eq!(got, c, "the float {x} with the estimate {o:+} off");
    }
    let values = word_kernel(2, &|f, t, tables, w| {
        let offset = f.convert(ConvOp::Bitcast, w[1], t.f32_)?;
        let real = f.convert(ConvOp::ConvertUToF, w[0], t.f32_)?;
        let unit = float(f, t, 1.0 / 4095.0)?;
        let linear = f.binary(BinOp::FMul, real, unit)?;
        let e = srgb_estimate(f, t, linear)?;
        let e = f.binary(BinOp::FAdd, e, offset)?;
        corrected_rank(f, t, tables.thresholds, w[0], e, BinOp::UGreaterThanEqual)
    });
    let cases: Vec<(u32, f32)> = (0..4096u32)
        .flat_map(|l| ESTIMATE_OFFSETS.map(|o| (l, o)))
        .collect();
    let rows: Vec<Vec<u32>> = cases.iter().map(|&(l, o)| vec![l, o.to_bits()]).collect();
    for (&(l, o), got) in cases.iter().zip(run_words(&values, &rows)) {
        let want = SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| u32::from(t) <= l) as u32;
        assert_eq!(
            got, want,
            "the 12-bit value {l} with the estimate {o:+} off"
        );
    }
}
