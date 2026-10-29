//! The extended-range resolve modules (the module docs' extended-range forms): the forms Metal
//! multisamples, valid on every committed profile and needing nothing of a device, and on the CPU
//! interpreter a linear form's colors and alpha by their rules, and a pixel of alike samples its
//! code (the `_sRGB` forms' transfer is the device's `Pow`, which the interpreter does not model).

use super::*;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

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

/// Every form Metal makes multisampled textures of.
const FORMS: [ExtendedRange; 5] = [
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: true },
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgra10 { srgb: true },
    ExtendedRange::Rgb10A2Srgb,
];

/// The forms Metal makes no multisampled texture of (probe `xrms`).
const UNMADE: [ExtendedRange; 9] = [
    ExtendedRange::Planes { srgb: false },
    ExtendedRange::Planes { srgb: true },
    ExtendedRange::R10X6,
    ExtendedRange::Rg10X12,
    ExtendedRange::R10X6Srgb,
    ExtendedRange::Rg10X12Srgb,
    ExtendedRange::R16Bfloat,
    ExtendedRange::Rg16Bfloat,
    ExtendedRange::Rgba16Bfloat,
];

/// The sources a test reads `form` from: the input attachment, and the sample planes of its raw
/// format's bits copy.
fn sources(form: ExtendedRange) -> [ResolveSource; 2] {
    let copy = PlanesCopy::for_planes(form.raw_format()).expect("a bits copy");
    [ResolveSource::InputAttachment, ResolveSource::Planes(copy)]
}

#[test]
fn the_forms_are_the_ones_metal_multisamples() {
    for form in FORMS {
        let rules = ColorResolveFormat::extended_range(form).expect("a form Metal multisamples");
        assert_eq!(rules.format(), form.raw_format(), "{form:?}");
        assert_eq!(rules.extended(), Some(form));
        assert_eq!(rules.channels(), [None; 4]);
        assert_eq!(rules.raw_format(), rules.format());
        assert_eq!(
            rules.exact(),
            form == ExtendedRange::Bgr10 { srgb: false },
            "{form:?}"
        );
    }
    for form in UNMADE {
        assert_eq!(ColorResolveFormat::extended_range(form), None, "{form:?}");
    }
    assert!(ColorResolveFormat::of(Format::R8G8B8A8_UNORM)
        .expect("closed")
        .exact());
}

/// Each form's fragment stage needs nothing of a device beyond Vulkan 1.2's shaders and is valid on
/// every committed profile, at 2, 4 and 8 samples, from each source.
#[test]
fn the_modules_are_valid_and_need_nothing_of_a_device() {
    let profiles = committed_profiles().expect("profiles");
    for form in FORMS {
        let rules = ColorResolveFormat::extended_range(form).expect("form");
        for source in sources(form) {
            for samples in [2, 4, 8] {
                let m = fragment_typed_module(rules, samples, source).expect("module");
                let r = crate::spirv::serialize::serialize(m.clone())
                    .expect("serialize")
                    .requirements;
                assert!(
                    r.features.is_empty(),
                    "{form:?} x{samples} {source:?}: {r:?}"
                );
                for (_, profile) in &profiles {
                    let _ = assert_valid(m.clone(), profile);
                }
            }
        }
    }
    // Planes not of a bits copy hold no codes.
    let rules = ColorResolveFormat::extended_range(FORMS[0]).expect("form");
    assert!(fragment_typed_module(rules, 4, ResolveSource::Planes(PlanesCopy::Alpha8)).is_err());
}

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

/// A kernel running `form`'s resolve: invocation `i` reads its `samples` raw texels as four words
/// each from `SAMPLES_IN` and writes the resolved raw texel's four words to `RESOLVED_OUT`.
fn kernel(form: ExtendedRange, samples: u32) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b).expect("types");
    let words = super::super::visibility_resolve::words_buffer;
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
        let mut raws = Vec::new();
        for s in 0..samples {
            let mut lanes = Vec::new();
            for w in 0..4 {
                let k = uint(f, t, s * 4 + w)?;
                let at = f.binary(BinOp::IAdd, base, k)?;
                let p = f.access_chain(input, &[zero, at])?;
                lanes.push(f.load(p, MemAccess::default())?);
            }
            raws.push(f.composite_construct(uvec4, &lanes)?);
        }
        let out = resolve_extended(f, t, form, &raws)?;
        let four = uint(f, t, 4)?;
        let out_base = f.binary(BinOp::IMul, i, four)?;
        for lane in 0..4 {
            let word = f.composite_extract(out, &[lane])?;
            let c = uint(f, t, lane)?;
            let at = f.binary(BinOp::IAdd, out_base, c)?;
            let p = f.access_chain(output, &[zero, at])?;
            f.store(p, word, MemAccess::default())?;
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

/// The raw texel of `form` (as an input attachment of its raw format reads it) of `codes`, red
/// first.
fn raw(form: ExtendedRange, [r, g, b, a]: [u32; 4]) -> [u32; 4] {
    match form {
        ExtendedRange::Bgr10 { .. } => [0xC000_0000 | r << 20 | g << 10 | b, 0, 0, 1],
        ExtendedRange::Bgra10 { .. } => [b << 6, g << 6, r << 6, a << 6],
        _ => [r | g << 10 | b << 20 | a << 30, 0, 0, 1],
    }
}

/// The codes (red first) of `form`'s raw texel `t` as the resolve writes it, and whether its
/// unused bits are as Metal writes them (`BGR10_XR`'s top two ones, `BGRA10_XR`'s low six zeros).
fn unpacked(form: ExtendedRange, t: [u32; 4]) -> ([u32; 4], bool) {
    match form {
        ExtendedRange::Bgr10 { .. } => (
            [(t[0] >> 20) & 1023, (t[0] >> 10) & 1023, t[0] & 1023, 894],
            t[0] >> 30 == 3,
        ),
        ExtendedRange::Bgra10 { .. } => (
            [t[2] >> 6, t[1] >> 6, t[0] >> 6, t[3] >> 6],
            t.iter().all(|&w| w & 63 == 0 && w < 1 << 16),
        ),
        _ => (
            [
                t[0] & 1023,
                (t[0] >> 10) & 1023,
                (t[0] >> 20) & 1023,
                t[0] >> 30,
            ],
            true,
        ),
    }
}

/// A random code of `form`'s channel `ch`, near `around` where it is given.
fn code(form: ExtendedRange, ch: usize, r: &mut SplitMix, around: Option<u32>) -> u32 {
    let (lo, hi) = match (form, ch) {
        (ExtendedRange::Bgr10 { .. }, 3) => (894, 894),
        (ExtendedRange::Bgra10 { .. }, 3) => (384, 894),
        (ExtendedRange::Rgb10A2Srgb, 3) => (0, 3),
        _ => (0, 1023),
    };
    let x = match around {
        Some(c) => (c + (r.next() % 5) as u32).saturating_sub(2),
        None => lo + (r.next() % u64::from(hi - lo + 1)) as u32,
    };
    x.clamp(lo, hi)
}

/// Run `form`'s resolve at `samples` over `pixels` (each sample's codes, red first).
fn resolved(form: ExtendedRange, samples: usize, pixels: &[Vec<[u32; 4]>]) -> Vec<[u32; 4]> {
    let input: Vec<u8> = pixels
        .iter()
        .flatten()
        .flat_map(|&c| raw(form, c))
        .flat_map(u32::to_le_bytes)
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
                bytes: vec![0; pixels.len() * 16],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let cfg = Config {
        order: Order::Forward,
        workgroups: [pixels.len() as u32, 1, 1],
        ..Config::default()
    };
    let out = run(&kernel(form, samples as u32), &cfg, res)
        .expect("run")
        .buffers
        .remove(1);
    out.as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let (words, _) = t.as_chunks::<4>();
            std::array::from_fn(|w| u32::from_le_bytes(words[w]))
        })
        .collect()
}

/// A linear form's color is the samples' codes' average rounded half up (Metal's rule); its
/// alpha its near-equivalent (`BGRA10_XR`'s a tie down); and a pixel of alike samples is their
/// code, with the unused bits as Metal writes them. The `_sRGB` forms' (and
/// `RGB10A2Unorm_sRGB`'s) decode and encode take the transfer by the device's `Pow`, which the
/// interpreter does not model: their colors, held within the samples' codes, are the device's.
#[test]
fn the_resolves_follow_their_rules() {
    let mut r = SplitMix(0x5eed_0e7e);
    let linear = FORMS
        .into_iter()
        .filter(|f| !f.srgb() && *f != ExtendedRange::Rgb10A2Srgb);
    for form in linear {
        for samples in [2usize, 4, 8] {
            let mut pixels: Vec<Vec<[u32; 4]>> = Vec::new();
            for i in 0..384 {
                let alike = i % 8 == 0;
                let base: [u32; 4] = std::array::from_fn(|ch| code(form, ch, &mut r, None));
                let mut pixel = Vec::new();
                for _ in 0..samples {
                    pixel.push(match alike {
                        true => base,
                        false => std::array::from_fn(|ch| {
                            let near = i % 2 == 1;
                            code(form, ch, &mut r, near.then_some(base[ch]))
                        }),
                    });
                }
                pixels.push(pixel);
            }
            let n = samples as u32;
            for (pixel, got) in pixels.iter().zip(resolved(form, samples, &pixels)) {
                let (codes, bits) = unpacked(form, got);
                assert!(bits, "{form:?} x{samples}: {got:x?}");
                for ch in 0..4 {
                    let each: Vec<u32> = pixel.iter().map(|s| s[ch]).collect();
                    let sum: u32 = each.iter().sum();
                    let tie_down = matches!(form, ExtendedRange::Bgra10 { .. }) && ch == 3;
                    let rule = match tie_down {
                        true => (sum + n / 2 - 1) / n,
                        false => (sum + n / 2) / n,
                    };
                    let srgb_color = ch < 3 && (form.srgb() || form == ExtendedRange::Rgb10A2Srgb);
                    let alike = pixel.iter().all(|s| s == &pixel[0]);
                    let lo = *each.iter().min().unwrap();
                    let hi = *each.iter().max().unwrap();
                    let ok = match (alike, srgb_color) {
                        (true, _) => codes[ch] == each[0],
                        (false, true) => (lo..=hi).contains(&codes[ch]),
                        (false, false) => codes[ch] == rule,
                    };
                    assert!(
                        ok,
                        "{form:?} x{samples} channel {ch}: samples {each:?} resolved {}",
                        codes[ch]
                    );
                }
            }
        }
    }
}
