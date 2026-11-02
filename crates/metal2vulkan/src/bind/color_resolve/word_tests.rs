//! The word resolve modules (the module docs' word forms): valid on every committed profile and
//! needing nothing of a device, and on the CPU interpreter `RGB9E5Float`'s resolve Metal's at every
//! pixel our probe `msaa_word_formats.m` measured and by its rule over random words, and a packed
//! 16-bit word's each field's codes averaged, a tie rounded down (a one-bit field's up).

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

/// The words a word resolve resolves: the packed 16-bit formats' and `RGB9E5Float`'s.
fn words() -> Vec<WordAttachment> {
    [
        crate::metal::PixelFormat::A1BGR5Unorm,
        crate::metal::PixelFormat::ABGR4Unorm,
        crate::metal::PixelFormat::B5G6R5Unorm,
        crate::metal::PixelFormat::BGR5A1Unorm,
        crate::metal::PixelFormat::BGRA4Unorm,
        crate::metal::PixelFormat::RGB9E5Float,
    ]
    .into_iter()
    .map(|pf| WordAttachment::layout(pf).expect("a word"))
    .collect()
}

#[test]
fn a_word_resolve_holds_the_word_and_an_srgb_word_has_none() {
    for word in words() {
        let rules = ColorResolveFormat::word(word).expect("a word resolve");
        assert_eq!(rules.format(), word.word_format(), "{word:?}");
        assert_eq!(rules.held_word(), Some(word));
        assert_eq!(rules.channels(), [None; 4]);
        assert_eq!(rules.raw_format(), rules.format());
        assert_eq!(rules.written_format(), rules.format());
        assert_eq!(rules.exact(), word == WordAttachment::Rgb9e5, "{word:?}");
    }
    let srgb = WordAttachment::layout(crate::metal::PixelFormat::R8UnormSRGB).expect("a word");
    assert_eq!(ColorResolveFormat::word(srgb), None);
}

/// Each word's fragment stage needs nothing of a device beyond Vulkan 1.2's shaders and is valid
/// on every committed profile, at 2, 4 and 8 samples, from the input attachment and the sample
/// planes of the word's bits copy; not from the raw twin's view, nor planes of another copy.
#[test]
fn the_word_modules_are_valid_and_need_nothing_of_a_device() {
    let profiles = committed_profiles().expect("profiles");
    for word in words() {
        let rules = ColorResolveFormat::word(word).expect("a word resolve");
        let copy = PlanesCopy::for_planes(word.word_format()).expect("a bits copy");
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            for samples in [2, 4, 8] {
                let m = fragment_typed_module(rules, samples, source).expect("module");
                let r = crate::spirv::serialize::serialize(m.clone())
                    .expect("serialize")
                    .requirements;
                assert!(r.features.is_empty(), "{word:?} x{samples}: {r:?}");
                for (_, profile) in &profiles {
                    let _ = assert_valid(m.clone(), profile);
                }
            }
        }
        assert!(fragment_typed_module(rules, 4, ResolveSource::RawInputAttachment).is_err());
        assert!(
            fragment_typed_module(rules, 4, ResolveSource::Planes(PlanesCopy::Alpha8)).is_err()
        );
    }
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

/// A kernel running `word`'s resolve: invocation `i` reads its `samples` words from `SAMPLES_IN`
/// and writes the resolved word to `RESOLVED_OUT`.
fn kernel(word: WordAttachment, samples: u32) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b).expect("types");
    let buffer = super::super::visibility_resolve::words_buffer;
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
        let per = uint(f, t, samples)?;
        let base = f.binary(BinOp::IMul, i, per)?;
        let mut words = Vec::new();
        for s in 0..samples {
            let k = uint(f, t, s)?;
            let at = f.binary(BinOp::IAdd, base, k)?;
            let p = f.access_chain(input, &[zero, at])?;
            words.push(f.load(p, MemAccess::default())?);
        }
        let out = resolve_word(f, t, word, &words)?;
        let p = f.access_chain(output, &[zero, i])?;
        f.store(p, out, MemAccess::default())?;
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

/// Run `word`'s resolve at `samples` over `pixels` (each sample's word).
fn resolved(word: WordAttachment, samples: usize, pixels: &[Vec<u32>]) -> Vec<u32> {
    let input: Vec<u8> = pixels
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
                bytes: vec![0; pixels.len() * 4],
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
    let out = run(&kernel(word, samples as u32), &cfg, res)
        .expect("run")
        .buffers
        .remove(1);
    out.as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Metal's resolve of `RGB9E5Float` words (the module docs): the channels decoded, their `f32`
/// sum in sample order times `1 / n`, rounded to the nearest `half` (a tie to even), and the
/// word of those values with its mantissas rounded half up, the exponent raised where one
/// reaches 512.
fn shared_exponent_model(words: &[u32]) -> u32 {
    let decode = |w: u32, c: u32| ((w >> (9 * c)) & 511) as f32 * 2f32.powi((w >> 27) as i32 - 24);
    let rgb: [f64; 3] = std::array::from_fn(|c| {
        let sum = words.iter().fold(0f32, |s, &w| s + decode(w, c as u32));
        let mean = f64::from(sum * (1.0 / words.len() as f32));
        let grid = 2f64.powi(mean.log2().floor().max(-14.0) as i32 - 10);
        match mean {
            0.0 => 0.0,
            _ => (mean / grid).round_ties_even() * grid,
        }
    });
    let largest = rgb.iter().copied().fold(0.0, f64::max);
    let mut e = match largest {
        0.0 => 0,
        _ => (largest.log2().floor().max(-16.0) as i32 + 16) as u32,
    };
    let mantissas = |e: u32| rgb.map(|x| (x * 2f64.powi(24 - e as i32) + 0.5).floor() as u32);
    if mantissas(e).iter().any(|&m| m > 511) {
        e += 1;
    }
    let [r, g, b] = mantissas(e).map(|m| m.min(511));
    e << 27 | r | g << 9 | b << 18
}

/// Metal's `RGB9E5Float` resolve as our probe `msaa_word_formats.m` read it on the M3 Max under
/// macOS 27.0.1: each pixel's samples' words and the resolved word, from the pass that drew them
/// and from one that loaded them alike, among them the pixels a mean of the codes with their
/// mantissas truncated, or rounded to nearest, misses.
const MEASURED: &[(&[u32], u32)] = &[
    (&[0x78000118, 0xa81c01fa], 0xa01c01ff),
    (&[0x7b360800, 0xc7d40000], 0xbfd80200),
    (&[0xbf840203, 0x6e1c5612], 0xb7880203),
    (&[0xa8000dfb, 0xb8345bd8], 0xb81c2f2c),
    (&[0x70c06389, 0xad340000], 0xa5380003),
    (&[0x80675a7e, 0xb4681a00], 0xac6c2802),
    (&[0x800f5c00, 0xa9275c00], 0xa1277800),
    (&[0x883bce00, 0xb4681a00], 0xac6c3800),
    (&[0xa7700002, 0x85980410], 0x9fcc0003),
    (&[0x880001b3, 0x880001b3], 0x880001b3),
    (&[0xc0000192, 0x8f460c00], 0xb8100592),
    (&[0x9f30f804, 0xfffc0000], 0xf7fc0000),
    (
        &[0xa6b01800, 0x9f4c0000, 0x98032218, 0xb4681a00],
        0xa7008403,
    ),
    (
        &[0x942c0000, 0xc423e609, 0x8f681800, 0xb4681a00],
        0xb55fec09,
    ),
    (
        &[0x880401a8, 0x9db00544, 0xc50c0800, 0x8803c603],
        0xb53c100d,
    ),
    (
        &[0xa8016566, 0x980c014d, 0xf80001ff, 0xb01f3a00],
        0xf0000100,
    ),
    (
        &[0xa7700002, 0xa7700002, 0x85980410, 0x85980410],
        0x9fcc0003,
    ),
    (
        &[0x98000122, 0xb0552df9, 0x9dd41c3c, 0xb4681a00],
        0xaabca513,
    ),
    (
        &[0x58e829fb, 0x58e829fb, 0x70040136, 0x70040136],
        0x68200776,
    ),
    (
        &[0x48543d6c, 0xa1ca6200, 0x6fa31ea5, 0xb4681a00],
        0xa4e0b400,
    ),
    (
        &[0xa8021600, 0x680401d3, 0xc00aa01b, 0xc0048744],
        0xb00f6b5f,
    ),
    (
        &[0xb0000197, 0x8cbc0005, 0xa8000195, 0xbba81fb4],
        0xb1dc1173,
    ),
    (
        &[0xc00001e0, 0xa803620b, 0x886ed005, 0xb4681a00],
        0xb11c79e2,
    ),
    (
        &[0x90fb2e7d, 0x5caee7b9, 0xa465015b, 0x58000506],
        0x94a5cf7c,
    ),
];

/// `RGB9E5Float`'s resolve is Metal's at every pixel measured, and its rule's over random words of
/// every exponent at 2, 4 and 8 samples.
#[test]
fn the_shared_exponent_resolve_is_metals() {
    for samples in [2usize, 4] {
        let cases: Vec<&(&[u32], u32)> = MEASURED
            .iter()
            .filter(|(w, _)| w.len() == samples)
            .collect();
        let pixels: Vec<Vec<u32>> = cases.iter().map(|(w, _)| w.to_vec()).collect();
        let got = resolved(WordAttachment::Rgb9e5, samples, &pixels);
        for ((words, metal), got) in cases.iter().zip(got) {
            assert_eq!(got, *metal, "{words:08x?}: {got:08x}, Metal {metal:08x}");
            assert_eq!(shared_exponent_model(words), *metal, "{words:08x?}");
        }
    }
    let mut r = SplitMix(0x9e5_0e7e);
    for samples in [2usize, 4, 8] {
        let pixels: Vec<Vec<u32>> = (0..512)
            .map(|i| {
                (0..samples)
                    .map(|_| {
                        // Mostly words Metal stores (a largest mantissa of at least 256 or a zero
                        // exponent), every exponent; one pixel in eight any word.
                        let e = (r.next() % 32) as u32;
                        let mut m: [u32; 3] = std::array::from_fn(|_| (r.next() % 512) as u32);
                        if i % 8 != 0 && e > 0 {
                            m[(r.next() % 3) as usize] |= 256;
                        }
                        e << 27 | m[0] | m[1] << 9 | m[2] << 18
                    })
                    .collect()
            })
            .collect();
        for (pixel, got) in pixels
            .iter()
            .zip(resolved(WordAttachment::Rgb9e5, samples, &pixels))
        {
            assert_eq!(
                got,
                shared_exponent_model(pixel),
                "x{samples} {pixel:08x?}: {got:08x}"
            );
        }
    }
}

/// A packed 16-bit word's resolve is each field's codes averaged, a tie rounded down and a one-bit
/// field's up, the word's other bits zero; a pixel of one word resolves to it.
#[test]
fn a_packed_word_resolves_each_field_by_its_codes() {
    let mut r = SplitMix(0x16b1_7e57);
    for word in words()
        .into_iter()
        .filter(|w| matches!(w, WordAttachment::Packed16 { .. }))
    {
        for samples in [2usize, 4, 8] {
            let n = samples as u32;
            let pixels: Vec<Vec<u32>> = (0..512)
                .map(|i| {
                    let one = (r.next() & 0xffff) as u32;
                    (0..samples)
                        .map(|_| match i % 8 {
                            0 => one,
                            _ => (r.next() & 0xffff) as u32,
                        })
                        .collect()
                })
                .collect();
            for (pixel, got) in pixels.iter().zip(resolved(word, samples, &pixels)) {
                let mut want = 0;
                for field in word.fields().into_iter().flatten() {
                    let mask = (1u32 << field.bits) - 1;
                    let sum: u32 = pixel.iter().map(|w| (w >> field.shift) & mask).sum();
                    let bias = if field.bits == 1 { n / 2 } else { n / 2 - 1 };
                    want |= ((sum + bias) / n) << field.shift;
                }
                assert_eq!(got, want, "{word:?} x{samples} {pixel:04x?}: {got:04x}");
                let fields = word.pack([u32::MAX; 4]);
                if pixel.iter().all(|&w| w == pixel[0]) {
                    assert_eq!(got, pixel[0] & fields, "{word:?} x{samples}");
                }
            }
        }
    }
}
