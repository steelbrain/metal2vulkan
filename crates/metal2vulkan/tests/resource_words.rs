//! The `gpuResourceID` high words' test vectors (`docs/RESOURCE-ID.md`): `resource_words.tsv`
//! beside this file holds one Metal descriptor per line, as raw Metal values, and its word. A
//! guest's own computation of the words checks itself against the same file.
//!
//! `the_vectors_are_the_words` reads every line and computes its word through the public API, so
//! the file means what the specification says of its columns; `the_vectors_are_current` holds the
//! file equal to what `bless_resource_words` writes, so a change of any word is a change of this
//! file, which the guest's copy then fails on. Bless with
//! `cargo nextest run -p metal2vulkan --all-features --run-ignored ignored-only -E
//! 'test(bless_resource_words)'`.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::resource_word::TextureWord;
use metal2vulkan::metal::sampler::SamplerWord;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    SamplerMipFilter, SamplerReductionMode, TextureSwizzle, TextureSwizzleChannels,
};

const HEADER: &str = "\
# gpuResourceID high words (docs/RESOURCE-ID.md): one Metal descriptor per line, raw Metal values.
# texture\tpixelFormat\tswizzleRed\tswizzleGreen\tswizzleBlue\tswizzleAlpha\tword
# sampler\tminFilter\tmagFilter\tmipFilter\tsAddressMode\ttAddressMode\trAddressMode\tborderColor\tnormalizedCoordinates\tlodMinClamp\tlodMaxClamp\tlodAverage\tmaxAnisotropy\tcompareFunction\treductionMode\tword
# lodMinClamp and lodMaxClamp are a float's IEEE-754 bits; booleans are 0 or 1; words are hex.
";

fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/resource_words.tsv")
}

fn swizzle(channels: TextureSwizzleChannels) -> [TextureSwizzle; 4] {
    [channels.red, channels.green, channels.blue, channels.alpha]
}

/// The texture lines: every pixel format with a word under the default swizzle, then each
/// channel of a few formats of one to four channels, depth, integer and rotated, read from each
/// other source.
fn textures(out: &mut String) {
    let mut line = |pf: PixelFormat, channels: TextureSwizzleChannels| {
        let Some(TextureWord(word)) = TextureWord::of(pf, channels) else {
            return;
        };
        let s = swizzle(channels).map(|s| s.as_raw().to_string()).join("\t");
        let _ = writeln!(out, "texture\t{}\t{s}\t{word:#010x}", pf.as_raw());
    };
    for &pf in PixelFormat::ALL {
        line(pf, TextureSwizzleChannels::DEFAULT);
    }
    let formats = [
        PixelFormat::A8Unorm,
        PixelFormat::R8Unorm,
        PixelFormat::RG8Unorm,
        PixelFormat::RGBA8Unorm,
        PixelFormat::BGRA8Unorm,
        PixelFormat::R32Uint,
        PixelFormat::Depth32Float,
    ];
    for pf in formats {
        for channel in 0..4 {
            for &source in TextureSwizzle::ALL {
                let mut s = swizzle(TextureSwizzleChannels::DEFAULT);
                if s[channel] == source {
                    continue;
                }
                s[channel] = source;
                line(pf, TextureSwizzleChannels::new(s[0], s[1], s[2], s[3]));
            }
        }
    }
}

fn sampler_line(out: &mut String, d: &SamplerDescriptor) {
    let SamplerWord(word) = SamplerWord::of(d);
    let fields = [
        d.min_filter.as_raw(),
        d.mag_filter.as_raw(),
        d.mip_filter.as_raw(),
        d.s_address_mode.as_raw(),
        d.t_address_mode.as_raw(),
        d.r_address_mode.as_raw(),
        d.border_color.as_raw(),
        u32::from(d.normalized_coordinates),
    ];
    let fields = fields.map(|f| f.to_string()).join("\t");
    let _ = writeln!(
        out,
        "sampler\t{fields}\t{:#010x}\t{:#010x}\t{}\t{}\t{}\t{}\t{word:#010x}",
        d.lod_min_clamp.to_bits(),
        d.lod_max_clamp.to_bits(),
        u32::from(d.lod_average),
        d.max_anisotropy,
        d.compare_function.as_raw(),
        d.reduction_mode.as_raw(),
    );
}

/// The sampler lines: the default descriptor, each field through each of its values with the
/// others at their defaults, the level clamps around the points a nearest mip filter's level
/// turns at, and one descriptor with every field away from its default.
fn samplers(out: &mut String) {
    let base = SamplerDescriptor::new();
    let mut each = |set: &dyn Fn(&mut SamplerDescriptor)| {
        let mut d = base;
        set(&mut d);
        sampler_line(out, &d);
    };
    each(&|_| {});
    for &f in SamplerMinMagFilter::ALL {
        each(&|d| d.min_filter = f);
        each(&|d| d.mag_filter = f);
    }
    for &f in SamplerMipFilter::ALL {
        each(&|d| d.mip_filter = f);
    }
    for &m in SamplerAddressMode::ALL {
        each(&|d| d.s_address_mode = m);
        each(&|d| d.t_address_mode = m);
        each(&|d| d.r_address_mode = m);
    }
    for &b in SamplerBorderColor::ALL {
        each(&|d| d.border_color = b);
    }
    each(&|d| d.normalized_coordinates = false);
    let mins = [
        0.0,
        0.25,
        0.5,
        0.515625,
        0.53125,
        1.0,
        1.4,
        1.6,
        2.75,
        14.5,
        14.6,
        15.0,
        100.0,
        f32::MAX,
    ];
    for c in mins {
        each(&|d| d.lod_min_clamp = c);
    }
    let maxes = [0.0, 0.5, 0.515625, 0.53125, 3.9, 14.0, 14.6, 15.0, 1000.0];
    for c in maxes {
        each(&|d| d.lod_max_clamp = c);
    }
    each(&|d| d.lod_average = true);
    for a in [2, 3, 16] {
        each(&|d| d.max_anisotropy = a);
    }
    for &f in CompareFunction::ALL {
        each(&|d| d.compare_function = f);
    }
    for &r in SamplerReductionMode::ALL {
        each(&|d| d.reduction_mode = r);
    }
    each(&|d| {
        d.min_filter = SamplerMinMagFilter::Linear;
        d.mag_filter = SamplerMinMagFilter::Cubic;
        d.mip_filter = SamplerMipFilter::Linear;
        d.s_address_mode = SamplerAddressMode::Repeat;
        d.t_address_mode = SamplerAddressMode::MirrorRepeat;
        d.r_address_mode = SamplerAddressMode::ClampToBorderColor;
        d.border_color = SamplerBorderColor::OpaqueWhite;
        d.normalized_coordinates = false;
        d.lod_min_clamp = 2.75;
        d.lod_max_clamp = 9.25;
        d.lod_average = true;
        d.max_anisotropy = 8;
        d.compare_function = CompareFunction::GreaterEqual;
        d.reduction_mode = SamplerReductionMode::Maximum;
    });
}

/// The file: the header, then the texture and sampler lines, each line once.
fn generate() -> String {
    let mut lines = String::new();
    textures(&mut lines);
    samplers(&mut lines);
    let mut seen = BTreeSet::new();
    let mut out = String::from(HEADER);
    for line in lines.lines().filter(|l| seen.insert(*l)) {
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn raw<T: Copy>(all: &[T], as_raw: impl Fn(T) -> u32, field: &str) -> T {
    let v: u32 = field.parse().unwrap_or_else(|_| panic!("field {field}"));
    all.iter()
        .copied()
        .find(|&t| as_raw(t) == v)
        .unwrap_or_else(|| panic!("no value {v}"))
}

fn hex(field: &str) -> u32 {
    let digits = field
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("hex {field}"));
    u32::from_str_radix(digits, 16).unwrap_or_else(|_| panic!("hex {field}"))
}

fn flag(field: &str) -> bool {
    match field {
        "0" => false,
        "1" => true,
        _ => panic!("boolean {field}"),
    }
}

/// The word a line's descriptor has, computed through the public API from its columns alone.
fn word_of(fields: &[&str]) -> u32 {
    match fields {
        ["texture", pf, r, g, b, a] => {
            let pf = raw(PixelFormat::ALL, PixelFormat::as_raw, pf);
            let s = |f: &str| raw(TextureSwizzle::ALL, TextureSwizzle::as_raw, f);
            let channels = TextureSwizzleChannels::new(s(r), s(g), s(b), s(a));
            TextureWord::of(pf, channels).expect("a word").0
        }
        ["sampler", min, mag, mip, sa, ta, ra, border, normalized, lo, hi, average, aniso, compare, reduction] =>
        {
            let mut d = SamplerDescriptor::new();
            let filter = |f: &str| raw(SamplerMinMagFilter::ALL, SamplerMinMagFilter::as_raw, f);
            let mode = |f: &str| raw(SamplerAddressMode::ALL, SamplerAddressMode::as_raw, f);
            d.min_filter = filter(min);
            d.mag_filter = filter(mag);
            d.mip_filter = raw(SamplerMipFilter::ALL, SamplerMipFilter::as_raw, mip);
            d.s_address_mode = mode(sa);
            d.t_address_mode = mode(ta);
            d.r_address_mode = mode(ra);
            d.border_color = raw(SamplerBorderColor::ALL, SamplerBorderColor::as_raw, border);
            d.normalized_coordinates = flag(normalized);
            d.lod_min_clamp = f32::from_bits(hex(lo));
            d.lod_max_clamp = f32::from_bits(hex(hi));
            d.lod_average = flag(average);
            d.max_anisotropy = aniso.parse().expect("maxAnisotropy");
            d.compare_function = raw(CompareFunction::ALL, CompareFunction::as_raw, compare);
            d.reduction_mode = raw(
                SamplerReductionMode::ALL,
                SamplerReductionMode::as_raw,
                reduction,
            );
            SamplerWord::of(&d).0
        }
        _ => panic!("unknown line {fields:?}"),
    }
}

#[test]
fn the_vectors_are_the_words() {
    let text = std::fs::read_to_string(path()).expect("resource_words.tsv");
    let mut lines = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split('\t').collect();
        let Some((word, descriptor)) = fields.split_last() else {
            panic!("empty line");
        };
        assert_eq!(word_of(descriptor), hex(word), "{line}");
        lines += 1;
    }
    assert!(lines > 0, "no vectors");
}

#[test]
fn the_vectors_are_current() {
    let text = std::fs::read_to_string(path()).expect("resource_words.tsv");
    assert!(
        text == generate(),
        "resource_words.tsv is not what the words are now: bless it (this file's docs) and tell \
         the guest's port"
    );
}

#[test]
#[ignore = "writes tests/resource_words.tsv"]
fn bless_resource_words() {
    std::fs::write(path(), generate()).expect("write resource_words.tsv");
}
