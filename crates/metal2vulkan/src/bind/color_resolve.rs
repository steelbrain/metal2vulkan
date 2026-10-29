//! The crate-owned multisample color resolves: the two no device's `AVERAGE` resolve writes.
//!
//! A device resolves a color attachment in its pass by Vulkan's `AVERAGE`, into a resolve
//! attachment of the color attachment's own format
//! (`VUID-VkSubpassDescription-pResolveAttachments-00850`), over the values the format holds.
//! These modules draw the two resolves that cannot be:
//!
//! - **into the attachment's format's sRGB twin** ([`ColorResolveFormat::of`]: the 8-bit unorm
//!   formats and their sRGB twins), which Metal resolves in the resolve texture's encoding. Metal's
//!   rule is one per channel, which probes closed on an Apple M3 Max and an M4 (macOS 26.7,
//!   byte-identical): every pair of 8-bit codes at 2 samples, and hundreds of thousands of 2- and
//!   4-sample texels per format, each matched by the rule ([`ResolveRule`]):
//!   - **unorm** (8-bit): `(sum + n / 2) / n` over the integer codes;
//!   - **sRGB** (8-bit): each code's 12-bit linear value ([`SRGB_RESOLVE_LINEAR`]), their average
//!     `(sum + n / 2) / n`, and Metal's encode of that 12-bit value ([`SRGB_RESOLVE_THRESHOLDS`]);
//!     the alpha of an sRGB format is unorm.
//!
//!   Metal has no texture of more than 4 samples on Apple silicon: at 8 and more the modules apply
//!   the same rules, which no Metal answer measures.
//!
//!   **The device's sRGB conversions** ([`ColorResolveFormat::device_srgb`]): where the device has
//!   the sRGB format, the sRGB rule is the device's own decode and encode, not Metal's 12-bit
//!   tables. The modules read the attachment through a view in the sRGB format
//!   ([`ResolveSource::InputAttachment`]), whose read decodes each sample's channel, average the
//!   decoded values (their `f32` sum times `1 / n`, exact at a power of two), and write the
//!   average through a view in the sRGB format, whose store encodes it; the alpha keeps the unorm
//!   rule. A near-equivalent of Metal's rule: with correctly rounded conversions, 1,758 of the
//!   65,536 pairs of codes at 2 samples resolve one code from Metal's code, and none further (the
//!   tests' model); the device's own conversions decide the rest. The crate's tables serve a
//!   device that lacks the sRGB format, through the raw twin's views.
//! - **of an extended-range attachment** ([`ColorResolveFormat::extended_range`], below), held in
//!   its raw format's codes, which no device's resolve averages.
//!
//! **The modules** ([`color_resolve_modules`]): a `Vertex` entry `main` that draws one triangle
//! over the whole viewport from `VertexIndex` 0 to 2 (the sample planes merge's), and a `Fragment`
//! entry `main` that reads each of a pixel's samples from its [`ResolveSource`] at
//! [`ColorResolveModules::SOURCE_BINDING`] of set [`ColorResolveModules::SET`], resolves each
//! channel by its rule, and writes the result to the color output at `Location` 0 as the float that
//! the resolve texture's format stores as the resolved code: `code / 255` for unorm, the code's
//! linear value for sRGB (the attachment encodes it), or `code / 255` where the result is written
//! through the raw twin's view ([`ColorResolveFormat::written_raw`]). A pixel whose samples all
//! hold the same bits, as every pixel inside a primitive does where one invocation writes each
//! covered sample, skips an sRGB format's average: there both rules give back the sample's code,
//! so the module writes that code's float. A format of unorm channels alone averages without
//! comparing, as its average costs about what comparing the samples does: on a Radeon 780M,
//! comparing first took 8% off an `RGBA8Unorm` 1080p 4x resolve's cost over a plain store where
//! every pixel's samples matched, and added 14% where one pixel in 16 straddled an edge.
//!
//! - [`ResolveSource::InputAttachment`]: the multisampled attachment as the input attachment of
//!   index 0 (`InputAttachmentIndex 0`), in its own format, read at each sample. A unorm channel's
//!   code is its float times 255, rounded; an sRGB channel's the code whose linear value is
//!   nearest the float.
//! - [`ResolveSource::RawInputAttachment`]: the same attachment viewed in the format's raw twin
//!   ([`ColorResolveFormat::raw_format`]: an 8-bit sRGB format's UNORM one), so an sRGB channel's
//!   float is its code over 255, whose code is read as a unorm channel's, with no table lookup.
//!   Another format's raw twin is itself, and the source reads it as
//!   [`ResolveSource::InputAttachment`] does.
//! - [`ResolveSource::Planes`]: an extended-range attachment's sample planes (`bind::PlanesCopy`),
//!   a sampled 2D-array view of every layer in the copy format, layer `s` holding sample `s`.
//!   The twin's formats are never read from planes: a twin resolve reads the attachment.
//!
//! Draw it into a render pass whose one color attachment is the resolve texture's single-sampled
//! level and layer in [`ColorResolveFormat::written_format`] (its own format, or its raw twin
//! where the result is written raw), over the render area, one sample, without blending; for
//! [`ResolveSource::InputAttachment`] the pass's subpass also reads the multisampled attachment as
//! its input attachment 0, and for [`ResolveSource::RawInputAttachment`] a view of it in its raw
//! twin.
//!
//! **Extended-range forms** ([`ColorResolveFormat::extended_range`]): Metal makes multisampled
//! textures of `BGR10_XR`, `BGRA10_XR`, their `_sRGB` twins and `RGB10A2Unorm_sRGB`, and of none of
//! Apple's other private forms (our probe `xrms`, the M3 Max under macOS 27.0.1). Such an attachment
//! is held in its raw format's codes (`metal::format::ExtendedRange`), so these modules read the
//! source in that unsigned format (sample planes of a bits copy, a two-word texel split into
//! `BGRA10_XR`'s four fields) and write the resolved raw texel, a `uvec4`, to `Location` 0 of a
//! resolve attachment in that format. Metal's resolve, measured by our probe `xr` on the M3 Max
//! under macOS 27.0.1 (4096 texels per form at 2 and at 4 samples, each sample its own value): a
//! linear form's color channel is the average of the samples' codes rounded half up, every texel,
//! which the modules apply. Metal's other channels average values the stored codes do not keep,
//! so the modules write a near-equivalent where a pixel's samples differ:
//!
//! - `BGRA10_XR`'s alpha (and its twin's): Metal averages `floor(4080 a)` over the samples'
//!   alphas `a` (each the written value as a `half`, clamped to `[0, 1]`) and rounds half up to a
//!   code, three bits finer than the code; the modules average the codes with a tie rounded down
//!   (Metal's code in 84% of the probe's texels at 2 samples and 88% at 4, every one within a code);
//! - an `_sRGB` twin's color and `RGB10A2Unorm_sRGB`'s: Metal encodes an average of the samples'
//!   linear values (near that of the written values as `half`s, clamped to the codes' range),
//!   which lies within the samples' codes; the modules average the codes' decoded values in `f32`,
//!   encode it as a kernel's write does, and hold the code within the samples' (81% and 85% of the
//!   twins' channels, 99.5% within a code and at most 7 codes off where the samples' values nearly
//!   cancel; 83% and 87% of `RGB10A2Unorm_sRGB`'s, within 2);
//! - `RGB10A2Unorm_sRGB`'s 2-bit alpha: the average of the codes rounded half up (94% and 97%,
//!   within a code).
//!
//! A pixel whose samples hold one code resolves to it, as Metal's does, without the decode and the
//! encode. The rules apply at every count; Apple's GPUs have no texture past 4 samples.
//! [`ColorResolveFormat::exact`] says which forms are Metal's at every texel.
//!
//! **Synchronization** (consumer): the samples' writes before the draw's reads, and the resolve
//! texture's earlier accesses before its color writes.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::lower::arith::extended_range as xr;
use crate::metal::format::ExtendedRange;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, GlslOp, ImageOperands, ImageType,
    ImageUsage, IoDir, MemAccess, StorageClass, TexelMemory, Type, TypedModule, UnOp,
};
use crate::vk::Format;

use super::sample_planes::PlanesCopy;

/// How one channel of a twin resolve's format resolves (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResolveRule {
    /// An 8-bit unorm code.
    Unorm,
    /// An 8-bit sRGB code.
    Srgb,
}

/// One channel of a format: its rule and where its bits lie in a texel (counted from the
/// texel's least significant bit, its first byte's lowest).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolveChannel {
    pub rule: ResolveRule,
    pub offset: u8,
}

/// A color format the crate-owned resolve resolves (the module docs): a twin resolve's, by
/// component (R, G, B, A), the formats [`ColorResolveFormat::of`] names, or an extended-range
/// form's ([`ColorResolveFormat::extended_range`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorResolveFormat {
    format: Format,
    channels: [Option<ResolveChannel>; 4],
    /// Whether the result is written through the raw twin's view
    /// ([`ColorResolveFormat::written_raw`]).
    written_raw: bool,
    /// Whether the sRGB channels take the device's decode and encode
    /// ([`ColorResolveFormat::device_srgb`]).
    device_srgb: bool,
    /// The extended-range form whose raw codes the format holds
    /// ([`ColorResolveFormat::extended_range`]).
    extended: Option<ExtendedRange>,
}

/// A channel of `rule` at bit `offset`.
const fn at(rule: ResolveRule, offset: u8) -> Option<ResolveChannel> {
    Some(ResolveChannel { rule, offset })
}

impl ColorResolveFormat {
    /// The resolve of `format` into which an attachment of its sRGB twin resolves (the module
    /// docs): Metal's 8-bit unorm formats and their sRGB twins, as Vulkan formats. `None` for
    /// another format, which resolves by the device's `AVERAGE` or, extended-range, by
    /// [`ColorResolveFormat::extended_range`].
    pub fn of(format: Format) -> Option<ColorResolveFormat> {
        use ResolveRule::{Srgb as S, Unorm as U};
        let channels = match format {
            Format::R8_UNORM => [at(U, 0), None, None, None],
            Format::R8G8_UNORM => [at(U, 0), at(U, 8), None, None],
            Format::R8G8B8A8_UNORM => [at(U, 0), at(U, 8), at(U, 16), at(U, 24)],
            Format::B8G8R8A8_UNORM => [at(U, 16), at(U, 8), at(U, 0), at(U, 24)],
            Format::R8_SRGB => [at(S, 0), None, None, None],
            Format::R8G8_SRGB => [at(S, 0), at(S, 8), None, None],
            Format::R8G8B8A8_SRGB => [at(S, 0), at(S, 8), at(S, 16), at(U, 24)],
            Format::B8G8R8A8_SRGB => [at(S, 16), at(S, 8), at(S, 0), at(U, 24)],
            _ => return None,
        };
        Some(ColorResolveFormat {
            format,
            channels,
            written_raw: false,
            device_srgb: false,
            extended: None,
        })
    }

    /// The resolve of a multisampled color attachment of the extended-range `form`, held in its
    /// raw format's codes (`metal::format::ExtendedRange::raw_format`): `BGR10_XR`, `BGRA10_XR`,
    /// their `_sRGB` twins and `RGB10A2Unorm_sRGB`, the forms Metal makes multisampled textures
    /// of. `None` for another form, of which Metal makes none (the module docs). It has no
    /// [`channels`](Self::channels): the modules read and write the raw codes by the rules the
    /// module docs state for these forms, at every sample count.
    pub fn extended_range(form: ExtendedRange) -> Option<ColorResolveFormat> {
        match form {
            ExtendedRange::Bgr10 { .. }
            | ExtendedRange::Bgra10 { .. }
            | ExtendedRange::Rgb10A2Srgb => Some(ColorResolveFormat {
                format: form.raw_format(),
                channels: [None; 4],
                written_raw: false,
                device_srgb: false,
                extended: Some(form),
            }),
            _ => None,
        }
    }

    /// The extended-range form whose raw codes the format holds
    /// ([`extended_range`](Self::extended_range)), if any.
    pub fn extended(self) -> Option<ExtendedRange> {
        self.extended
    }

    /// Whether the resolve writes Metal's resolve at every texel: every format's but an
    /// extended-range form's whose resolve averages values its codes do not keep (`BGRA10_XR`'s
    /// alpha, an `_sRGB` twin's color, `RGB10A2Unorm_sRGB`'s channels), which the modules write a
    /// near-equivalent of where a pixel's samples differ (the module docs).
    pub fn exact(self) -> bool {
        matches!(
            self.extended,
            None | Some(ExtendedRange::Bgr10 { srgb: false })
        )
    }

    /// The Vulkan format.
    pub fn format(self) -> Format {
        self.format
    }

    /// Each component's channel (R, G, B, A), `None` where the format has none.
    pub fn channels(self) -> [Option<ResolveChannel>; 4] {
        self.channels
    }

    /// The format of the view [`ResolveSource::RawInputAttachment`] reads: an 8-bit sRGB format's
    /// UNORM twin, whose floats are the codes over 255; any other format itself.
    pub fn raw_format(self) -> Format {
        match self.format {
            Format::R8_SRGB => Format::R8_UNORM,
            Format::R8G8_SRGB => Format::R8G8_UNORM,
            Format::R8G8B8A8_SRGB => Format::R8G8B8A8_UNORM,
            Format::B8G8R8A8_SRGB => Format::B8G8R8A8_UNORM,
            format => format,
        }
    }

    /// The resolve writing its result through a view of the format's raw twin
    /// ([`raw_format`](Self::raw_format)): an sRGB channel's code as an 8-bit unorm code's float,
    /// `code / 255`, which the twin's store keeps as the code, in place of the code's linear value,
    /// which the format's own store encodes. Another format's is itself.
    pub fn written_raw(self) -> ColorResolveFormat {
        ColorResolveFormat {
            written_raw: self.raw_format() != self.format,
            device_srgb: false,
            ..self
        }
    }

    /// The resolve taking the device's own sRGB decode and encode for its sRGB channels (the
    /// module docs): it reads the attachment through a view in the format
    /// ([`ResolveSource::InputAttachment`], whose read decodes each sample), averages the decoded
    /// values, and writes the average through a view in the format, whose store encodes it, in
    /// place of Metal's 12-bit tables. For a device that has the sRGB format; its alpha keeps the
    /// unorm rule. A format with no sRGB channel is itself.
    pub fn device_srgb(self) -> ColorResolveFormat {
        ColorResolveFormat {
            device_srgb: self.srgb(),
            written_raw: self.written_raw && !self.srgb(),
            ..self
        }
    }

    /// Whether the sRGB channels take the device's decode and encode
    /// ([`device_srgb`](Self::device_srgb)).
    pub fn decodes_on_device(self) -> bool {
        self.device_srgb
    }

    /// The format of the view the result is written through: the raw twin's where
    /// [`written_raw`](Self::written_raw), else the format.
    pub fn written_format(self) -> Format {
        match self.written_raw {
            true => self.raw_format(),
            false => self.format,
        }
    }

    /// Whether a channel is sRGB.
    fn srgb(self) -> bool {
        self.channels
            .iter()
            .flatten()
            .any(|c| c.rule == ResolveRule::Srgb)
    }
}

/// Where the fragment stage reads a pixel's samples (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResolveSource {
    /// The multisampled attachment, as input attachment 0.
    InputAttachment,
    /// The multisampled attachment, as input attachment 0, viewed in its format's raw twin
    /// ([`ColorResolveFormat::raw_format`]).
    RawInputAttachment,
    /// An extended-range attachment's sample planes, viewed in their copy format.
    Planes(PlanesCopy),
}

/// The crate-owned resolve modules of one format, sample count and source on one device,
/// validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorResolveModules {
    /// The vertex stage: SPIR-V 1.5 words, entry point `main`, no resources.
    pub vertex: Arc<[u32]>,
    /// The fragment stage: SPIR-V 1.5 words, entry point `main`, the source at
    /// [`ColorResolveModules::SOURCE_BINDING`] of set [`ColorResolveModules::SET`], the output
    /// at `Location` 0.
    pub fragment: Arc<[u32]>,
    /// What the two stages need from the device.
    pub requirements: Requirements,
}

impl ColorResolveModules {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The source: an input attachment, or the planes' sampled image.
    pub const SOURCE_BINDING: u32 = 0;
}

/// Metal's 12-bit linear value of each 8-bit sRGB code, `round(decode(k) * 4095)` (probes, the M3
/// Max and the M4 alike).
pub const SRGB_RESOLVE_LINEAR: [u16; 256] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 40,
    42, 45, 48, 50, 53, 56, 59, 62, 66, 69, 72, 76, 79, 83, 87, 91, 95, 99, 103, 107, 112, 116,
    121, 126, 131, 136, 141, 146, 151, 156, 162, 168, 173, 179, 185, 191, 197, 204, 210, 216, 223,
    230, 237, 244, 251, 258, 265, 273, 280, 288, 296, 304, 312, 320, 329, 337, 346, 354, 363, 372,
    381, 390, 400, 409, 419, 428, 438, 448, 458, 469, 479, 490, 500, 511, 522, 533, 544, 555, 567,
    578, 590, 602, 614, 626, 639, 651, 664, 676, 689, 702, 715, 728, 742, 755, 769, 783, 797, 811,
    825, 840, 854, 869, 884, 899, 914, 929, 945, 960, 976, 992, 1008, 1024, 1041, 1057, 1074, 1091,
    1108, 1125, 1142, 1159, 1177, 1195, 1213, 1231, 1249, 1267, 1286, 1304, 1323, 1342, 1361, 1381,
    1400, 1420, 1440, 1459, 1480, 1500, 1520, 1541, 1562, 1582, 1603, 1625, 1646, 1668, 1689, 1711,
    1733, 1755, 1778, 1800, 1823, 1846, 1869, 1892, 1916, 1939, 1963, 1987, 2011, 2035, 2059, 2084,
    2109, 2133, 2159, 2184, 2209, 2235, 2260, 2286, 2312, 2339, 2365, 2392, 2419, 2446, 2473, 2500,
    2527, 2555, 2583, 2611, 2639, 2668, 2696, 2725, 2754, 2783, 2812, 2841, 2871, 2901, 2931, 2961,
    2991, 3022, 3052, 3083, 3114, 3146, 3177, 3209, 3240, 3272, 3304, 3337, 3369, 3402, 3435, 3468,
    3501, 3535, 3568, 3602, 3636, 3670, 3705, 3739, 3774, 3809, 3844, 3879, 3915, 3950, 3986, 4022,
    4059, 4095,
];

/// Metal's encode of a 12-bit linear value: the 8-bit code `c` from
/// `SRGB_RESOLVE_THRESHOLDS[c - 1]` on, code 0 below the first. Probes 1 to 3 observed each of
/// the 3991 values a 2- or 4-sample average reaches. Of the 105 no such average reaches (3923 and
/// above), 97 lie between two observed values of one code and take it; 4069 to 4076, between 254
/// and 255 and reached only at 8 or more samples, take the exact encode rounded half up, 254. The
/// table is the exact encode rounded half up but at 804, 1371, 2121 and 2569, each one code
/// higher.
pub const SRGB_RESOLVE_THRESHOLDS: [u16; 255] = [
    1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 23, 24, 26, 28, 30, 32, 34, 37, 39,
    42, 44, 47, 49, 52, 55, 58, 61, 64, 68, 71, 74, 78, 82, 85, 89, 93, 97, 102, 106, 110, 115,
    119, 124, 129, 134, 139, 144, 149, 154, 160, 165, 171, 177, 183, 189, 195, 201, 207, 214, 220,
    227, 234, 241, 248, 255, 262, 270, 277, 285, 293, 300, 308, 317, 325, 333, 342, 350, 359, 368,
    377, 386, 395, 405, 414, 424, 434, 444, 454, 464, 474, 485, 495, 506, 517, 528, 539, 550, 562,
    573, 585, 597, 609, 621, 633, 645, 658, 670, 683, 696, 709, 722, 736, 749, 763, 777, 790, 804,
    819, 833, 848, 862, 877, 892, 907, 922, 938, 953, 969, 985, 1001, 1017, 1033, 1049, 1066, 1083,
    1100, 1117, 1134, 1151, 1169, 1186, 1204, 1222, 1240, 1259, 1277, 1296, 1314, 1333, 1352, 1371,
    1391, 1410, 1430, 1450, 1470, 1490, 1510, 1531, 1552, 1572, 1593, 1615, 1636, 1657, 1679, 1701,
    1723, 1745, 1767, 1790, 1812, 1835, 1858, 1881, 1904, 1928, 1951, 1975, 1999, 2023, 2048, 2072,
    2097, 2121, 2146, 2172, 2197, 2222, 2248, 2274, 2300, 2326, 2352, 2379, 2406, 2433, 2460, 2487,
    2514, 2542, 2569, 2597, 2626, 2654, 2682, 2711, 2740, 2769, 2798, 2827, 2857, 2886, 2916, 2946,
    2977, 3007, 3038, 3068, 3099, 3130, 3162, 3193, 3225, 3257, 3289, 3321, 3354, 3386, 3419, 3452,
    3485, 3518, 3552, 3586, 3620, 3654, 3688, 3722, 3757, 3792, 3827, 3862, 3897, 3933, 3969, 4005,
    4041, 4077,
];

/// The exact linear value of an 8-bit sRGB code (IEC 61966-2-1).
fn srgb_decode(code: u32) -> f64 {
    let c = f64::from(code) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The binding symbol of the source in the fragment module ([`ColorResolveModules::SOURCE_BINDING`]):
/// an input attachment.
const SOURCE_INPUT: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::InputAttachment(0),
    index: 0,
};

/// The binding symbol of the planes in the fragment module.
const SOURCE_PLANES: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Texture,
    index: 0,
};

/// The scalar and vector types the fragment module uses.
#[derive(Clone, Copy)]
struct Types {
    u32_: TypeId,
    i32_: TypeId,
    f32_: TypeId,
    vec4: TypeId,
}

impl Types {
    fn new(b: &mut ModuleBuilder) -> BResult<Types> {
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let i32_ = b.ty(Type::Int {
            width: 32,
            signed: true,
        })?;
        let f32_ = b.ty(Type::Float { width: 32 })?;
        let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
        Ok(Types {
            u32_,
            i32_,
            f32_,
            vec4,
        })
    }
}

/// A scalar constant of `ty` with `bits`, as a value.
fn scalar(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })?;
    f.module().const_value(c)
}

/// The `u32` constant `v`.
fn uint(f: &mut FunctionBuilder<'_>, t: Types, v: u32) -> BResult<Value> {
    scalar(f, t.u32_, u64::from(v))
}

/// The `i32` constant `v`.
fn int(f: &mut FunctionBuilder<'_>, t: Types, v: i32) -> BResult<Value> {
    scalar(f, t.i32_, u64::from(v as u32))
}

/// The `f32` constant `v`.
fn float(f: &mut FunctionBuilder<'_>, t: Types, v: f32) -> BResult<Value> {
    scalar(f, t.f32_, u64::from(v.to_bits()))
}

/// A `Private` array of `elem` holding `values` (each a scalar's bits).
fn table(
    b: &mut ModuleBuilder,
    elem: TypeId,
    u32_: TypeId,
    values: impl ExactSizeIterator<Item = u64>,
) -> BResult<GlobalId> {
    let len = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(values.len() as u64),
    })?;
    let ty = b.ty(Type::Array {
        elem,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(values.len());
    for bits in values {
        parts.push(b.constant(Constant::Scalar {
            ty: elem,
            bits: ConstBits(bits),
        })?);
    }
    let init = b.constant(Constant::Composite {
        ty,
        parts: parts.into(),
    })?;
    b.plain_global(StorageClass::Private, ty, Some(init))
}

/// The sRGB tables of a module with an sRGB channel.
#[derive(Clone, Copy)]
struct SrgbTables {
    /// [`SRGB_RESOLVE_LINEAR`].
    linear: GlobalId,
    /// [`SRGB_RESOLVE_THRESHOLDS`].
    thresholds: GlobalId,
    /// Each code's linear value as an `f32`: what the attachment encodes as the code. Declared
    /// only where the module writes an sRGB channel's linear value (not
    /// [`ColorResolveFormat::written_raw`]).
    decoded: Option<GlobalId>,
    /// The `f32` midway between each code's linear value and the next code's: an input
    /// attachment's float is the code of the midpoints at or below it. Declared only where the
    /// module reads an sRGB attachment's floats ([`ResolveSource::InputAttachment`]).
    midpoints: Option<GlobalId>,
}

impl SrgbTables {
    /// The tables of `format`'s module reading `source`: [`SrgbTables::midpoints`] among them
    /// where it reads an sRGB attachment's floats, and [`SrgbTables::decoded`] where it writes
    /// linear values.
    fn declare(
        b: &mut ModuleBuilder,
        t: Types,
        format: ColorResolveFormat,
        source: ResolveSource,
    ) -> BResult<SrgbTables> {
        let linear = table(
            b,
            t.u32_,
            t.u32_,
            SRGB_RESOLVE_LINEAR.iter().map(|&v| u64::from(v)),
        )?;
        let thresholds = table(
            b,
            t.u32_,
            t.u32_,
            SRGB_RESOLVE_THRESHOLDS.iter().map(|&v| u64::from(v)),
        )?;
        let decoded = match format.written_raw {
            true => None,
            false => Some(table(
                b,
                t.f32_,
                t.u32_,
                (0..256).map(|k| u64::from((srgb_decode(k) as f32).to_bits())),
            )?),
        };
        let midpoints = match source == ResolveSource::InputAttachment {
            true => Some(table(
                b,
                t.f32_,
                t.u32_,
                (0..255).map(|k| {
                    let mid = (srgb_decode(k) + srgb_decode(k + 1)) / 2.0;
                    u64::from((mid as f32).to_bits())
                }),
            )?),
            false => None,
        };
        Ok(SrgbTables {
            linear,
            thresholds,
            decoded,
            midpoints,
        })
    }
}

/// The float an sRGB channel's resolved `code` (a `uint`) is written as: its linear value, which
/// the format's store encodes as the code, or, written through the raw twin (`tables` without
/// [`SrgbTables::decoded`]), `code / 255`, which the twin's store keeps as the code, as an 8-bit
/// unorm channel's.
fn srgb_written(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    tables: SrgbTables,
    code: Value,
) -> BResult<Value> {
    match tables.decoded {
        Some(decoded) => element(f, decoded, code),
        None => {
            let x = f.convert(ConvOp::ConvertUToF, code, t.f32_)?;
            let scale = float(f, t, (1.0 / 255.0f64) as f32)?;
            f.binary(BinOp::FMul, x, scale)
        }
    }
}

/// The element of the `Private` array `table` at `index`.
fn element(f: &mut FunctionBuilder<'_>, table: GlobalId, index: Value) -> BResult<Value> {
    let base = f.module().global_value(table)?;
    let at = f.access_chain(base, &[index])?;
    f.load(at, MemAccess::default())
}

/// The cubic in `q = x^(1/4)` that [`srgb_estimate`] evaluates, lowest power first: a least
/// squares fit of `255 (1.055 q^(5/3) - 0.055)`, the sRGB encode of `x` in codes (IEC 61966-2-1),
/// over `q` from `0.0031308^(1/4)` to 1, within 0.12 of it there.
const SRGB_CURVE: [f32; 4] = [-18.61511, 66.77221, 240.6806, -33.919685];

/// An estimate of `255 encode(x)`, the sRGB encode of the linear `x` (from 0 to 1) in codes, within
/// 0.12 of it: `255 · 12.92 x` up to 0.0031308, else [`SRGB_CURVE`] at `sqrt(sqrt(x))`.
fn srgb_estimate(f: &mut FunctionBuilder<'_>, t: Types, x: Value) -> BResult<Value> {
    let knee = float(f, t, 0.0031308)?;
    let low = f.binary(BinOp::FOrdLessThanEqual, x, knee)?;
    let slope = float(f, t, 255.0 * 12.92)?;
    let line = f.binary(BinOp::FMul, x, slope)?;
    let root = f.glsl(GlslOp::Sqrt, &[x])?;
    let q = f.glsl(GlslOp::Sqrt, &[root])?;
    let [c0, c1, c2, c3] = SRGB_CURVE;
    let mut curve = float(f, t, c3)?;
    for c in [c2, c1, c0] {
        let product = f.binary(BinOp::FMul, curve, q)?;
        let c = float(f, t, c)?;
        curve = f.binary(BinOp::FAdd, product, c)?;
    }
    f.select(low, line, curve)
}

/// The count of the 255 ascending entries of `table` that `x` reaches (`at_least(x, entry)`), from
/// `estimate`, a float that rounds to within one of that count: the estimate rounded and held to
/// `[0, 255]`, one lower where `x` does not reach the entry below it, one higher where `x` reaches
/// the entry it names. Exact for any such estimate, whatever its error.
fn corrected_rank(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    table: GlobalId,
    x: Value,
    estimate: Value,
    at_least: BinOp,
) -> BResult<Value> {
    let lowest = float(f, t, 0.0)?;
    let highest = float(f, t, 255.0)?;
    let e = f.glsl(GlslOp::RoundEven, &[estimate])?;
    let e = f.glsl(GlslOp::FClamp, &[e, lowest, highest])?;
    let e = f.convert(ConvOp::ConvertFToU, e, t.u32_)?;
    let zero = uint(f, t, 0)?;
    let one = uint(f, t, 1)?;
    // Entry `e - 1`, where `e` is above 0.
    let above_first = f.binary(BinOp::INotEqual, e, zero)?;
    let lifted = f.glsl(GlslOp::UMax, &[e, one])?;
    let below_at = f.binary(BinOp::ISub, lifted, one)?;
    let below = element(f, table, below_at)?;
    let reaches_below = f.binary(at_least, x, below)?;
    let short = f.unary(UnOp::LogicalNot, reaches_below)?;
    let down = f.binary(BinOp::LogicalAnd, above_first, short)?;
    // Entry `e`, where `e` is below 255.
    let entries = uint(f, t, 255)?;
    let inside = f.binary(BinOp::ULessThan, e, entries)?;
    let last = uint(f, t, 254)?;
    let at_index = f.glsl(GlslOp::UMin, &[e, last])?;
    let at = element(f, table, at_index)?;
    let reaches = f.binary(at_least, x, at)?;
    let up = f.binary(BinOp::LogicalAnd, inside, reaches)?;
    let lower = f.binary(BinOp::ISub, e, one)?;
    let higher = f.binary(BinOp::IAdd, e, one)?;
    let moved = f.select(up, higher, e)?;
    f.select(down, lower, moved)
}

/// The code of `channel` in a sample's `texel` read from `source` (the module docs), a `u32`.
fn code(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    srgb: Option<SrgbTables>,
    channel: ResolveChannel,
    component: u32,
    texel: Value,
    source: ResolveSource,
) -> BResult<Value> {
    let lane = f.composite_extract(texel, &[component])?;
    match channel.rule {
        ResolveRule::Srgb if source != ResolveSource::RawInputAttachment => {
            let midpoints = srgb
                .and_then(|tables| tables.midpoints)
                .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
            let estimate = srgb_estimate(f, t, lane)?;
            corrected_rank(f, t, midpoints, lane, estimate, BinOp::FOrdGreaterThanEqual)
        }
        // Read through the raw twin, an sRGB channel's float is its code over 255: an 8-bit unorm
        // code's.
        ResolveRule::Unorm | ResolveRule::Srgb => {
            let scale = float(f, t, 255.0)?;
            let half = float(f, t, 0.5)?;
            let x = f.binary(BinOp::FMul, lane, scale)?;
            let x = f.binary(BinOp::FAdd, x, half)?;
            f.convert(ConvOp::ConvertFToU, x, t.u32_)
        }
    }
}

/// The resolved output float of `channel` from each sample's `codes` (the module docs).
fn resolve(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    srgb: Option<SrgbTables>,
    channel: ResolveChannel,
    codes: &[Value],
) -> BResult<Value> {
    let n = codes.len() as u32;
    match channel.rule {
        ResolveRule::Unorm => {
            let mut sum = uint(f, t, n / 2)?;
            for &c in codes {
                sum = f.binary(BinOp::IAdd, sum, c)?;
            }
            let count = uint(f, t, n)?;
            let mean = f.binary(BinOp::UDiv, sum, count)?;
            let mean = f.convert(ConvOp::ConvertUToF, mean, t.f32_)?;
            let scale = float(f, t, (1.0 / 255.0f64) as f32)?;
            f.binary(BinOp::FMul, mean, scale)
        }
        ResolveRule::Srgb => {
            let tables = srgb.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
            let mut sum = uint(f, t, n / 2)?;
            for &c in codes {
                let linear = element(f, tables.linear, c)?;
                sum = f.binary(BinOp::IAdd, sum, linear)?;
            }
            let count = uint(f, t, n)?;
            let mean = f.binary(BinOp::UDiv, sum, count)?;
            // The 12-bit value as a linear value, `mean / 4095`, estimates its code.
            let real = f.convert(ConvOp::ConvertUToF, mean, t.f32_)?;
            let unit = float(f, t, 1.0 / 4095.0)?;
            let linear = f.binary(BinOp::FMul, real, unit)?;
            let estimate = srgb_estimate(f, t, linear)?;
            let code = corrected_rank(
                f,
                t,
                tables.thresholds,
                mean,
                estimate,
                BinOp::UGreaterThanEqual,
            )?;
            srgb_written(f, t, tables, code)
        }
    }
}

/// The resolved output float of `channel` where every sample holds `code`: what [`resolve`] gives
/// for `n` copies of it, at every count, without the average (the module docs):
///
/// - unorm: `(n c + n / 2) / n = c`, as `floor(n / 2) < n`;
/// - sRGB: the linear value `L` of `c` averages to `L`, and Metal's encode of `L` is `c` for every
///   code (`SRGB_RESOLVE_THRESHOLDS` against `SRGB_RESOLVE_LINEAR`, which a test checks).
fn resolve_alike(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    srgb: Option<SrgbTables>,
    channel: ResolveChannel,
    code: Value,
) -> BResult<Value> {
    match channel.rule {
        ResolveRule::Unorm => {
            let x = f.convert(ConvOp::ConvertUToF, code, t.f32_)?;
            let scale = float(f, t, (1.0 / 255.0f64) as f32)?;
            f.binary(BinOp::FMul, x, scale)
        }
        ResolveRule::Srgb => {
            let tables = srgb.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
            srgb_written(f, t, tables, code)
        }
    }
}

/// Whether every texel of `texels` (each a `vec4`) has the first's bits, as a `bool`.
fn alike(f: &mut FunctionBuilder<'_>, t: Types, texels: &[Value]) -> BResult<Value> {
    let uvec4 = f.module().ty(Type::Vector {
        elem: t.u32_,
        len: 4,
    })?;
    let mut words = Vec::with_capacity(texels.len());
    for &texel in texels {
        words.push(f.convert(ConvOp::Bitcast, texel, uvec4)?);
    }
    all_equal(f, t, &words)
}

/// The mean of `component` of each sample's `texels`, the values the device's sRGB view decoded:
/// their `f32` sum, in sample order, times `1 / n`, exact for the power-of-two counts the
/// modules take, which the sRGB view the result is written through encodes
/// ([`ColorResolveFormat::device_srgb`]).
fn device_mean(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    component: u32,
    texels: &[Value],
) -> BResult<Value> {
    let mut sum: Option<Value> = None;
    for &texel in texels {
        let lane = f.composite_extract(texel, &[component])?;
        sum = Some(match sum {
            None => lane,
            Some(s) => f.binary_no_contraction(BinOp::FAdd, s, lane)?,
        });
    }
    let sum = sum.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let scale = float(f, t, 1.0 / texels.len() as f32)?;
    f.binary_no_contraction(BinOp::FMul, sum, scale)
}

/// The resolved output floats (R, G, B, A; 0 where the format has no channel) of `format` from
/// each sample's `texels` read from `source` (the module docs).
///
/// A pixel of an sRGB format whose samples all hold the same bits (every pixel inside a
/// primitive, where one invocation writes every covered sample) takes [`resolve_alike`], which
/// gives what the rule gives without its average; another takes the rule ([`resolve`]). A format
/// of unorm channels alone takes the rule without comparing.
fn resolve_texel(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    srgb: Option<SrgbTables>,
    format: ColorResolveFormat,
    texels: &[Value],
    source: ResolveSource,
) -> BResult<[Value; 4]> {
    let zero = float(f, t, 0.0)?;
    let channels: Vec<(u32, ResolveChannel)> = (0u32..)
        .zip(format.channels)
        .filter_map(|(component, channel)| Some((component, channel?)))
        .collect();
    // The device's sRGB conversions: the mean of the decoded samples, no comparison (its cost is
    // the average's).
    if format.device_srgb {
        let mut lanes = [zero; 4];
        for &(component, channel) in &channels {
            lanes[component as usize] = match channel.rule {
                ResolveRule::Srgb => device_mean(f, t, component, texels)?,
                ResolveRule::Unorm => {
                    let mut codes = Vec::with_capacity(texels.len());
                    for &texel in texels {
                        codes.push(code(f, t, None, channel, component, texel, source)?);
                    }
                    resolve(f, t, None, channel, &codes)?
                }
            };
        }
        return Ok(lanes);
    }
    let (&first, rest) = texels
        .split_first()
        .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let mut firsts = Vec::with_capacity(channels.len());
    for &(component, channel) in &channels {
        firsts.push(code(f, t, srgb, channel, component, first, source)?);
    }
    let averaged = |f: &mut FunctionBuilder<'_>| -> BResult<Vec<Value>> {
        let mut out = Vec::with_capacity(channels.len());
        for (&(component, channel), &first) in channels.iter().zip(&firsts) {
            let mut codes = Vec::with_capacity(texels.len());
            codes.push(first);
            for &texel in rest {
                codes.push(code(f, t, srgb, channel, component, texel, source)?);
            }
            out.push(resolve(f, t, srgb, channel, &codes)?);
        }
        Ok(out)
    };
    let resolved = if format.srgb() {
        let same = alike(f, t, texels)?;
        let kept: ArmFn<'_, ()> = Box::new(|f, _| {
            let mut out = Vec::with_capacity(channels.len());
            for (&(_, channel), &first) in channels.iter().zip(&firsts) {
                out.push(resolve_alike(f, t, srgb, channel, first)?);
            }
            Ok(ArmEnd::FallThrough(out))
        });
        let average: ArmFn<'_, ()> = Box::new(|f, _| Ok(ArmEnd::FallThrough(averaged(f)?)));
        f.if_(&mut (), same, kept, Some(average))?.0
    } else {
        averaged(f)?
    };
    let mut lanes = [zero; 4];
    for (&(component, _), value) in channels.iter().zip(resolved) {
        lanes[component as usize] = value;
    }
    Ok(lanes)
}

/// The fragment stage of `format` at `samples` reading `source`, as a typed module, before
/// serialization: the module [`color_resolve_modules`] serializes. Its binding is the symbol
/// `SOURCE_INPUT` or `SOURCE_PLANES`. Refused where `source` is sample planes of a twin
/// resolve's format, which reads the attachment, or of an extended-range form whose copy does not
/// move its bits.
pub fn fragment_typed_module(
    format: ColorResolveFormat,
    samples: u32,
    source: ResolveSource,
) -> BResult<TypedModule> {
    if let Some(form) = format.extended {
        return extended_fragment(form, samples, source);
    }
    // The device's sRGB decode reads the attachment in the sRGB format itself.
    if matches!(source, ResolveSource::Planes(_))
        || (format.device_srgb && source != ResolveSource::InputAttachment)
    {
        return Err(crate::spirv::error::BuildError::Type(
            crate::spirv::error::TypeRule::FunctionSignature,
        ));
    }
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let srgb = if format.srgb() && !format.device_srgb {
        Some(SrgbTables::declare(&mut b, t, format, source)?)
    } else {
        None
    };
    let (image, coord) = declare_source(&mut b, t, source, t.f32_)?;
    let output = b.fixed_io(0, IoDir::Output, t.vec4, false)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let texels = read_samples(&mut f, t, (image, coord), samples)?;
    let lanes = resolve_texel(&mut f, t, srgb, format, &texels, source)?;
    let value = f.composite_construct(t.vec4, &lanes)?;
    let output = f.module().global_value(output)?;
    f.store(output, value, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The source binding of `source`, whose texels are `sampled` (`u32` or `f32`) lanes: the
/// multisampled input attachment, or the planes' sampled 2D-array image and the `FragCoord`
/// input it is fetched at.
fn declare_source(
    b: &mut ModuleBuilder,
    t: Types,
    source: ResolveSource,
    sampled: TypeId,
) -> BResult<(GlobalId, Option<GlobalId>)> {
    Ok(match source {
        ResolveSource::InputAttachment | ResolveSource::RawInputAttachment => {
            let ty = b.ty(Type::Image(ImageType {
                sampled_type: sampled,
                dim: Dim::SubpassData,
                depth: false,
                arrayed: false,
                multisampled: true,
                usage: ImageUsage::Storage,
                format: crate::spv::ImageFormat::Unknown,
            }))?;
            (
                b.resource(SOURCE_INPUT, StorageClass::UniformConstant, ty)?,
                None,
            )
        }
        ResolveSource::Planes(_) => {
            let ty = b.ty(Type::Image(ImageType {
                sampled_type: sampled,
                dim: Dim::D2,
                depth: false,
                arrayed: true,
                multisampled: false,
                usage: ImageUsage::Sampled,
                format: crate::spv::ImageFormat::Unknown,
            }))?;
            let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, t.vec4)?;
            (
                b.resource(SOURCE_PLANES, StorageClass::UniformConstant, ty)?,
                Some(coord),
            )
        }
    })
}

/// Each of the `samples` texels of the pixel the invocation shades, from the source `image`
/// ([`declare_source`]): the input attachment's samples, or the planes' layers at `coord`.
fn read_samples(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    (image, coord): (GlobalId, Option<GlobalId>),
    samples: u32,
) -> BResult<Vec<Value>> {
    let image = f.module().global_value(image)?;
    let image = f.load(image, MemAccess::default())?;
    let ivec2 = f.module().ty(Type::Vector {
        elem: t.i32_,
        len: 2,
    })?;
    let uvec3 = f.module().ty(Type::Vector {
        elem: t.u32_,
        len: 3,
    })?;
    let pixel = match coord {
        Some(coord) => {
            // A pixel's center is at `(x + 0.5, y + 0.5)`: the conversion truncates to its integer.
            let coord = f.module().global_value(coord)?;
            let coord = f.load(coord, MemAccess::default())?;
            let x = f.composite_extract(coord, &[0])?;
            let y = f.composite_extract(coord, &[1])?;
            let x = f.convert(ConvOp::ConvertFToU, x, t.u32_)?;
            let y = f.convert(ConvOp::ConvertFToU, y, t.u32_)?;
            Some((x, y))
        }
        None => None,
    };
    let mut texels = Vec::with_capacity(samples as usize);
    for s in 0..samples {
        let texel = match pixel {
            None => {
                let origin = f.module().constant(Constant::Null { ty: ivec2 })?;
                let origin = f.module().const_value(origin)?;
                let sample = int(f, t, s as i32)?;
                let ops = ImageOperands {
                    sample: Some(sample),
                    ..ImageOperands::default()
                };
                f.read(image, origin, &ops, TexelMemory::PRIVATE)?
            }
            Some((x, y)) => {
                let layer = uint(f, t, s)?;
                let at = f.composite_construct(uvec3, &[x, y, layer])?;
                let level = uint(f, t, 0)?;
                let ops = ImageOperands {
                    lod: Some(level),
                    ..ImageOperands::default()
                };
                f.fetch(image, at, &ops)?
            }
        };
        texels.push(texel);
    }
    Ok(texels)
}

/// The fragment stage of the extended-range `form` at `samples` reading `source` (the module
/// docs): the samples' raw texels in, the resolve's raw texel out, a `uvec4` at `Location` 0.
fn extended_fragment(
    form: ExtendedRange,
    samples: u32,
    source: ResolveSource,
) -> BResult<TypedModule> {
    // The planes hold the codes only where their copy moves the texels' bits.
    if matches!(source, ResolveSource::Planes(copy) if !copy.moves_bits()) {
        return Err(crate::spirv::error::BuildError::Type(
            crate::spirv::error::TypeRule::FunctionSignature,
        ));
    }
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let uvec4 = b.ty(Type::Vector {
        elem: t.u32_,
        len: 4,
    })?;
    let read = declare_source(&mut b, t, source, t.u32_)?;
    let output = b.fixed_io(0, IoDir::Output, uvec4, false)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let mut raws = Vec::with_capacity(samples as usize);
    for texel in read_samples(&mut f, t, read, samples)? {
        raws.push(extended_texel(&mut f, t, form, texel, source)?);
    }
    let value = resolve_extended(&mut f, t, form, &raws)?;
    let output = f.module().global_value(output)?;
    f.store(output, value, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The `uvec4` of `lanes`, each a `u32` constant.
fn uvec4_const(f: &mut FunctionBuilder<'_>, t: Types, lanes: [u32; 4]) -> BResult<Value> {
    let uvec4 = f.module().ty(Type::Vector {
        elem: t.u32_,
        len: 4,
    })?;
    let mut v = Vec::with_capacity(4);
    for lane in lanes {
        v.push(uint(f, t, lane)?);
    }
    f.composite_construct(uvec4, &v)
}

/// Whether the `uvec4`s `words` all hold the first's bits, as a `bool`.
fn all_equal(f: &mut FunctionBuilder<'_>, t: Types, words: &[Value]) -> BResult<Value> {
    let (&first, rest) = words
        .split_first()
        .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let mut differ: Option<Value> = None;
    for &other in rest {
        let bits = f.binary(BinOp::BitwiseXor, first, other)?;
        differ = Some(match differ {
            None => bits,
            Some(d) => f.binary(BinOp::BitwiseOr, d, bits)?,
        });
    }
    let differ = differ.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let mut any = f.composite_extract(differ, &[0])?;
    for lane in 1..4 {
        let word = f.composite_extract(differ, &[lane])?;
        any = f.binary(BinOp::BitwiseOr, any, word)?;
    }
    let zero = uint(f, t, 0)?;
    f.binary(BinOp::IEqual, any, zero)
}

/// The raw texel (the raw format's `uvec4`, as an input attachment of it reads) of `form` in a
/// sample's `texel` read from `source`: the texel itself, or, from sample planes of 8-byte texels
/// (`PlanesCopy::Rg32`, two words), `BGRA10_XR`'s four 16-bit fields, the lowest first.
fn extended_texel(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    form: ExtendedRange,
    texel: Value,
    source: ResolveSource,
) -> BResult<Value> {
    if !(matches!(source, ResolveSource::Planes(PlanesCopy::Rg32))
        && matches!(form, ExtendedRange::Bgra10 { .. }))
    {
        return Ok(texel);
    }
    let uvec4 = f.module().ty(Type::Vector {
        elem: t.u32_,
        len: 4,
    })?;
    let low = uint(f, t, 0xffff)?;
    let sixteen = uint(f, t, 16)?;
    let mut fields = Vec::with_capacity(4);
    for word in 0..2 {
        let w = f.composite_extract(texel, &[word])?;
        fields.push(f.binary(BinOp::BitwiseAnd, w, low)?);
        fields.push(f.binary(BinOp::ShiftRightLogical, w, sixteen)?);
    }
    f.composite_construct(uvec4, &fields)
}

/// The codes (red first) of `form` averaged over `codes`, each sample's, as Metal's resolve or its
/// near-equivalent takes them (the module docs): the sum plus `bias` per lane, shifted down by
/// the count's power of two.
fn averaged_codes(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    codes: &[Value],
    bias: [u32; 4],
) -> BResult<Value> {
    let (&first, rest) = codes
        .split_first()
        .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let mut sum = first;
    for &c in rest {
        sum = f.binary(BinOp::IAdd, sum, c)?;
    }
    let bias = uvec4_const(f, t, bias)?;
    let sum = f.binary(BinOp::IAdd, sum, bias)?;
    let n = codes.len() as u32;
    let shift = uvec4_const(f, t, [n.trailing_zeros(); 4])?;
    f.binary(BinOp::ShiftRightLogical, sum, shift)
}

/// The raw texel the resolve of `form` writes from each sample's raw texel `raws` (the module
/// docs).
fn resolve_extended(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    form: ExtendedRange,
    raws: &[Value],
) -> BResult<Value> {
    let n = raws.len() as u32;
    let mut codes = Vec::with_capacity(raws.len());
    for &raw in raws {
        codes.push(xr::codes(f, form, raw)?);
    }
    let &first = codes
        .first()
        .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    // A color code rounds a tie up; `BGRA10_XR`'s alpha down, and `RGB10A2Unorm_sRGB`'s up.
    let half = n / 2;
    let alpha = match form {
        ExtendedRange::Bgra10 { .. } => half - 1,
        _ => half,
    };
    let bias = [half, half, half, alpha];
    if matches!(
        form,
        ExtendedRange::Bgr10 { srgb: false } | ExtendedRange::Bgra10 { srgb: false }
    ) {
        let code = averaged_codes(f, t, &codes, bias)?;
        return xr::pack(f, form, code);
    }
    // The decode, the average and the encode run only where the samples' codes differ.
    let same = all_equal(f, t, &codes)?;
    let kept: ArmFn<'_, ()> =
        Box::new(|f, _| Ok(ArmEnd::FallThrough(vec![xr::pack(f, form, first)?])));
    let average: ArmFn<'_, ()> = Box::new(|f, _| {
        let mut sum: Option<Value> = None;
        for &raw in raws {
            let v = xr::decode(f, form, raw, false)?;
            sum = Some(match sum {
                None => v,
                Some(s) => f.binary_no_contraction(BinOp::FAdd, s, v)?,
            });
        }
        let sum = sum.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
        // `1 / n` of a power of two: the product is exact.
        let scale = float(f, t, 1.0 / n as f32)?;
        let scale = f.composite_construct(t.vec4, &[scale; 4])?;
        let mean = f.binary_no_contraction(BinOp::FMul, sum, scale)?;
        let alpha_code = averaged_codes(f, t, &codes, bias)?;
        let colors = xr::write_codes(f, form, mean)?;
        // Within the samples' codes, where Metal's encode of an average of values that encode to
        // them lies: a decoded code's encode can fall below it where codes share a decode step
        // (`RGB10A2Unorm_sRGB`'s darkest).
        let (mut lo, mut hi) = (first, first);
        for &c in &codes[1..] {
            lo = f.glsl(GlslOp::UMin, &[lo, c])?;
            hi = f.glsl(GlslOp::UMax, &[hi, c])?;
        }
        let colors = f.glsl(GlslOp::UClamp, &[colors, lo, hi])?;
        let code = f.vector_shuffle(colors, alpha_code, &[0, 1, 2, 7])?;
        Ok(ArmEnd::FallThrough(vec![xr::pack(f, form, code)?]))
    });
    let out = f.if_(&mut (), same, kept, Some(average))?.0;
    out.first()
        .copied()
        .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)
}

/// The fixed binding numbers of the modules' symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ColorResolveModules::SET),
        Symbol::Binding(s) if *s == SOURCE_INPUT || *s == SOURCE_PLANES => {
            Some(ColorResolveModules::SOURCE_BINDING)
        }
        _ => None,
    }
}

/// `m` verified against `device`'s profile (requirements included), serialized, and numbered
/// with the fixed bindings; `what` names it in an internal error. Refused, `Unsupported` naming
/// `feature` and the first capability, where the device lacks what `m` needs.
fn finish(
    device: &Device,
    m: TypedModule,
    feature: &Feature,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    profile
        .check(m.requirements())
        .map_err(|c| Error::Unsupported(Unsupported::needs(feature.clone(), c)))?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok((words.into(), r.requirements))
}

/// The resolve modules of `format` at `samples` (a power of two from 2 to 64) reading `source`,
/// on `device`: built with the typed builder (the vertex stage is the sample planes merge's,
/// `bind::sample_planes`), checked by the owned verifier against the device's profile
/// (requirements included), serialized, and numbered with the fixed binding of
/// [`ColorResolveModules`]. Refused, `Unsupported` naming `feature` (what the consumer resolves
/// for) and what the device lacks, where it lacks what the modules need; an internal error for
/// another sample count; an error where `source` is sample planes of a twin resolve's format,
/// which reads the attachment, or of an extended-range form whose copy does not move its bits.
pub fn color_resolve_modules(
    device: &Device,
    format: ColorResolveFormat,
    samples: u32,
    source: ResolveSource,
    feature: Feature,
) -> Result<ColorResolveModules, Error> {
    if !samples.is_power_of_two() || !(2..=64).contains(&samples) {
        return Err(Error::Internal(InternalError::new(
            InternalKind::Invariant,
            format!(
                "a color resolve of {:?} at {samples} samples",
                format.format()
            ),
        )));
    }
    let (vertex, vertex_needs) = finish(
        device,
        super::sample_planes::merge_vertex_typed_module()?,
        &feature,
        "color resolve's vertex stage",
    )?;
    let (fragment, fragment_needs) = finish(
        device,
        fragment_typed_module(format, samples, source)?,
        &feature,
        "color resolve",
    )?;
    let requirements = device
        .profile()
        .union_requirements(&vertex_needs, &fragment_needs);
    Ok(ColorResolveModules {
        vertex,
        fragment,
        requirements,
    })
}

#[cfg(test)]
mod extended_tests;
#[cfg(test)]
mod tests;
