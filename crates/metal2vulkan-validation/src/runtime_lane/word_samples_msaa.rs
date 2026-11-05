//! Multisampled textures of the four formats a device may render only through the view of their
//! word (`metal::format::WordAttachment`): `R8Unorm_sRGB`, `A1BGR5Unorm`, `ABGR4Unorm` and
//! `RGB9E5Float`, at 2 and 4 samples, on the device the cfg selects, under the validation layer,
//! over the owned fixture `wsm_samples`.
//!
//! Each leg runs our probe `msaa_word_formats.m`'s passes over the first [`PIXELS`] pixels of its
//! target (one row of 16): a pass clears the target and draws each sample its own value (the
//! probe's generator; the last sample of every eighth pixel discarded, left to the clear), storing
//! and resolving; a second pass loads the samples, draws nothing and resolves. Each sample's word
//! is read through an unsigned view of the texture and its value through `texture2d_ms<float>`,
//! and the resolves are copied out, against Metal's answers on an Apple M3 Max under macOS 27.0.1
//! ([`METAL`]).
//!
//! Two devices run the legs: the device as it is, and one that withholds the four formats' own
//! attachments (`DeviceOptions::withheld_attachment_formats`), which holds each multisampled
//! texture the device renders in its word's format (`image_plan::word_samples`) where the device
//! renders the word at the count. Each leg prints which way it held each format. A word's samples
//! are Metal's words, read as Metal reads them (an sRGB word's by the device's decode,
//! `srgb-texel-decode`, within a code of the transfer where it is not Metal's, and a 5- or 6-bit
//! field's by the device's widening, `device-texel-conversion`, within its bound there: [`read`]),
//! and its
//! resolve, the crate-owned one, is Metal's
//! for `R8Unorm_sRGB` and `RGB9E5Float` and within one code of each field of Metal's for the
//! packed formats, announced as `word-resolve`. A format the device renders itself stores and
//! resolves by the device's own conversions and average, each within one code (an `RGB9E5Float`
//! channel within its mantissa's last bits) of Metal's. A blend is within one code of Metal's, and
//! further by as far as the sample it blends over lies from Metal's: the device's blend over the
//! device's stored code, each its own arithmetic (`device-color-conversion`; lavapipe, Mesa 26.2.4,
//! blends an `A1BGR5Unorm` sample at 4 samples two codes from Metal's over a sample its fill
//! stored a code from Metal's). A sample count the device's textures or pipelines lack is skipped
//! with a printed line, and the other counts are checked.

use super::degrade_notices::{assert_once_allowing, made_render_kinds, resolve_kind};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkProperty;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::format::{direct_format, WordAttachment};
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, PixelFormat, PrimitiveType, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan::vk;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, Error, Library,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The kind this lane's word resolves announce, for `degrade_notices`' census (the crate-owned
/// resolve's extra pass, `shader-resolve`, is `extended_range_resolve`'s site).
pub(super) const WORD_RESOLVE: [Degradation; 1] = [Degradation::WordResolve];

/// The target's width (the probe's), and the pixels the lane draws: its first row.
const WIDTH: u64 = 16;
const PIXELS: usize = 16;

/// Metal's answers for one format and sample count, over the probe's first [`PIXELS`] pixels:
/// each sample's word after the fill (pixel by pixel, its samples in turn), the fill's resolve,
/// the resolve of a pass that loaded the samples and drew nothing, and each sample's word and the
/// resolve after a pass that loaded them and blended.
struct Metal {
    format: PixelFormat,
    samples: u32,
    stored: &'static [u32],
    resolved: [u32; PIXELS],
    loaded: [u32; PIXELS],
    blended: &'static [u32],
    blend_resolved: [u32; PIXELS],
}

/// Probe `msaa_word_formats.m` on an Apple M3 Max under macOS 27.0.1 (`FILL W`, `FILL R`,
/// `LOADED R`, `BLEND W` and `BLEND R` lines).
#[rustfmt::skip]
const METAL: [Metal; 8] = [
    Metal {
        format: PixelFormat::R8UnormSRGB,
        samples: 2,
        stored: &[
            0x6a, 0x6a, 0xb9, 0x90, 0xcb, 0x85, 0xba, 0xa9,
            0x9a, 0x9a, 0xd5, 0xee, 0xc1, 0xae, 0xd0, 0x59,
            0xbb, 0xbb, 0xc4, 0x8c, 0xe0, 0x7e, 0xbe, 0xff,
            0xd5, 0xd5, 0xd0, 0xda, 0xa2, 0x93, 0xd6, 0x59,
        ],
        resolved: [
            0x6a, 0xa6, 0xad, 0xb2, 0x9a, 0xe2, 0xb8, 0xa3,
            0xbb, 0xab, 0xb8, 0xe2, 0xd5, 0xd5, 0x9b, 0xa7,
        ],
        loaded: [
            0x6a, 0xa6, 0xad, 0xb2, 0x9a, 0xe2, 0xb8, 0xa3,
            0xbb, 0xab, 0xb8, 0xe2, 0xd5, 0xd5, 0x9b, 0xa7,
        ],
        blended: &[
            0x6d, 0x73, 0xb5, 0xdc, 0xd2, 0xb6, 0xcd, 0xc6,
            0x8b, 0x84, 0xbc, 0xec, 0xe1, 0xbd, 0xb5, 0x7c,
            0xb5, 0x9c, 0xe9, 0x87, 0xde, 0x7b, 0xb3, 0xcf,
            0xa2, 0x9a, 0xd9, 0xcf, 0xa1, 0xb2, 0xb0, 0x80,
        ],
        blend_resolved: [
            0x70, 0xca, 0xc5, 0xca, 0x88, 0xd6, 0xd0, 0x9c,
            0xa9, 0xc0, 0xb5, 0xc2, 0x9e, 0xd4, 0xaa, 0x9a,
        ],
    },
    Metal {
        format: PixelFormat::R8UnormSRGB,
        samples: 4,
        stored: &[
            0x6a, 0x6a, 0x6a, 0x6a, 0xcb, 0xcb, 0x85, 0x85,
            0x9a, 0xd7, 0xd5, 0xee, 0xc1, 0xae, 0xd0, 0x4a,
            0xbb, 0xbb, 0xbb, 0xbb, 0xe0, 0xe0, 0x7e, 0x7e,
            0xd5, 0xda, 0xd0, 0xda, 0xa2, 0x93, 0xd6, 0x59,
            0xda, 0xda, 0xda, 0xda, 0xd8, 0xd8, 0x93, 0x93,
            0x90, 0xbb, 0xec, 0xb5, 0x9a, 0xdf, 0x00, 0xd7,
            0xcf, 0xcf, 0xcf, 0xcf, 0xe7, 0xe7, 0xc3, 0xc3,
            0xaa, 0x7f, 0x93, 0x83, 0xdc, 0xd2, 0x63, 0x59,
        ],
        resolved: [
            0x6a, 0xad, 0xd0, 0xac, 0xbb, 0xb8, 0xd6, 0xa1,
            0xda, 0xba, 0xbf, 0xb0, 0xcf, 0xd6, 0x91, 0xa9,
        ],
        loaded: [
            0x6a, 0xad, 0xd0, 0xac, 0xbb, 0xb8, 0xd6, 0xa1,
            0xda, 0xba, 0xbf, 0xb0, 0xcf, 0xd6, 0x91, 0xa9,
        ],
        blended: &[
            0x74, 0x79, 0x72, 0x7f, 0xbe, 0x6f, 0x7b, 0x84,
            0xa7, 0xde, 0xd5, 0xd6, 0xc1, 0xad, 0xdd, 0xb9,
            0xd2, 0xc0, 0xc0, 0xd3, 0xc0, 0x95, 0x61, 0x61,
            0x77, 0xbd, 0xcb, 0x97, 0x9c, 0xdd, 0xce, 0x7c,
            0xda, 0xda, 0xda, 0xda, 0x9f, 0x99, 0x97, 0xb5,
            0x96, 0xb9, 0xe3, 0xde, 0x61, 0x92, 0xe9, 0x97,
            0xba, 0xcd, 0xbd, 0xc7, 0xe4, 0xd4, 0xbb, 0xbf,
            0xba, 0xcc, 0x90, 0x94, 0xd2, 0xb3, 0x6c, 0xab,
        ],
        blend_resolved: [
            0x78, 0x8f, 0xcd, 0xc2, 0xca, 0x8d, 0xaa, 0xb6,
            0xda, 0xa2, 0xc7, 0xa6, 0xc3, 0xcd, 0xad, 0xac,
        ],
    },
    Metal {
        format: PixelFormat::A1BGR5Unorm,
        samples: 2,
        stored: &[
            0x256d, 0x256d, 0x7b3e, 0x4d28, 0x96ff, 0x3c5f, 0x7fe5, 0x6574,
            0x5137, 0x5137, 0xa835, 0xd4a0, 0x85d9, 0x6f35, 0x9cbb, 0x1a6c,
            0x7f07, 0x7f07, 0x8f1d, 0x403d, 0xbcf3, 0x3dfe, 0x8011, 0xff17,
            0xae02, 0xae02, 0x9800, 0xb6d0, 0x5d70, 0x4bcb, 0xafe7, 0x1a6c,
        ],
        resolved: [
            0x256d, 0x6432, 0x6daf, 0x76ac, 0x5137, 0xba6a, 0x7e67, 0x5bb3,
            0x7f07, 0x6bad, 0x7d79, 0xbb93, 0xae02, 0xab88, 0x549c, 0x6529,
        ],
        loaded: [
            0x256d, 0x6432, 0x65af, 0x76ad, 0x5137, 0xba6b, 0x7667, 0x5b73,
            0x7f07, 0x63ad, 0x7d79, 0xbb93, 0xae02, 0xa388, 0x549d, 0x6529,
        ],
        blended: &[
            0x24e1, 0x2bcd, 0x733c, 0xb5d0, 0xa5f9, 0x745d, 0x99ab, 0x8e5c,
            0x4579, 0x3f3b, 0x80f5, 0xd4a0, 0xbb4b, 0x85b5, 0x75fb, 0x332c,
            0x7687, 0x5405, 0xcc61, 0x38fd, 0xbcb3, 0x3dbc, 0x70cf, 0x9edd,
            0x5dd4, 0x5596, 0xa95c, 0x9c94, 0x5d2e, 0x71dd, 0x75a7, 0x3a97,
        ],
        blend_resolved: [
            0x2c57, 0x9466, 0x8d2b, 0x93e3, 0x4679, 0xaaaa, 0x9ca1, 0x5472,
            0x6547, 0x82af, 0x7d36, 0x8bd5, 0x5dd6, 0xa2d8, 0x63a4, 0x541f,
        ],
    },
    Metal {
        format: PixelFormat::A1BGR5Unorm,
        samples: 4,
        stored: &[
            0x256d, 0x256d, 0x256d, 0x256d, 0x96ff, 0x96ff, 0x3c5f, 0x3c5f,
            0x5137, 0xa996, 0xa835, 0xd4a0, 0x85d9, 0x6f35, 0x9cbb, 0x10a2,
            0x7f07, 0x7f07, 0x7f07, 0x7f07, 0xbcf3, 0xbcf3, 0x3dfe, 0x3dfe,
            0xae02, 0xb2fe, 0x9800, 0xb6d0, 0x5d70, 0x4bcb, 0xafe7, 0x1a6c,
            0xb2a1, 0xb2a1, 0xb2a1, 0xb2a1, 0xa82b, 0xa82b, 0x489c, 0x489c,
            0x4a2e, 0x7a55, 0xd4e5, 0x75ff, 0x5014, 0xb833, 0x049b, 0xa9ac,
            0x9e61, 0x9e61, 0x9e61, 0x9e61, 0xcc86, 0xcc86, 0x89d2, 0x89d2,
            0x66be, 0x3d9d, 0x4cf6, 0x3bf9, 0xb668, 0xa028, 0x265d, 0x1a6c,
        ],
        resolved: [
            0x256d, 0x6daf, 0xa1e8, 0x6cab, 0x7f07, 0x7d79, 0xac14, 0x5ce2,
            0xb2a1, 0x7865, 0x83e9, 0x69a5, 0x9e61, 0xab4c, 0x4d32, 0x63e6,
        ],
        loaded: [
            0x256d, 0x65af, 0x99e9, 0x64ab, 0x7f07, 0x7d79, 0xabd4, 0x5ce3,
            0xb2a1, 0x7863, 0x83e9, 0x69a3, 0x9e61, 0xab0c, 0x4d33, 0x63e6,
        ],
        blended: &[
            0x2d9f, 0x3597, 0x2da1, 0x35cf, 0x7ebf, 0x2d7f, 0x33db, 0x3c5f,
            0x622d, 0xb99a, 0xa835, 0xac6a, 0x861b, 0x6ae7, 0xb529, 0x7862,
            0xa38b, 0x8647, 0x8647, 0xa38b, 0x8535, 0x4d35, 0x259c, 0x259c,
            0x3309, 0x80e1, 0x90c8, 0x54e4, 0x53b0, 0xb6f3, 0x9af5, 0x33f2,
            0xb265, 0xb16d, 0xb1a9, 0xb225, 0x5be7, 0x5427, 0x5120, 0x75b7,
            0x522c, 0x7c57, 0xc463, 0xbcf7, 0x2414, 0x4a1f, 0xcf15, 0x52b4,
            0x7cfd, 0x9e23, 0x8539, 0x8deb, 0xc448, 0xa356, 0x79c6, 0x81cc,
            0x7d36, 0x9c5f, 0x4db8, 0x4c35, 0xa4ae, 0x729a, 0x2ddd, 0x6d51,
        ],
        blend_resolved: [
            0x3599, 0x44ef, 0x9a28, 0x83a3, 0x9509, 0x4569, 0x6256, 0x7472,
            0xb1e7, 0x5ba9, 0x93e9, 0x641f, 0x8db1, 0x9acc, 0x6cf1, 0x6c9c,
        ],
    },
    Metal {
        format: PixelFormat::ABGR4Unorm,
        samples: 2,
        stored: &[
            0x2abb, 0x2abb, 0x76f5, 0x4a93, 0x9dff, 0x487d, 0x7f9d, 0x6ad1,
            0x52da, 0x52da, 0xa0ca, 0xd884, 0x8b69, 0x6ede, 0x99ef, 0x15b7,
            0x7e1a, 0x7e1a, 0x8e79, 0x40fc, 0xb9cf, 0x3bf4, 0x804e, 0xfd5b,
            0xac15, 0xac15, 0x9005, 0xbd46, 0x5ab0, 0x472b, 0xaf9b, 0x15b7,
        ],
        resolved: [
            0x2abb, 0x68c4, 0x6bbe, 0x7db7, 0x52da, 0xb4a7, 0x7c9c, 0x57cb,
            0x7e1a, 0x67bb, 0x7ad9, 0xb74d, 0xac15, 0xa725, 0x5976, 0x6aa9,
        ],
        loaded: [
            0x2abb, 0x58c4, 0x6abe, 0x6cb7, 0x52da, 0xb4a7, 0x7c9b, 0x57cb,
            0x7e1a, 0x67ba, 0x7ad9, 0xb74c, 0xac15, 0xa725, 0x4866, 0x5aa9,
        ],
        blended: &[
            0x2989, 0x373b, 0x76f5, 0xbb47, 0xabed, 0x787a, 0x93ac, 0x9c75,
            0x4aea, 0x3ded, 0x81c8, 0xd884, 0xb63a, 0x7bdb, 0x7beb, 0x36b6,
            0x7d19, 0x581a, 0xc88e, 0x41fb, 0xb9ce, 0x3bf4, 0x723c, 0x9d7e,
            0x5b57, 0x5b68, 0xa277, 0xa957, 0x5ab0, 0x736a, 0x6b99, 0x356a,
        ],
        blend_resolved: [
            0x286a, 0x9996, 0x99ac, 0x9799, 0x4ceb, 0xa5a6, 0x998a, 0x59c9,
            0x6a19, 0x85bd, 0x7ad9, 0x875d, 0x5b57, 0xa567, 0x6695, 0x587a,
        ],
    },
    Metal {
        format: PixelFormat::ABGR4Unorm,
        samples: 4,
        stored: &[
            0x2abb, 0x2abb, 0x2abb, 0x2abb, 0x9dff, 0x9dff, 0x487d, 0x487d,
            0x52da, 0xa354, 0xa0ca, 0xd884, 0x8b69, 0x6ede, 0x99ef, 0x1186,
            0x7e1a, 0x7e1a, 0x7e1a, 0x7e1a, 0xb9cf, 0xb9cf, 0x3bf4, 0x3bf4,
            0xac15, 0xa5f3, 0x9005, 0xbd46, 0x5ab0, 0x472b, 0xaf9b, 0x15b7,
            0xb58e, 0xb58e, 0xb58e, 0xb58e, 0xa0ae, 0xa0ae, 0x4172, 0x4172,
            0x44b1, 0x745e, 0xd998, 0x7bf9, 0x5055, 0xb0cc, 0x096d, 0xa3b0,
            0x9c8b, 0x9c8b, 0x9c8b, 0x9c8b, 0xc912, 0xc912, 0x8440, 0x8440,
            0x6cf0, 0x3a78, 0x49d1, 0x37d8, 0xbca6, 0xa0a4, 0x2c78, 0x15b7,
        ],
        resolved: [
            0x2abb, 0x6bbe, 0x93a7, 0x69ab, 0x7e1a, 0x7ad9, 0xa854, 0x5987,
            0xb58e, 0x7198, 0x87a8, 0x7398, 0x9c8b, 0xa631, 0x4ac4, 0x6796,
        ],
        loaded: [
            0x2abb, 0x6abe, 0x9397, 0x69ab, 0x7e1a, 0x7ad9, 0xa755, 0x5987,
            0xb58e, 0x7088, 0x87a8, 0x6387, 0x9c8b, 0xa621, 0x49c4, 0x6796,
        ],
        blended: &[
            0x3a79, 0x3b69, 0x2a89, 0x3b3b, 0x8cfd, 0x2afd, 0x376b, 0x487d,
            0x64b8, 0xb364, 0xa0ca, 0xa8a4, 0x8b68, 0x65ac, 0xbaac, 0x7187,
            0xa72b, 0x8d19, 0x8d19, 0xa72b, 0x89cc, 0x4adb, 0x2b77, 0x2b77,
            0x362d, 0x8189, 0x9125, 0x5999, 0x57b2, 0xbdcc, 0x95dc, 0x37c7,
            0xb49c, 0xb2bb, 0xb3ab, 0xb49c, 0x579b, 0x589b, 0x4282, 0x7bdd,
            0x44b1, 0x786b, 0xc887, 0xb9db, 0x2858, 0x447e, 0xce5d, 0x55d6,
            0x79ee, 0x9c9a, 0x8aed, 0x8bb9, 0xc922, 0xa657, 0x7417, 0x8432,
            0x79d5, 0x9879, 0x4ad2, 0x48c7, 0xa9b7, 0x7565, 0x2b77, 0x6a4a,
        ],
        blend_resolved: [
            0x3b69, 0x49bd, 0x94a7, 0x878a, 0x9a2a, 0x4aa9, 0x6459, 0x78c8,
            0xb4ac, 0x57a9, 0x9798, 0x687a, 0x8acb, 0x9634, 0x69c6, 0x6977,
        ],
    },
    Metal {
        format: PixelFormat::RGB9E5Float,
        samples: 2,
        stored: &[
            0x880001b3, 0x880001b3, 0xc0000192, 0x8f460c00,
            0xb6740016, 0x8dd07600, 0xb1b80177, 0xbff4000f,
            0x98015d13, 0x98015d13, 0x9f30f804, 0xfffc0000,
            0xb8030e00, 0x9c981000, 0x86401c00, 0xb4681a00,
            0xa0000304, 0xa0000304, 0xa0032c3f, 0x9800015d,
            0x87506000, 0x8802d26c, 0x78000118, 0xa81c01fa,
            0x9444078b, 0x9444078b, 0x7b360800, 0xc7d40000,
            0x3925f910, 0xc44c0000, 0x7409b001, 0xb4681a00,
        ],
        resolved: [
            0x880001b3, 0xb8100592, 0xaea40416, 0xbc680065,
            0x98015d13, 0xf7fc0000, 0xb04b1000, 0xac801a00,
            0xa0000304, 0x98032cee, 0x83ab026c, 0xa01c01ff,
            0x9444078b, 0xbfd80200, 0xbc4c0000, 0xac6c1c00,
        ],
        loaded: [
            0x880001b3, 0xb8100592, 0xaea40416, 0xbc680065,
            0x98015d13, 0xf7fc0000, 0xb04b1000, 0xac801a00,
            0xa0000304, 0x98032cee, 0x83ab026c, 0xa01c01ff,
            0x9444078b, 0xbfd80200, 0xbc4c0000, 0xac6c1c00,
        ],
        blended: &[
            0x608003ec, 0x80040100, 0xc0000152, 0x877a1c00,
            0xb8f80120, 0x94442400, 0xb2440109, 0xb44c0008,
            0xe80001fc, 0xd8000167, 0x9df0ca03, 0xf5cc0000,
            0xf802a600, 0xa154051f, 0x90024804, 0x96e86c58,
            0x88001780, 0x90000bd6, 0xa002ea35, 0x9800017b,
            0x84b04200, 0xa594560c, 0xa5cc0002, 0xa22001c6,
            0x8cac07b2, 0x9394054c, 0x8c7cca04, 0xc4480000,
            0x68029013, 0xac000000, 0xa8018de8, 0xf8029c00,
        ],
        blend_resolved: [
            0x780c011f, 0xb8080352, 0xb11c0320, 0xae900111,
            0xe800012b, 0xedcc0000, 0xf002a600, 0x8eeab45c,
            0x90000b4b, 0x9802eaf3, 0x9de05a0c, 0x9fec01c8,
            0x92f40513, 0xbc500200, 0xa4000200, 0xf0029c00,
        ],
    },
    Metal {
        format: PixelFormat::RGB9E5Float,
        samples: 4,
        stored: &[
            0x880001b3, 0x880001b3, 0x880001b3, 0x880001b3,
            0xb6740016, 0xb6740016, 0x8dd07600, 0x8dd07600,
            0x98015d13, 0x9017b000, 0x9f30f804, 0xfffc0000,
            0xb8030e00, 0x9c981000, 0x86401c00, 0x80180156,
            0xa0000304, 0xa0000304, 0xa0000304, 0xa0000304,
            0x87506000, 0x87506000, 0x8802d26c, 0x8802d26c,
            0x9444078b, 0x9f3c0000, 0x7b360800, 0xc7d40000,
            0x3925f910, 0xc44c0000, 0x7409b001, 0xb4681a00,
            0xb803ac00, 0xb803ac00, 0xb803ac00, 0xb803ac00,
            0xc0022d27, 0xc0022d27, 0xb00003a2, 0xb00003a2,
            0x99f30432, 0xbe540000, 0x90001960, 0x60e805cd,
            0xf803fe00, 0xade001a7, 0xa6140400, 0xc7122200,
            0x9812140e, 0x9812140e, 0x9812140e, 0x9812140e,
            0x98040159, 0x98040159, 0xb803b400, 0xb803b400,
            0xa0003d93, 0xfffc0000, 0xa803f80b, 0x3936c868,
            0x5864197c, 0x7800011c, 0x512be36b, 0xb4681a00,
        ],
        resolved: [
            0x880001b3, 0xaea40416, 0xeffc0000, 0xa85b1003,
            0xa0000304, 0x83ab026c, 0xbc100003, 0xb5680600,
            0xb803ac00, 0xb8022d90, 0xae74320e, 0xf00a0200,
            0x9812140e, 0xb003b416, 0xeffc0200, 0xa4681a02,
        ],
        loaded: [
            0x880001b3, 0xaea40416, 0xeffc0000, 0xa85b1003,
            0xa0000304, 0x83ab026c, 0xbc100003, 0xb5680600,
            0xb803ac00, 0xb8022d90, 0xae74320e, 0xf00a0200,
            0x9812140e, 0xb003b416, 0xeffc0200, 0xa4681a02,
        ],
        blended: &[
            0xa873ee1e, 0x8814cfb3, 0xb8466c08, 0xa052d638,
            0xe80a2800, 0xf803ae00, 0xb026f400, 0xb80f2800,
            0x900671ec, 0x900e1200, 0x9d42770e, 0xfe080000,
            0xb806bc00, 0xf0031000, 0xacccf800, 0x8014010d,
            0xa0000304, 0xa0002911, 0xa0003113, 0xa0004319,
            0xad500800, 0xa5940a00, 0x88069a59, 0x880a704b,
            0x932fe67c, 0x9b690d47, 0xb0039200, 0xbf8c0000,
            0xbe300000, 0xbb4e2e00, 0xc6b2a200, 0x977c3600,
            0xac1aae00, 0xb82f2800, 0xabf71a00, 0xb852c000,
            0xa8022123, 0xa802cf7f, 0xb0000177, 0xa80001c2,
            0xf806f802, 0xafb80000, 0xa0080349, 0x98000195,
            0xe8021a00, 0xacb801dd, 0xaa17a282, 0xc5559c1c,
            0xa004c322, 0xa004c12e, 0xa0048ff1, 0x980db391,
            0xb8026c16, 0xb8022c16, 0xb9db4800, 0xc5809c00,
            0xf002bc00, 0xff200000, 0xa8027a07, 0x780001a1,
            0x60300b2c, 0xa5380005, 0x70067008, 0xa79c8278,
        ],
        blend_resolved: [
            0xa86fc61d, 0xf0021e00, 0xee080000, 0xe0071600,
            0xa0002910, 0xa40c2e05, 0xafddfa18, 0xbdc9de00,
            0xb143ae00, 0xa8013dd5, 0xe806f802, 0xd8322a03,
            0xa004bb43, 0xbb3a460b, 0xef215e00, 0x9e6c463f,
        ],
    },
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

    /// A channel's value: `RGB9E5Float`'s `2^u`, `u` uniform over [-12, 9], times a uniform
    /// mantissa, a few of them zero, negative or past the format's largest; another format's
    /// uniform over [-0.05, 1.05].
    fn channel(&mut self, pf: PixelFormat) -> f32 {
        if pf != PixelFormat::RGB9E5Float {
            return (-0.05 + 1.1 * self.next()) as f32;
        }
        match self.next() {
            u if u < 0.04 => 0.0,
            u if u < 0.06 => -1.5,
            u if u < 0.08 => 70000.0,
            _ => {
                let e = -12.0 + 21.0 * self.next();
                (e.floor().exp2() * (1.0 + self.next())) as f32
            }
        }
    }
}

/// The probe's values of its first [`PIXELS`] pixels for `pf` at `samples`, each pixel's samples
/// in turn: the fill's, and the blend's source colors. Each set draws every one of its 256 pixels
/// in turn: a pixel `p` holds one value in every sample where `p % 4 == 0`, two (the first half's
/// and the rest's) where `p % 4 == 1`, and its own value in each sample otherwise; then each blend
/// source's alpha is drawn anew, uniform over [0, 1].
fn inputs(pf: PixelFormat, samples: u32) -> [Vec<[f32; 4]>; 2] {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let n = samples as usize;
    let mut set = || {
        let mut out = Vec::with_capacity(256 * n);
        for p in 0..256 {
            let distinct: Vec<[f32; 4]> = (0..n).map(|_| [0; 4].map(|_| rng.channel(pf))).collect();
            for s in 0..n {
                let from = match p % 4 {
                    0 => 0,
                    1 => usize::from(s >= n / 2),
                    _ => s,
                };
                out.push(distinct[from]);
            }
        }
        out
    };
    let mut fill = set();
    let mut blend = set();
    for source in &mut blend {
        source[3] = rng.next() as f32;
    }
    fill.truncate(PIXELS * n);
    blend.truncate(PIXELS * n);
    [fill, blend]
}

/// The probe's clear: a color no code holds exactly.
fn clear(pf: PixelFormat) -> ClearColor {
    let [r, g, b, a] = match pf {
        PixelFormat::RGB9E5Float => [0.1f32, 3.3, 70.7, 1.0],
        _ => [0.1, 0.3, 0.7, 0.45],
    };
    ClearColor::new(r.into(), g.into(), b.into(), a.into())
}

/// The unsigned format of a texel of `pf`, whose view reads its word, and the texel's bytes.
fn raw(pf: PixelFormat) -> (PixelFormat, usize) {
    match pf {
        PixelFormat::R8UnormSRGB => (PixelFormat::R8Uint, 1),
        PixelFormat::RGB9E5Float => (PixelFormat::R32Uint, 4),
        _ => (PixelFormat::R16Uint, 2),
    }
}

/// `RGB9E5Float`'s channels of `word`, red first.
fn shared_exponent(word: u32) -> [f32; 3] {
    let scale = ((word >> 27) as f32 - 24.0).exp2();
    [0, 9, 18].map(|shift| ((word >> shift) & 511) as f32 * scale)
}

/// The sRGB decode of an 8-bit code.
fn srgb_linear(code: u32) -> f32 {
    let c = code as f32 / 255.0;
    match c <= 0.04045 {
        true => c / 12.92,
        false => ((c + 0.055) / 1.055).powf(2.4),
    }
}

/// What `texture2d_ms<float>::read` returns for a sample of `pf` holding `word`, as Metal reads it,
/// and how far the device's own read may be from it: a packed field's code over its largest,
/// `RGB9E5Float`'s value exactly, and an sRGB code's linear value, the device's decode
/// (`PrecisionNote::SrgbTexelDecode`, for which Vulkan sets no precision): within `1e-3` of the
/// transfer where the device decodes through Metal's own texture unit (`metal_sampler`,
/// `VkProperty::MetalSampler`), within a code's `1 / 255` elsewhere (the RTX 5080 rounds its
/// decode of an `R8_SRGB` sample to a `1 / 256` step: 1.88e-3 from the transfer at most over the
/// lane's 29 reads of 0b556ddfd6 that a `1e-3` bound refused); a packed word's 5- or 6-bit field
/// within 0.18 of a code where the device's texture unit is not Metal's, which may widen it by
/// replicating its bits (`PrecisionNote::DeviceTexelConversion`'s bound: the RTX 5080 read
/// `A1BGR5Unorm`'s 5-bit 4 as 33 / 255, Metal 4 / 31); a channel the format lacks 0, alpha 1.
fn read(pf: PixelFormat, word: u32, metal_sampler: bool) -> ([f32; 4], f32) {
    match WordAttachment::layout(pf).unwrap() {
        WordAttachment::Rgb9e5 => {
            let [r, g, b] = shared_exponent(word);
            ([r, g, b, 1.0], 0.0)
        }
        WordAttachment::Srgb8 { .. } => {
            let within = match metal_sampler {
                true => 1e-3,
                false => 1.0 / 255.0,
            };
            ([srgb_linear(word & 255), 0.0, 0.0, 1.0], within)
        }
        w => {
            let value = |f: Option<metal2vulkan::metal::format::WordField>| {
                f.map_or(1.0, |f| {
                    let max = (1u32 << f.bits) - 1;
                    ((word >> f.shift) & max) as f32 / max as f32
                })
            };
            // The widest bound of a replicated 5- or 6-bit field, where the device's own.
            let replicated = (w.fields().into_iter().flatten())
                .filter(|f| !metal_sampler && matches!(f.bits, 5 | 6))
                .map(|f| 0.18 / ((1u32 << f.bits) - 1) as f32)
                .fold(1e-6, f32::max);
            (w.fields().map(value), replicated)
        }
    }
}

/// How far `got` is from Metal's `metal` in each of `pf`'s channels: in codes of a field, and for
/// `RGB9E5Float` in units of the larger value's last mantissa bit.
fn apart(pf: PixelFormat, got: u32, metal: u32) -> u32 {
    match WordAttachment::layout(pf).unwrap() {
        WordAttachment::Rgb9e5 => {
            let (g, m) = (shared_exponent(got), shared_exponent(metal));
            let unit = ((got.max(metal) >> 27) as f32 - 24.0).exp2();
            (0..3)
                .map(|c| ((g[c] - m[c]).abs() / unit).ceil() as u32)
                .max()
                .unwrap()
        }
        w => w
            .fields()
            .into_iter()
            .flatten()
            .map(|f| {
                let code = |x: u32| (x >> f.shift) & ((1u32 << f.bits) - 1);
                code(got).abs_diff(code(metal))
            })
            .max()
            .unwrap(),
    }
}

/// The pixels of the resolve `got` of `pf` that are Metal's `metal`; each further than `near`
/// (in [`apart`]'s units) a line of `wrong`.
fn resolve_against(
    pf: PixelFormat,
    what: &str,
    leg: &str,
    got: &[u32],
    metal: &[u32; PIXELS],
    near: u32,
    wrong: &mut Vec<String>,
) -> usize {
    let mut same = 0;
    for (px, (&got, &metal)) in got.iter().zip(metal).enumerate() {
        let off = apart(pf, got, metal);
        same += usize::from(off == 0);
        if off > near {
            wrong.push(format!(
                "{what}: {leg} resolve, pixel {px}: {got:#x} against Metal's {metal:#x}"
            ));
        }
    }
    same
}

struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("wsm_samples.ll")).unwrap();
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

    /// Whether the device decodes sRGB through Metal's own texture unit
    /// (`VkProperty::MetalSampler`: MoltenVK on Apple's GPU), so its decode is Metal's.
    fn metal_sampler(&self) -> bool {
        let profile = self.device.compile_device().profile();
        (profile.enabled().properties).contains(&VkProperty::MetalSampler)
    }

    /// Whether the device renders `pf` through the view of its word.
    fn word(&self, pf: PixelFormat) -> bool {
        WordAttachment::of(pf, self.device.compile_device().profile()).is_some()
    }

    /// A private texture of `pf`, `samples` deep, of `usage`; `None`, with a printed line, where
    /// the device refuses the count.
    fn texture(&self, pf: PixelFormat, samples: u32, usage: TextureUsage) -> Option<Texture> {
        let mut d = TextureDescriptor::texture_2d(pf, WIDTH, 1, false);
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = u64::from(samples);
        }
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        match self.device.new_texture(&d) {
            Ok(t) => Some(t),
            Err(Error::Unsupported(u)) if u.feature == Feature::TextureSampleCount => {
                println!("{}: {pf:?} at {samples} refused: {u:?}", self.device.name());
                None
            }
            Err(e) => panic!("{pf:?} at {samples}: {e}"),
        }
    }

    /// One pass over `ms`, as the probe's: a fill (cleared to the probe's clear, drawn with
    /// `colors`, each pixel's samples in turn, the last sample of every eighth pixel discarded), a
    /// load that draws nothing (`draw` `None`), or a load that draws every sample (a blend); stored
    /// and resolved into a new single-sampled texture, whose words it returns.
    fn pass(
        &self,
        ms: &Texture,
        draw: Option<(&RenderPipelineState, &[[f32; 4]])>,
        fill: bool,
    ) -> Vec<u32> {
        let pf = ms.pixel_format();
        let samples = ms.sample_count() as u32;
        let resolved = self
            .texture(pf, 1, TextureUsage::RENDER_TARGET)
            .expect("one sample");
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(ms.clone());
        a.resolve_texture = Some(resolved.clone());
        a.store_action = StoreAction::StoreAndMultisampleResolve;
        match fill {
            true => {
                a.load_action = LoadAction::Clear;
                a.clear_color = clear(pf);
            }
            false => a.load_action = LoadAction::Load,
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        if let Some((pipeline, colors)) = draw {
            let bytes: Vec<u8> = colors
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let values = buffer_with(&self.device, &bytes).unwrap();
            let shape = [WIDTH as u32, samples, u32::from(fill)]
                .map(u32::to_le_bytes)
                .concat();
            let shape = buffer_with(&self.device, &shape).unwrap();
            e.set_render_pipeline_state(pipeline).unwrap();
            e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
            e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        e.end_encoding().unwrap();
        let (_, texel) = raw(pf);
        let length = (texel * PIXELS) as u64;
        let out = buffer_with(&self.device, &vec![0; length as usize]).unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &resolved,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, 1, 1),
            &out,
            0,
            length,
            length,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{pf:?} at {samples}");
        read_buffer(&out, 0, length)
            .unwrap()
            .chunks(texel)
            .map(|t| t.iter().rev().fold(0, |w, &b| w << 8 | u32::from(b)))
            .collect()
    }

    /// Every sample of `t` through `kernel` (`wsm_read_f` or `wsm_read_u`), `per` bytes a sample.
    fn samples(&self, kernel: &str, t: &Texture, per: usize) -> Vec<u8> {
        let state = self
            .device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(self.function(kernel)),
                &PipelineOptions::default(),
            )
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
        let length = per * PIXELS * t.sample_count() as usize;
        let out = buffer_with(&self.device, &vec![0xee; length]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        encoder.set_texture(Some(t), 0).unwrap();
        // One threadgroup of whole simdgroups, the threads past the row returning: a threadgroup
        // of the row's 16 threads is a partial simdgroup, which a device whose simdgroups are its
        // subgroups runs on the kernel's emulated-simdgroups variant (announced), a route this
        // lane does not read.
        let width = u64::from(state.thread_execution_width());
        encoder
            .dispatch_threadgroups(
                Size::new(1, 1, 1),
                Size::new(WIDTH.div_ceil(width) * width, 1, 1),
            )
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        read_buffer(&out, 0, length as u64).unwrap()
    }

    /// Each sample's word (through `view`, an unsigned view of `ms`) and its value as a function
    /// reads it.
    fn read_samples(&self, ms: &Texture, view: &Texture) -> (Vec<u32>, Vec<[f32; 4]>) {
        let words = self
            .samples("wsm_read_u", view, 4)
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        let reads = self
            .samples("wsm_read_f", ms, 16)
            .chunks(16)
            .map(|v| {
                std::array::from_fn(|c| f32::from_le_bytes(v[4 * c..4 * c + 4].try_into().unwrap()))
            })
            .collect();
        (words, reads)
    }

    /// A render pipeline drawing the fixture's per-sample colors into `pf` at `samples`, blending
    /// source over where `blend`.
    fn pipeline(&self, pf: PixelFormat, samples: u32, blend: bool) -> RenderPipelineDescriptor {
        let mut c = ColorAttachmentDescriptor::new(pf);
        if blend {
            c.blending_enabled = true;
            c.rgb_blend_operation = BlendOperation::Add;
            c.alpha_blend_operation = BlendOperation::Add;
            c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            c.source_alpha_blend_factor = BlendFactor::SourceAlpha;
            c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            c.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
        let mut d = RenderPipelineDescriptor::new(self.function("wsm_vertex"));
        d.fragment = Some(self.function("wsm_fill"));
        d.color_attachments[0] = Some(c);
        d.raster_sample_count = samples;
        d
    }

    /// The legs over `metal`'s format and count, each answer against Metal's (`wrong` collects
    /// what is not): `None`, with a printed line, where the device lacks the count; else the kinds
    /// the legs' pipeline states' makings announce, and whether the blend was made.
    fn run(&self, metal: &Metal, wrong: &mut Vec<String>) -> Option<(BTreeSet<Degradation>, bool)> {
        let Metal {
            format: pf,
            samples,
            stored,
            resolved,
            loaded,
            blended,
            blend_resolved,
        } = *metal;
        let name = self.device.name();
        let usage = TextureUsage::RENDER_TARGET
            | TextureUsage::SHADER_READ
            | TextureUsage::PIXEL_FORMAT_VIEW;
        let ms = self.texture(pf, samples, usage)?;
        let d = self.pipeline(pf, samples, false);
        let pipeline = match self
            .device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
        {
            Ok(pipeline) => pipeline,
            Err(Error::Unsupported(u)) if u.feature == Feature::RasterSampleCount => {
                println!("{name}: {pf:?} at {samples} refused: {u:?}");
                return None;
            }
            Err(e) => panic!("{pf:?} at {samples}: {e}"),
        };
        let mut made = made_render_kinds(&self.device, &d);
        let word = self.word(pf);
        let what = format!(
            "{pf:?} at {samples} ({})",
            if word { "word" } else { "own format" }
        );
        let [fill_colors, blend_colors] = inputs(pf, samples);
        let view = ms.new_texture_view(raw(pf).0).unwrap();
        let fill = self.pass(&ms, Some((&pipeline, &fill_colors)), true);
        let (words, reads) = self.read_samples(&ms, &view);
        let load = self.pass(&ms, None, false);
        // A word's samples are Metal's words, read as Metal reads them; a format's own within the
        // device's conversion.
        let mut exact = [0; 3];
        for (i, ((&got, &metal), value)) in words.iter().zip(stored).zip(&reads).enumerate() {
            let at = format!(
                "pixel {} sample {}",
                i / samples as usize,
                i % samples as usize
            );
            let off = apart(pf, got, metal);
            exact[0] += usize::from(off == 0);
            if off > u32::from(!word) {
                wrong.push(format!(
                    "{what}: {at}: word {got:#x} against Metal's {metal:#x}"
                ));
            }
            let (expected, within) = read(pf, got, self.metal_sampler());
            let far = value
                .iter()
                .zip(expected)
                .any(|(&v, e)| (v - e).abs() > within * e.abs().max(1.0));
            if far {
                wrong.push(format!(
                    "{what}: {at}: read {value:?} of word {got:#x}, not {expected:?}"
                ));
            }
        }
        // The resolves: a word's `R8Unorm_sRGB` and `RGB9E5Float` Metal's, a packed word's within
        // a code of each field; a format's own within the device's average.
        let packed = matches!(
            WordAttachment::layout(pf),
            Some(WordAttachment::Packed16 { .. })
        );
        let rule = word && !packed;
        let mut resolves = [0; 3];
        let near = u32::from(!rule);
        resolves[0] = resolve_against(pf, &what, "fill", &fill, &resolved, near, wrong);
        resolves[1] = resolve_against(pf, &what, "load", &load, &loaded, near, wrong);
        // A blend over the loaded samples: Metal's within a code of each field, and a code further
        // for each code the sample it blends over lies from Metal's, where the device makes it. A
        // word's `R8Unorm_sRGB` and `RGB9E5Float` blend is not built: refused as the
        // device lacking the format, as a single-sampled one is.
        let b = self.pipeline(pf, samples, true);
        let blend = match self
            .device
            .new_render_pipeline_state(&b, &PipelineOptions::default())
        {
            Ok(blend) => Some(blend),
            Err(Error::Unsupported(u)) if u.feature == Feature::Format => {
                println!("{name}: {what}: blend refused: {u:?}");
                if word && packed {
                    wrong.push(format!("{what}: a packed word's blend refused: {u:?}"));
                }
                None
            }
            Err(e) => panic!("{what}: the blend: {e}"),
        };
        if let Some(blend) = &blend {
            made.extend(made_render_kinds(&self.device, &b));
            let got = self.pass(&ms, Some((blend, &blend_colors)), false);
            let (blended_words, _) = self.read_samples(&ms, &view);
            let under = words.iter().zip(stored).map(|(&w, &m)| apart(pf, w, m));
            for (i, ((&got, &metal), under)) in
                blended_words.iter().zip(blended).zip(under).enumerate()
            {
                let off = apart(pf, got, metal);
                exact[2] += usize::from(off == 0);
                if off > 1 + under {
                    wrong.push(format!(
                        "{what}: blend, pixel {} sample {}: word {got:#x} against Metal's \
                         {metal:#x}",
                        i / samples as usize,
                        i % samples as usize
                    ));
                }
            }
            resolves[2] = resolve_against(pf, &what, "blend", &got, &blend_resolved, 1, wrong);
        }
        println!(
            "WORD-SAMPLES-MSAA {name}: {what}: Metal's words in {} of {} samples, resolves \
             Metal's in {} (fill) and {} (load) of {PIXELS} pixels; {}",
            exact[0],
            stored.len(),
            resolves[0],
            resolves[1],
            match blend {
                Some(_) => format!(
                    "blended, Metal's words in {} samples and resolves in {} pixels",
                    exact[2], resolves[2]
                ),
                None => "the blend refused".to_owned(),
            }
        );
        Some((made, blend.is_some()))
    }
}

/// The formats' attachments a device withholds to hold each in its word's format.
fn withheld() -> BTreeSet<vk::Format> {
    METAL
        .iter()
        .filter_map(|m| direct_format(m.format))
        .collect()
}

/// Each format at 2 and 4 samples on the device, and on one withholding the four formats' own
/// attachments, against Metal's samples, reads, resolves and blends; the word resolve's
/// near-equivalent announced where a packed format's word resolves, beside the crate-owned
/// resolve's extra pass. The word's way runs at least once.
#[test]
fn multisampled_word_formats_store_read_resolve_and_blend_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let (mut drawn, mut words) = (0, 0);
    for withhold in [false, true] {
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |o| {
            if withhold {
                o.withheld_attachment_formats = withheld();
            }
        })
        .unwrap();
        let lane = Lane::new(device);
        let name = lane.device.name().to_owned();
        let mut wrong = Vec::new();
        let mut kinds = BTreeSet::new();
        let mut allowed = BTreeSet::from([Degradation::SharedInSystemMemory]);
        for metal in &METAL {
            let (pf, samples) = (metal.format, metal.samples);
            let word = lane.word(pf);
            println!(
                "{name}{}: {pf:?} at {samples}: {}",
                if withhold {
                    " (attachments withheld)"
                } else {
                    ""
                },
                if word {
                    "held in its word's format"
                } else {
                    "held in its own format"
                }
            );
            let Some((made, blended)) = lane.run(metal, &mut wrong) else {
                continue;
            };
            drawn += 1;
            words += usize::from(word);
            allowed.extend(made);
            // The reads of each sample's value (`wsm_read_f`), the device's conversion where its
            // texture unit is not Metal's: recorded for every such read, whatever the format and
            // however it is held (a packed format's own as much as its word's).
            if !lane.metal_sampler() {
                allowed.insert(Degradation::DeviceTexelConversion);
            }
            match word {
                true => {
                    kinds.insert(Degradation::ShaderResolve);
                    if matches!(
                        WordAttachment::layout(pf),
                        Some(WordAttachment::Packed16 { .. })
                    ) {
                        kinds.extend(WORD_RESOLVE);
                    }
                    // A blended word's samples, through the sample planes.
                    if blended {
                        allowed.insert(Degradation::SamplePlanes);
                    }
                    // An sRGB word's reads, the device's decode where it is not Metal's.
                    let srgb = matches!(
                        WordAttachment::layout(pf),
                        Some(WordAttachment::Srgb8 { .. })
                    );
                    if srgb && !lane.metal_sampler() {
                        allowed.insert(Degradation::SrgbTexelDecode);
                    }
                }
                false => allowed.extend(resolve_kind(&lane.device, samples)),
            }
        }
        assert!(
            wrong.is_empty(),
            "{name}: {} answers:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
        let kinds: Vec<Degradation> = kinds.into_iter().collect();
        assert_once_allowing(&messages.take_notices(), &kinds, &allowed);
        drop(lane);
        layer_reported_nothing_at_all(&messages);
    }
    assert!(drawn > 0, "no sample count drawn");
    assert!(words > 0, "no format held in its word's format");
}
