//! A linear downscale by 8 through a pixel-coordinate sampler (the consumer's r24 ask of
//! 2026-10-09: a texture drawn at an eighth of its size into a 256x256 target through a runtime
//! sampler of linear filters, no mip filter, clamp to edge and pixel coordinates) on the device
//! the cfg selects, under the validation layer, against plain Metal's answers, at each of
//! [`SIZES`].
//!
//! Metal's answers (our own probe, an Apple M4 under macOS 27.0.1, two
//! processes, identical): a `RGBA8Unorm` texture of xorshift32 bytes ([`texels`]) sampled
//! through a runtime sampler state of linear min and mag filters, no mip filter, clamp to edge on
//! every axis and `normalizedCoordinates` NO:
//! - by a kernel at a slot, at level 0, on three sets of the size's rectangle of coordinates
//!   ([`coordinates`]): the downscale's texel centres `8i + 4`, `8j + 4` (`grid`), one ulp off
//!   each (`ulp`), and a random fraction of the 8-texel cell past each (`frac`); each set's
//!   `float4` results hash ([`METAL`]). Metal answered each set alike through an argument
//!   buffer's normalized state of the same filters and modes at the coordinate snapped to its
//!   `1/256` cell, half up, divided by the size (the `norm` control);
//! - by a fragment function at the interpolated coordinate (implicit level), drawing the
//!   rectangle into the top left of a 256x256 `RGBA8Unorm` target cleared to transparent black,
//!   its texture coordinates eight times its texels (`render`): the target's bytes hash.
//!
//! The consumer's r25 found the pass's source a 1440x1080 texture: the sizes sweep that one,
//! 1080 rows under other widths that are not powers of two, and each axis apart (the
//! orchestrator, 2026-10-09). Plain Metal answered every size, and MoltenVK on an M4 and lavapipe
//! answered Metal's at every one.
//!
//! Each configuration runs through the runtime, which hands the device the pixel coordinate
//! divided by the texture's size; the device's sampler reads that normalized coordinate in its
//! own arithmetic, not at Metal's `1/256` cell of the pixel coordinate, which the stages record
//! (`PrecisionNote::PixelSamplerScaling`) and every device announces. Each kernel set is held to
//! the device's own `norm` control: every channel within [`CODES`] of `1/255` of the control's
//! (the widest is printed). Elsewhere than on a device with `VkProperty::MetalSampler` every
//! sample is also the device's sampler's (`PrecisionNote::DeviceSampling`, announced). Each
//! configuration's hash is printed beside Metal's, and one that is not Metal's prints the cells
//! that differ from the control, without failing.

use super::argument_linear_samples::kernel;
use super::degrade_notices::counts;
use super::depth_compares::{pipeline_of, run_held, Held, Through};
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, SamplerAddressMode, SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, Degradation, Device, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, ResourceOptions, SamplerState, SamplerStateOptions, StoreAction, Texture,
    TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// One size of the sweep: the texture's, and the downscaled rectangle's texels (`gw` by `gh`,
/// at most the target's), which sample its first `8 gw` by `8 gh` texels.
#[derive(Clone, Copy, Debug)]
struct Size {
    w: u64,
    h: u64,
    gw: usize,
    gh: usize,
}

impl Size {
    const fn new(w: u64, h: u64, gw: usize, gh: usize) -> Size {
        Size { w, h, gw, gh }
    }
}

/// The sizes (the module docs): r24's 1536x1152 texture drawn 180x135, r25's 1440x1080 one, 1080
/// rows under 1920, 1366 and 1000 texels, and each of 1440 and 1080 alone.
const SIZES: [Size; 7] = [
    Size::new(1536, 1152, 180, 135),
    Size::new(1440, 1080, 180, 135),
    Size::new(1920, 1080, 240, 135),
    Size::new(1366, 1080, 170, 135),
    Size::new(1000, 1080, 125, 135),
    Size::new(1440, 1152, 180, 144),
    Size::new(1536, 1080, 192, 135),
];

/// The target's size.
const TARGET: u64 = 256;

/// The widest difference of a kernel set's channel from the device's own `norm` control, in
/// `1/255` (the module docs): a linear weight one sub-texel step from Metal's on each axis, each
/// moving the value by at most `1/256` of the texels' range (measured: 1 on MoltenVK, 2 on
/// lavapipe).
const CODES: u32 = 2;

/// One size's answers: each configuration's name and hash.
type Answers = [(&'static str, &'static str); 4];

/// Plain Metal's answers by the texture's size (the module docs): our probe's for 1536x1152 and for
/// the other sizes (the same probe at each size, an Apple M4 under macOS 27.0.1, two
/// processes, identical; its pixel sampler equal to the snapped normalized control at every
/// size) for every size, 1536x1152's the same again.
const METAL: [((u64, u64), Answers); 7] = [
    (
        (1536, 1152),
        [
            (
                "grid",
                "a5a6cb0f7fd6c18020b95b55ecdbb589fef8b15512e6c87c63a47f4fe64f0b92",
            ),
            (
                "ulp",
                "a5a6cb0f7fd6c18020b95b55ecdbb589fef8b15512e6c87c63a47f4fe64f0b92",
            ),
            (
                "frac",
                "810ec665b6fef8901c943b832632d1ae5e7dc6e05b54e7536429d9420a9d5a14",
            ),
            (
                "render",
                "f9d2eaa5a424d942072902dbf72004c763224529198d327329f251a9597604bf",
            ),
        ],
    ),
    (
        (1440, 1080),
        [
            (
                "grid",
                "e15c34bb8f1ca197ae85f64c43ad4d18c866d8b9640cf478c8ce02e7e4cf7236",
            ),
            (
                "ulp",
                "e15c34bb8f1ca197ae85f64c43ad4d18c866d8b9640cf478c8ce02e7e4cf7236",
            ),
            (
                "frac",
                "fd36975bce4f3ede7a2c59afbd86c0691166679fc63791ba1cb086c99f3e2573",
            ),
            (
                "render",
                "abe2c739eacc170915c242dc0376e9c63ad0f8b48aefa4e32416d5fff6ba408f",
            ),
        ],
    ),
    (
        (1920, 1080),
        [
            (
                "grid",
                "48d428cb1852e5e54ac020269ad34f361f3816bfa1975384e1c1e3bef1b03387",
            ),
            (
                "ulp",
                "48d428cb1852e5e54ac020269ad34f361f3816bfa1975384e1c1e3bef1b03387",
            ),
            (
                "frac",
                "71d2afefba11442056055f137daa6eb529172022ea43bf43431f258eeb7181ed",
            ),
            (
                "render",
                "cfee7cea5485061bbb03f8d1f53800b6eba2d3c52686791d87410f3945d4115b",
            ),
        ],
    ),
    (
        (1366, 1080),
        [
            (
                "grid",
                "9857ff78b453810e4bcec887963d8251ef38d34500d278515a13ec047b4fc2cc",
            ),
            (
                "ulp",
                "9857ff78b453810e4bcec887963d8251ef38d34500d278515a13ec047b4fc2cc",
            ),
            (
                "frac",
                "caec2fd9f40845412cc4b24c5e57d6eeabcbefdb6699695dc4a600b506c2244b",
            ),
            (
                "render",
                "c2cc01de4f736391b2b097a696c2e83d329c4c9888070ef62c6f5de312b8aebe",
            ),
        ],
    ),
    (
        (1000, 1080),
        [
            (
                "grid",
                "753599b580fcc1231f00e0642effc44f37955f75a7f16bb7a9d6647809548337",
            ),
            (
                "ulp",
                "753599b580fcc1231f00e0642effc44f37955f75a7f16bb7a9d6647809548337",
            ),
            (
                "frac",
                "264ceee98d6179ad0db33d7a10d196da1c8db63673b939b97ad07d9f0dcb7716",
            ),
            (
                "render",
                "b24b5e66bc6a73a790bb4aac78c473f9208be11e8819689c0c5bad0831533f16",
            ),
        ],
    ),
    (
        (1440, 1152),
        [
            (
                "grid",
                "b99e5cd231a628f1fcf76fe1668db429c2827eb7733a363ef9a3a406bd83a88b",
            ),
            (
                "ulp",
                "b99e5cd231a628f1fcf76fe1668db429c2827eb7733a363ef9a3a406bd83a88b",
            ),
            (
                "frac",
                "aa690279b927f7b00810a4f688eba29d013e134b258b7730c08e07ef72343c0d",
            ),
            (
                "render",
                "adda4abc7241116dcf7e3d8cec6fa215df1dc61df284cda7662198f0955d80c2",
            ),
        ],
    ),
    (
        (1536, 1080),
        [
            (
                "grid",
                "67e124800a21073ed1ed42534aa41ad81d37a759b62cd5fc57134fbcaf7bc32b",
            ),
            (
                "ulp",
                "67e124800a21073ed1ed42534aa41ad81d37a759b62cd5fc57134fbcaf7bc32b",
            ),
            (
                "frac",
                "6c9baf4f13010e70a06f1a1c5a82f9328fd681892d2cabdee00e2f5feb6b9d8f",
            ),
            (
                "render",
                "0d9fec6d892a4d17e8cf6cfd29e3e4fee5d0c347312ff439781b3e1f9ed4fea6",
            ),
        ],
    ),
];

/// A xorshift32 step.
fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// The bytes of a texture of `size`: each the top byte of the next xorshift32 step from
/// `0x9e3779b9`, in rows of texels of four channels.
fn texels(size: Size) -> Vec<u8> {
    let mut state = 0x9e37_79b9;
    (0..size.w * size.h * 4)
        .map(|_| (xorshift(&mut state) >> 24) as u8)
        .collect()
}

/// The probe's set `set` of coordinates, `x` fastest (the module docs): `grid`, `ulp` (up where
/// `i + j`, and for `y` where `i`, is even, else down) or `frac` (`(r >> 8) * 2^-21` of
/// xorshift32 from `0x2545f491`, `x` then `y`), over the rectangle of `size`.
fn coordinates(set: &str, size: Size) -> Vec<[f32; 2]> {
    let mut state = 0x2545_f491;
    let mut cells = Vec::with_capacity(size.gw * size.gh);
    for j in 0..size.gh {
        for i in 0..size.gw {
            let (x, y) = (8.0 * i as f32 + 4.0, 8.0 * j as f32 + 4.0);
            cells.push(match set {
                "grid" => [x, y],
                "ulp" => [
                    match (i + j) % 2 {
                        0 => x.next_up(),
                        _ => x.next_down(),
                    },
                    match i % 2 {
                        0 => y.next_up(),
                        _ => y.next_down(),
                    },
                ],
                _ => {
                    let dx = (xorshift(&mut state) >> 8) as f32 * 2f32.powi(-21);
                    let dy = (xorshift(&mut state) >> 8) as f32 * 2f32.powi(-21);
                    [x + dx, y + dy]
                }
            });
        }
    }
    cells
}

/// The SHA-256 of every size's texels and of its three coordinate sets' bytes, the sizes in
/// `SIZES` order and each size's sets in `grid`, `ulp`, `frac` order.
const INPUTS_SHA: &str = "2e8d095059bdcbb66f74ac0c5f3c98d2dfa6f72c065c5fdfd56debbab9a397dd";

/// The inputs are the probes': every size's texture bytes and coordinate sets.
#[test]
fn the_inputs_are_the_probes() {
    let mut sha = sha2::Sha256::new();
    for size in SIZES {
        sha.update(texels(size));
        for set in ["grid", "ulp", "frac"] {
            for cell in coordinates(set, size) {
                sha.update(cell[0].to_le_bytes());
                sha.update(cell[1].to_le_bytes());
            }
        }
    }
    assert_eq!(crate::hash::hex_encode(sha.finalize()), INPUTS_SHA);
}

/// Metal's `1/256` cell of the pixel coordinate `x`, half up.
fn cell(x: f32) -> f32 {
    let y = x * 256.0;
    let fl = y.floor();
    (fl + if y - fl >= 0.5 { 1.0 } else { 0.0 }) / 256.0
}

/// A runtime sampler state of linear filters, clamp to edge on every axis, argument-buffer
/// support, of pixel coordinates unless `normalized`.
fn sampler(device: &Device, normalized: bool) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = SamplerMinMagFilter::Linear;
    d.mag_filter = SamplerMinMagFilter::Linear;
    d.s_address_mode = SamplerAddressMode::ClampToEdge;
    d.t_address_mode = SamplerAddressMode::ClampToEdge;
    d.r_address_mode = SamplerAddressMode::ClampToEdge;
    d.normalized_coordinates = normalized;
    let mut options = SamplerStateOptions::default();
    options.support_argument_buffers = true;
    device.new_sampler_state_with_options(&d, options).unwrap()
}

/// The vertex function `v` (each vertex's clip position and texture coordinate from buffer 0's
/// pair of `float4`s at its ID) and the fragment function `f` (texture 0 sampled through sampler
/// 0 at the interpolated coordinate). Hand-written AIR.
pub(super) const RENDER: &str = r#"define <{ <4 x float>, <2 x float> }> @v(i32 %vid, ptr addrspace(1) %b) {
  %i = zext i32 %vid to i64
  %j = shl i64 %i, 1
  %pp = getelementptr inbounds <4 x float>, ptr addrspace(1) %b, i64 %j
  %p = load <4 x float>, ptr addrspace(1) %pp, align 16
  %k = or i64 %j, 1
  %up = getelementptr inbounds <4 x float>, ptr addrspace(1) %b, i64 %k
  %u4 = load <4 x float>, ptr addrspace(1) %up, align 16
  %uv = shufflevector <4 x float> %u4, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %o0 = insertvalue <{ <4 x float>, <2 x float> }> undef, <4 x float> %p, 0
  %o1 = insertvalue <{ <4 x float>, <2 x float> }> %o0, <2 x float> %uv, 1
  ret <{ <4 x float>, <2 x float> }> %o1
}

define <4 x float> @f(<4 x float> %pos, <2 x float> %uv, ptr addrspace(1) %t, ptr addrspace(2) %s) {
  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %c = extractvalue { <4 x float>, i8 } %r, 0
  ret <4 x float> %c
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.vertex = !{!0}
!air.fragment = !{!7}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @v, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!3 = !{!"air.vertex_output", !"generated(2uvDv2_f)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"b"}
!7 = !{ptr @f, !8, !10}
!8 = !{!9}
!9 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!10 = !{!11, !12, !13, !14}
!11 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos", !"air.arg_unused"}
!12 = !{i32 1, !"air.fragment_input", !"generated(2uvDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!13 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!14 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The draw's six vertices for the rectangle of `size`: per vertex its clip position, then its
/// texture coordinate in `xy`.
fn vertices(size: Size) -> Vec<f32> {
    let (gw, gh) = (size.gw as f32, size.gh as f32);
    let corners = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ];
    let right = 2.0 * gw / TARGET as f32 - 1.0;
    let bottom = 1.0 - 2.0 * gh / TARGET as f32;
    corners
        .iter()
        .flat_map(|&(a, b): &(f32, f32)| {
            [
                if a == 0.0 { -1.0 } else { right },
                if b == 0.0 { 1.0 } else { bottom },
                0.0,
                1.0,
                a * 8.0 * gw,
                b * 8.0 * gh,
                0.0,
                0.0,
            ]
        })
        .collect()
}

/// The target's bytes after the draw of the rectangle of `size` through the pixel sampler
/// `state`.
fn render(device: &Device, texture: &Texture, state: &SamplerState, size: Size) -> Vec<u8> {
    let library = device
        .new_library_with_air(&llvm_assemble_text(RENDER).unwrap())
        .unwrap();
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("v"));
    descriptor.fragment = Some(function("f"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let pipeline = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, TARGET, TARGET, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&d).unwrap();
    let bytes: Vec<u8> = vertices(size)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let vertices = buffer_with(device, &bytes).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&vertices), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder.set_fragment_sampler_state(Some(state), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let mut out = vec![0; (4 * TARGET * TARGET) as usize];
    target
        .get_bytes(
            &mut out,
            4 * TARGET,
            0,
            Region::new_2d(0, 0, TARGET, TARGET),
            0,
            0,
        )
        .unwrap();
    out
}

/// One configuration's results beside Metal's.
struct Outcome {
    name: &'static str,
    /// Whether it is Metal's, where Metal's answer at its size is known.
    metal: Option<bool>,
    /// Its hash.
    hash: String,
    /// A kernel set's widest channel difference from the device's own `norm` control, in `1/255`
    /// rounded up (none for the draw).
    codes: Option<u32>,
    /// Its hash where it is not Metal's, its cells that differ from the device's own `norm`
    /// control (a kernel's set), and its rows' hashes (the draw's), to name where it differs.
    differs: Vec<String>,
}

/// Each configuration on `device` at `size` beside Metal's.
fn compare(device: &Device, size: Size) -> Vec<Outcome> {
    let (tw, th) = (size.w, size.h);
    let texture = device
        .new_texture(&TextureDescriptor::texture_2d(
            PixelFormat::RGBA8Unorm,
            tw,
            th,
            false,
        ))
        .unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, tw, th), 0, 0, &texels(size), 4 * tw, 0)
        .unwrap();
    let (pixel_state, norm_state) = (sampler(device, false), sampler(device, true));
    let make = |heap: bool, through: Through| {
        pipeline_of(
            device,
            &llvm_assemble_text(&kernel(heap, through, false, false)).unwrap(),
            "k",
        )
    };
    let (slot, argument) = (make(false, Through::Runtime), make(true, Through::Argument));
    let hash = |bytes: &[u8]| crate::hash::hex_encode(Sha256::digest(bytes));
    let known = METAL
        .iter()
        .find(|(at, _)| *at == (tw, th))
        .map(|(_, hashes)| hashes);
    let metal = |name: &str, got: &str| {
        known.map(|hashes| hashes.iter().any(|(n, hash)| *n == name && *hash == got))
    };
    let mut outcomes = Vec::new();
    for name in ["grid", "ulp", "frac"] {
        let cells = coordinates(name, size);
        let n = cells.len();
        let pixel: Vec<[f32; 4]> = cells.iter().map(|&[x, y]| [x, y, 0.0, 0.0]).collect();
        let snapped: Vec<[f32; 4]> = cells
            .iter()
            .map(|&[x, y]| [cell(x) / tw as f32, cell(y) / th as f32, 0.0, 0.0])
            .collect();
        let mut out = run_held(
            device,
            (&texture, Held::Slot, Some(&pixel_state)),
            &slot,
            &pixel,
            1,
        );
        out.truncate(4 * n);
        let mut norm = run_held(
            device,
            (&texture, Held::Argument(Some(&norm_state)), None),
            &argument,
            &snapped,
            1,
        );
        norm.truncate(4 * n);
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        let got = hash(&bytes);
        let is_metal = metal(name, &got);
        let mut differs = Vec::new();
        let wrong: Vec<usize> = (0..n)
            .filter(|&j| {
                out[4 * j..4 * j + 4]
                    .iter()
                    .zip(&norm[4 * j..4 * j + 4])
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            })
            .collect();
        let codes = (0..4 * n)
            .map(|c| ((out[c] - norm[c]).abs() * 255.0).ceil() as u32)
            .max();
        if !wrong.is_empty() {
            differs.push(format!(
                "{} of {n} cells differ from the device's own norm control",
                wrong.len()
            ));
        }
        for &j in wrong.iter().take(6) {
            differs.push(format!(
                "{:?}: {:?}, the norm control {:?}",
                cells[j],
                &out[4 * j..4 * j + 4],
                &norm[4 * j..4 * j + 4]
            ));
        }
        outcomes.push(Outcome {
            name,
            metal: is_metal,
            hash: got,
            codes,
            differs,
        });
    }
    let target = render(device, &texture, &pixel_state, size);
    let got = hash(&target);
    let is_metal = metal("render", &got);
    let mut differs = Vec::new();
    if is_metal == Some(false) {
        // The rows of the rectangle, each its hash's first 16 digits.
        let row = (4 * TARGET) as usize;
        differs.extend(
            target
                .chunks(row)
                .take(size.gh)
                .enumerate()
                .map(|(y, r)| format!("row {y} {}", &hash(r)[..16])),
        );
    }
    outcomes.push(Outcome {
        name: "render",
        metal: is_metal,
        hash: got,
        codes: None,
        differs,
    });
    outcomes
}

/// Whether `device` samples through Metal's own sampler (`VkProperty::MetalSampler`).
fn metal_sampler(device: &Device) -> bool {
    device
        .compile_device()
        .profile()
        .enabled()
        .properties
        .contains(&VkProperty::MetalSampler)
}

/// Every device announces `pixel-sampler-scaling`, and each kernel set at each size is within
/// [`CODES`] of the device's own `norm` control (the module docs); `device-sampling` is announced
/// exactly where the device's sampler is not Metal's. Every configuration's hash is printed beside
/// Metal's where they are known, and the configurations that are not Metal's are counted.
#[test]
fn a_pixel_sampler_downscale_answers_metal_s_texels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (outcomes, metal_sampler) = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        eprintln!("pixel downscales on {}", device.name());
        let outcomes: Vec<(Size, Vec<Outcome>)> = SIZES
            .iter()
            .map(|&size| (size, compare(&device, size)))
            .collect();
        (outcomes, metal_sampler(&device))
    };
    layer_reported_nothing_at_all(&messages);
    let announced: BTreeSet<Degradation> = counts(&messages.take_notices()).into_keys().collect();
    eprintln!("announced: {announced:?}");
    let mut wrong = Vec::new();
    let mut beyond = Vec::new();
    for (size, outcomes) in &outcomes {
        for o in outcomes {
            let verdict = match o.metal {
                Some(true) => "Metal's",
                Some(false) => "not Metal's",
                None => "unmeasured",
            };
            eprintln!(
                "pixel downscale {}x{} {}: {verdict} hash {}, widest {:?} of 1/255 from the \
                 control{}",
                size.w,
                size.h,
                o.name,
                o.hash,
                o.codes,
                o.differs
                    .iter()
                    .map(|d| format!("\n  {d}"))
                    .collect::<String>()
            );
            if o.metal == Some(false) {
                wrong.push(format!("{}x{} {}", size.w, size.h, o.name));
            }
            if o.codes.is_some_and(|c| c > CODES) {
                beyond.push(format!("{}x{} {}: {:?}", size.w, size.h, o.name, o.codes));
            }
        }
    }
    assert!(
        announced.contains(&Degradation::PixelSamplerScaling),
        "{announced:?}"
    );
    assert_eq!(
        announced.contains(&Degradation::DeviceSampling),
        !metal_sampler,
        "device-sampling exactly where the device's sampler is not Metal's: {announced:?}"
    );
    eprintln!(
        "PIXEL-DOWNSCALE-DIFFERS: {} configurations are not Metal's: {wrong:?}",
        wrong.len()
    );
    assert!(beyond.is_empty(), "beyond {CODES} of 1/255: {beyond:?}");
}
