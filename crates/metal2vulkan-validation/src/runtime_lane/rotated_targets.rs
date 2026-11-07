//! `ARGB8Unorm` render targets on the device the cfg selects, under the validation layer, against
//! the device's own `RGBA8Unorm` targets.
//!
//! Metal stores, masks and blends `ARGB8Unorm` as `RGBA8Unorm` with the bytes rotated, alpha
//! first (16 colors, every write mask, eleven blend states, byte for byte). The
//! runtime holds it in an `R8G8B8A8_UNORM` image of those bytes: the fragment function writes its
//! output rotated, and a blend whose color factors read Metal's source alpha reads it from a
//! second output of copies of it (`pipeline::rotated_blend`); under straight alpha on Apple's
//! blend, Metal's blue premultiplied in the image's alpha. Each case draws the owned fixture
//! `rt6_samples`'s full-screen `float4` fill (`rt6_fill_rgba32float`) of 65,536 pseudo-random
//! outputs ([`outputs`]: codes, ties at half a code, other values in `[0, 1]`, and some outside it;
//! the tie-heavy set, the one its Metal probes measured) over a 256 x 256 target of each format
//! loaded with the same pseudo-random texels (rotated for `ARGB8Unorm`), under one write mask and
//! blend state ([`CASES`]), and checks every `ARGB8Unorm` texel holds the `RGBA8Unorm` texel's
//! bytes rotated, as Metal's do. A blend no Vulkan factor gives on the rotated image (Metal's
//! destination alpha in a color factor) the function computes over the texel where the device
//! orders framebuffer fetch: its bytes must be the `RGBA8Unorm` target's rotated where the
//! device's blend is Apple's (`VkProperty::MetalColorWrites`), and elsewhere how many differ is
//! printed, the `RGBA8Unorm` target's blend being the device's own; a device that orders fetch in
//! neither form refuses it naming `FramebufferFetch`. A case the device refuses for a need of its
//! own (`dualSrcBlend`) is printed as refused. A differing case is printed with its first differing
//! texels on one `ARGB8-TARGETS-DIFFERS` line.

use super::buffer_with;
use super::resolves::{lane, quiet, Lane};
use super::two_plane_copies::Rng;
use crate::candidate::runtime_samples::{library, shape};
use metal2vulkan::device::VkProperty;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Error, LoadAction, PipelineOptions, Region, RenderPassDescriptor, ResourceOptions,
    StoreAction, Texture, TextureDescriptor,
};

const WIDTH: u64 = 256;
const HEIGHT: u64 = 256;
const COUNT: usize = (WIDTH * HEIGHT) as usize;

/// The blend color every pass sets.
const BLEND_COLOR: [f32; 4] = [0.2, 0.4, 0.6, 0.8];

/// One case: its name, the write mask, and the blend (`None`: off) as the color and alpha
/// factors and operations.
type Case = (
    &'static str,
    ColorWriteMask,
    Option<[(BlendFactor, BlendFactor, BlendOperation); 2]>,
);

/// The cases: the image answers the first five and the masked one in place; the rest read a
/// second output of copies of Metal's source alpha.
const CASES: [Case; 13] = {
    use BlendFactor as F;
    use BlendOperation as O;
    const ALL: ColorWriteMask = ColorWriteMask::ALL;
    [
        ("unblended", ALL, None),
        (
            "unblended_red_alpha",
            ColorWriteMask::RED.union(ColorWriteMask::ALPHA),
            None,
        ),
        (
            "multiply",
            ALL,
            Some([
                (F::DestinationColor, F::Zero, O::Add),
                (F::DestinationAlpha, F::Zero, O::Add),
            ]),
        ),
        (
            "source_color",
            ALL,
            Some([
                (F::SourceColor, F::OneMinusSourceColor, O::Add),
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        (
            "blend_alpha",
            ALL,
            Some([
                (F::BlendAlpha, F::OneMinusBlendAlpha, O::Add),
                (F::BlendAlpha, F::OneMinusBlendAlpha, O::Add),
            ]),
        ),
        (
            "max",
            ALL,
            Some([(F::One, F::One, O::Max), (F::One, F::One, O::Max)]),
        ),
        (
            "straight",
            ALL,
            Some([
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        (
            "straight_alpha_one",
            ALL,
            Some([
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
                (F::One, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        (
            "straight_red_green_alpha",
            ColorWriteMask::RED
                .union(ColorWriteMask::GREEN)
                .union(ColorWriteMask::ALPHA),
            Some([
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
                (F::One, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        (
            "premultiplied",
            ALL,
            Some([
                (F::One, F::OneMinusSourceAlpha, O::Add),
                (F::One, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        (
            "additive_alpha",
            ALL,
            Some([(F::SourceAlpha, F::One, O::Add), (F::Zero, F::One, O::Add)]),
        ),
        (
            "reverse_subtract_alpha",
            ALL,
            Some([
                (F::SourceAlpha, F::One, O::ReverseSubtract),
                (F::SourceAlpha, F::One, O::ReverseSubtract),
            ]),
        ),
        (
            "subtract_one_minus_alpha",
            ALL,
            Some([
                (F::One, F::OneMinusSourceAlpha, O::Subtract),
                (F::One, F::OneMinusSourceAlpha, O::Subtract),
            ]),
        ),
    ]
};

/// The 65,536 `float4` outputs, from seed 85: each channel a code's value (6 in 10), a tie at
/// half a code (2 in 10), a value in `[0, 1]` off the codes (1 in 10), or one of -0.25, 1.25, -0
/// and 1 (1 in 10). The Metal probes measured these outputs over [`destinations`].
fn outputs() -> Vec<[f32; 4]> {
    let mut r = Rng::new(85);
    let channel = |r: &mut Rng| -> f32 {
        match r.below(10) {
            0..=5 => r.below(256) as f32 / 255.0,
            6 | 7 => (r.below(255) as f32 + 0.5) / 255.0,
            8 => r.below(1 << 24) as f32 / (1 << 24) as f32,
            _ => [-0.25, 1.25, -0.0, 1.0][r.below(4) as usize],
        }
    };
    (0..COUNT)
        .map(|_| [(); 4].map(|_| channel(&mut r)))
        .collect()
}

/// The `RGBA8Unorm` texels the targets hold before each draw, from seed 8585.
fn destinations() -> Vec<[u8; 4]> {
    let mut r = Rng::new(8585);
    (0..COUNT)
        .map(|_| [(); 4].map(|_| r.below(256) as u8))
        .collect()
}

/// `ARGB8Unorm`'s bytes of an `RGBA8Unorm` texel: alpha first.
fn rotated([r, g, b, a]: [u8; 4]) -> [u8; 4] {
    [a, r, g, b]
}

/// A shared 2D render target of `pf` holding `texels`.
fn target(lane: &Lane, pf: PixelFormat, texels: &[[u8; 4]]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let t = lane
        .device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {pf:?} texture: {e}"));
    let bytes: Vec<u8> = texels.iter().flatten().copied().collect();
    t.replace_region(
        Region::new_2d(0, 0, WIDTH, HEIGHT),
        0,
        0,
        &bytes,
        4 * WIDTH,
        0,
    )
    .unwrap();
    t
}

/// One draw of `values` into `target`, loaded and stored, through a pipeline whose color 0 is
/// `color`, under [`BLEND_COLOR`]; the pipeline's error where the runtime refuses it.
fn draw(
    lane: &Lane,
    target: &Texture,
    values: &[u8],
    color: ColorAttachmentDescriptor,
) -> Result<(), Error> {
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
    pipeline.fragment = Some(function("rt6_fill_rgba32float"));
    let what = format!("{:?} under {color:?}", color.pixel_format);
    pipeline.color_attachments[0] = Some(color);
    let pipeline = lane
        .device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())?;
    let values = buffer_with(&lane.device, values).unwrap();
    let shape = shape(&lane.device, WIDTH as u32, 1).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    lane.run(&pass, |_, e, _| {
        e.set_render_pipeline_state(&pipeline).unwrap();
        let [r, g, b, a] = BLEND_COLOR;
        e.set_blend_color(r, g, b, a);
        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap_or_else(|e| panic!("{what}: the pass: {e}"));
    Ok(())
}

/// `pf`'s attachment under `case`.
fn attachment(pf: PixelFormat, (_, mask, blend): Case) -> ColorAttachmentDescriptor {
    let mut c = ColorAttachmentDescriptor::new(pf);
    c.write_mask = mask;
    if let Some([(sr, dr, rgb), (sa, da, alpha)]) = blend {
        c.blending_enabled = true;
        c.source_rgb_blend_factor = sr;
        c.destination_rgb_blend_factor = dr;
        c.rgb_blend_operation = rgb;
        c.source_alpha_blend_factor = sa;
        c.destination_alpha_blend_factor = da;
        c.alpha_blend_operation = alpha;
    }
    c
}

/// Every case's `ARGB8Unorm` target holds its `RGBA8Unorm` target's bytes rotated, as Metal's
/// does, and a blend of Metal's destination alpha in a color factor, which the function computes
/// over the texel, holds them too where the device's blend is Apple's.
#[test]
fn argb8_targets_hold_rgba8_texels_rotated() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let values: Vec<u8> = outputs()
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let dst = destinations();
    let dst_rotated: Vec<[u8; 4]> = dst.iter().copied().map(rotated).collect();
    let (mut checked, mut refused, mut wrong) = (0, 0, Vec::new());
    for case in CASES {
        let plain = target(&lane, PixelFormat::RGBA8Unorm, &dst);
        draw(
            &lane,
            &plain,
            &values,
            attachment(PixelFormat::RGBA8Unorm, case),
        )
        .unwrap_or_else(|e| panic!("{}: RGBA8Unorm: {e}", case.0));
        let held = target(&lane, PixelFormat::ARGB8Unorm, &dst_rotated);
        match draw(
            &lane,
            &held,
            &values,
            attachment(PixelFormat::ARGB8Unorm, case),
        ) {
            Ok(()) => {}
            // A device without what the rotation needs (dual-source blending) refuses the
            // pipeline as a need of its own.
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {
                println!("ARGB8-TARGETS {name}: {}: refused: {u:?}", case.0);
                refused += 1;
                continue;
            }
            Err(e) => panic!("{}: ARGB8Unorm: {e}", case.0),
        }
        let want: Vec<u8> = lane
            .read(&plain, (0, 0), 4, BlitOption::None)
            .chunks(4)
            .flat_map(|t| rotated([t[0], t[1], t[2], t[3]]))
            .collect();
        let got = lane.read(&held, (0, 0), 4, BlitOption::None);
        checked += 1;
        let differing: Vec<String> = (0..COUNT)
            .filter(|&i| got[4 * i..4 * i + 4] != want[4 * i..4 * i + 4])
            .take(4)
            .map(|i| {
                format!(
                    "texel {i}: {:02x?}, RGBA8Unorm's rotated {:02x?}",
                    &got[4 * i..4 * i + 4],
                    &want[4 * i..4 * i + 4]
                )
            })
            .collect();
        if !differing.is_empty() {
            wrong.push(format!("{}: {}", case.0, differing.join("; ")));
        }
    }
    // Metal's destination alpha in a color factor: no Vulkan factor reads it on the image, so the
    // function computes Metal's blend over the texel where the device orders framebuffer fetch,
    // and a device that orders it in neither form refuses the pipeline naming both orderings.
    let mut dst_alpha = ColorAttachmentDescriptor::new(PixelFormat::ARGB8Unorm);
    dst_alpha.blending_enabled = true;
    dst_alpha.source_rgb_blend_factor = BlendFactor::DestinationAlpha;
    dst_alpha.destination_rgb_blend_factor = BlendFactor::OneMinusDestinationAlpha;
    let mut plain_dst_alpha = dst_alpha;
    plain_dst_alpha.pixel_format = PixelFormat::RGBA8Unorm;
    let plain = target(&lane, PixelFormat::RGBA8Unorm, &dst);
    draw(&lane, &plain, &values, plain_dst_alpha)
        .unwrap_or_else(|e| panic!("a destination alpha blend: RGBA8Unorm: {e}"));
    let held = target(&lane, PixelFormat::ARGB8Unorm, &dst_rotated);
    let shader_blend = match draw(&lane, &held, &values, dst_alpha) {
        Ok(()) => {
            let want: Vec<u8> = lane
                .read(&plain, (0, 0), 4, BlitOption::None)
                .chunks(4)
                .flat_map(|t| rotated([t[0], t[1], t[2], t[3]]))
                .collect();
            let got = lane.read(&held, (0, 0), 4, BlitOption::None);
            let differing = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            let largest = got
                .iter()
                .zip(&want)
                .map(|(&g, &w)| g.abs_diff(w))
                .max()
                .unwrap_or(0);
            println!(
                "ARGB8-TARGETS {name}: destination alpha: {differing} bytes differ from \
                 RGBA8Unorm's rotated, the largest by {largest}"
            );
            Some(differing)
        }
        Err(Error::Unsupported(u)) if u.feature == Feature::FramebufferFetch => {
            println!("ARGB8-TARGETS {name}: destination alpha: refused: {u:?}");
            None
        }
        Err(e) => panic!("a destination alpha blend: ARGB8Unorm: {e}"),
    };
    for line in &wrong {
        println!("ARGB8-TARGETS-DIFFERS {name}: {line}");
    }
    println!(
        "ARGB8-TARGETS {name}: {checked} cases of {COUNT} texels checked, {refused} refused, {} \
         differ from RGBA8Unorm's rotated",
        wrong.len()
    );
    assert!(wrong.is_empty(), "{} cases differ", wrong.len());
    // Metal blends the format as `RGBA8Unorm` with its channels rotated, and where the device's
    // blend is Apple's (`VkProperty::MetalColorWrites`) its `RGBA8Unorm` target's blend is Metal's
    // too: there the function's blend holds those bytes exactly. Elsewhere the `RGBA8Unorm`
    // target's blend is the device's own and the function's Metal's, so the line above counts
    // what they differ by.
    let metals = lane
        .device
        .compile_device()
        .profile()
        .enabled()
        .properties
        .contains(&VkProperty::MetalColorWrites);
    if metals {
        assert_eq!(
            shader_blend,
            Some(0),
            "a destination alpha blend on Apple's blend"
        );
    }
    assert!(checked >= 6, "{checked} cases checked");
    quiet(lane, &messages);
}
