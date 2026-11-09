//! A fetch of a blended 16-bit float attachment through a two-lane output, on the device the cfg
//! selects, under the validation layer: the owned fixture `fbl_fetch_blend16_narrow`, in the shape
//! of a macOS 26 window-server pass. Color 1 is a `SIDE`² private `RG16Float` target blended
//! ([`BLENDS`]) with the destination color times the source plus zero, for color and alpha (the
//! window server's), or with one and one added for color and one and zero for alpha (the guest's
//! `con_fetch_coverage` pipeline's), under a mask of red and green; colors 0 and 2 are
//! `RGBA16Float` targets of mask none. Each function reads color 1 under
//! fast math and returns there, at pixel `i = SIDE · y + x`, a loaded source chosen by the fetched
//! red (`src[i + SIDE²]` where it exceeds 2, else `src[i]`): `fbl_narrow_half2` a `half2`,
//! `fbl_narrow_float2` a `float2`. One draw of the fullscreen triangle.
//!
//! Metal's answer (plain Metal on the M3 Max under macOS 27.0.1, our own MSL, every
//! blend state of the 677 and 1,350 sweeps into `R16Float` and `RG16Float` through `half` and
//! `float` outputs of one to three lanes, and the partial masks of `RG16Float`): an output of fewer
//! than four lanes blends as four whose missing lanes are zero, so here each channel is `h(d · s)`
//! (`h(s + d)` for one and one) for a `half2` output, rounded to `half` once, nearest even (`s` the
//! output's channel, `d` the texel's), and for a `float2` output the `float` product (or sum),
//! rounded once, then converted to `half` toward zero (one and one: 16384 of 16384 lanes of each
//! channel bit-exact, `half2` and `float2`, fast and safe math, the same and cross sweeps). The
//! texels are random `half`s of magnitude below 8, the sources random `half`s or `float`s of the
//! same range, each a signed zero at times: every product and sum is exact in `f64`.
//!
//! Where the draw announces none of [`super::NOTES`], every channel is Metal's. On a device that
//! fetches through the pixel interlock (RADV, ANV, NVIDIA, MoltenVK) the function computes Metal's
//! blend itself, exactly, announcing none of them, which is asserted. Through input attachments
//! (lavapipe) the device blends; what it announces and how many channels are Metal's is printed.
//! Each draw prints a `FETCH-BLEND16-NARROW` line, naming its blend. A device that has neither
//! form skips it (`fetches`).

use super::super::{fetches, form, library, through_inputs};
use super::{half_bits, half_value, Words, NOTES, SIDE, TEXELS};
use crate::runtime_lane::buffer_with;
use crate::runtime_lane::degrade_notices::counts;
use crate::runtime_lane::resolves::{lane_with, quiet, Lane};
use crate::runtime_lane::WAIT;
use metal2vulkan::metal::{BlendFactor, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, Origin, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};

/// A case: the function, and whether its output is a `float2`.
#[derive(Clone, Copy, Debug)]
struct Case {
    entry: &'static str,
    float: bool,
}

const CASES: [Case; 2] = [
    Case {
        entry: "fbl_narrow_half2",
        float: false,
    },
    Case {
        entry: "fbl_narrow_float2",
        float: true,
    },
];

/// Color 1's blend: its source and destination factors for color, then for alpha, added.
#[derive(Clone, Copy, Debug)]
struct Blend {
    name: &'static str,
    color: (BlendFactor, BlendFactor),
    alpha: (BlendFactor, BlendFactor),
}

/// The blends: the window server's product, and the guest's sum.
const BLENDS: [Blend; 2] = [
    Blend {
        name: "product",
        color: (BlendFactor::DestinationColor, BlendFactor::Zero),
        alpha: (BlendFactor::DestinationAlpha, BlendFactor::Zero),
    },
    Blend {
        name: "sum",
        color: (BlendFactor::One, BlendFactor::One),
        alpha: (BlendFactor::One, BlendFactor::Zero),
    },
];

/// The texels' `half` bits, two a texel, and the sources (two a pixel), as `f64`s.
fn data(case: Case) -> (Vec<u16>, Vec<[f64; 2]>) {
    let mut words = Words(0x2545_f491_4f6c_dd1d);
    let texels: Vec<u16> = (0..TEXELS * 2)
        .map(|_| half_bits(words.half(2), false))
        .collect();
    let sources = (0..TEXELS * 2)
        .map(|_| {
            [2, 2].map(|t| match case.float {
                true => f64::from(words.float(t)),
                false => words.half(t),
            })
        })
        .collect();
    (texels, sources)
}

/// Metal's `blend` of `s` over the texel `d`, as `half` bits.
fn metal(case: Case, blend: Blend, s: [f64; 2], d: [f64; 2]) -> [u16; 2] {
    let sum = blend.color.1 == BlendFactor::One;
    std::array::from_fn(|c| match (case.float, sum) {
        (true, false) => half_bits(f64::from(d[c] as f32 * s[c] as f32), true),
        (true, true) => half_bits(f64::from(d[c] as f32 + s[c] as f32), true),
        (false, false) => half_bits(d[c] * s[c], false),
        (false, true) => half_bits(d[c] + s[c], false),
    })
}

/// A private `SIDE`² `RG16Float` texture holding `texels`.
fn filled(lane: &Lane, texels: &[u16]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RG16Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = lane.device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = texels.iter().flat_map(|h| h.to_le_bytes()).collect();
    let source = buffer_with(&lane.device, &bytes).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        4 * SIDE,
        bytes.len() as u64,
        Size::new(SIDE, SIDE, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    texture
}

/// A private `SIDE`² `RGBA16Float` render target the draw leaves alone.
fn unwritten(lane: &Lane) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    lane.device.new_texture(&descriptor).unwrap()
}

/// Each draw's channels are Metal's where it announces none of [`NOTES`], and on a device that
/// fetches through the pixel interlock it announces none (the module docs).
#[test]
fn a_two_lane_output_into_a_blended_two_channel_fetch_stores_metals_blend() {
    let kinds = NOTES.map(Degradation::from_precision_note);
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = kinds.into_iter().collect();
    });
    if !fetches(&lane.device) {
        return;
    }
    let fixture = library(&lane, "fbl_fetch_blend16_narrow");
    let device = lane.device.name();
    let interlock = !through_inputs(&lane);
    let function = |name| {
        StageSource::Function(
            fixture
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    for (blend, case) in BLENDS
        .into_iter()
        .flat_map(|blend| CASES.map(|case| (blend, case)))
    {
        let what = format!("{} under the {} on {device}", case.entry, blend.name);
        let mut descriptor = RenderPipelineDescriptor::new(function("fbl_narrow_vertex"));
        descriptor.fragment = Some(function(case.entry));
        let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
        blended.blending_enabled = true;
        (
            blended.source_rgb_blend_factor,
            blended.destination_rgb_blend_factor,
        ) = blend.color;
        (
            blended.source_alpha_blend_factor,
            blended.destination_alpha_blend_factor,
        ) = blend.alpha;
        blended.write_mask = ColorWriteMask::RED.union(ColorWriteMask::GREEN);
        let mut none = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
        none.write_mask = ColorWriteMask::NONE;
        descriptor.color_attachments[0] = Some(none);
        descriptor.color_attachments[1] = Some(blended);
        descriptor.color_attachments[2] = Some(none);
        let state = lane
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap_or_else(|error| panic!("{what}: {error}"));
        let (texels, sources) = data(case);
        let target = filled(&lane, &texels);
        let bytes: Vec<u8> = match case.float {
            true => sources
                .iter()
                .flat_map(|s| s.map(|v| v as f32))
                .flat_map(f32::to_le_bytes)
                .collect(),
            false => sources
                .iter()
                .flat_map(|s| s.map(|v| half_bits(v, false)))
                .flat_map(u16::to_le_bytes)
                .collect(),
        };
        let src = buffer_with(&lane.device, &bytes).unwrap();
        let mut pass = RenderPassDescriptor::new();
        for (k, texture) in [(0, unwritten(&lane)), (2, unwritten(&lane))] {
            let color = &mut pass.color_attachments[k];
            color.texture = Some(texture);
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = StoreAction::DontCare;
        }
        let color = &mut pass.color_attachments[1];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let _ = messages.take_notices();
        lane.run(&pass, |_, encoder, _| {
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_fragment_buffer(Some(&src), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        })
        .unwrap();
        let announced = counts(&messages.take_notices());
        let noted: Vec<_> = kinds
            .iter()
            .filter(|k| announced.contains_key(*k))
            .collect();
        let read = lane.read(&target, (0, 0), 4, BlitOption::None);
        let got: Vec<u16> = read
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect();
        let mut wrong = Vec::new();
        for i in 0..TEXELS {
            let texel: [u16; 2] = got[2 * i..2 * i + 2].try_into().unwrap();
            let d: [f64; 2] = std::array::from_fn(|c| half_value(texels[2 * i + c]));
            let s = sources[if d[0] > 2.0 { i + TEXELS } else { i }];
            let want = metal(case, blend, s, d);
            if texel != want {
                wrong.push((i, texel, want));
            }
        }
        println!(
            "FETCH-BLEND16-NARROW {} | {} | {device} | through {} | Metal's {}/{} texels | \
             announced {noted:?}",
            case.entry,
            blend.name,
            form(&lane),
            TEXELS - wrong.len(),
            TEXELS,
        );
        if interlock {
            assert!(noted.is_empty(), "{what}: announced {noted:?}");
        }
        if noted.is_empty() {
            let shown = &wrong[..wrong.len().min(8)];
            assert!(
                wrong.is_empty(),
                "{what}: {} of {TEXELS} texels differ from Metal's, first (index, read, Metal's) \
                 {shown:x?}",
                wrong.len()
            );
        }
    }
    let _ = messages.take_notices();
    quiet(lane, &messages);
}
