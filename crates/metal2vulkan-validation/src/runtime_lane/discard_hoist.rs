//! A simdgroup sum before a discard of whole quads, in the thirteen shapes measured for where
//! Metal's compiler places it, on the device the cfg selects, under the validation layer.
//!
//! The owned fixture `fragment_discard_hoist` draws Metal's fullscreen triangle over an 8x4
//! `RGBA32Sint` target cleared to -7, keeping one quad (`kept_quads` 1, pixels (0..2, 0..2)) and
//! discarding every other quad whole. Each kept thread writes `(before > after, after - kept,
//! (before - after) & 3, before < after)` from its simdgroup's lane counts before the discard and
//! after it: values that hold however a device packs quads into simdgroups.
//!
//! Metal's answer, as Apple M4 and M3 Max give it (from the fixture's AIR and from source, with and
//! without the debug layer), is `(0, 0, 0, 0)` for every `hoisted_*` shape (Metal's compiler moves
//! the sum past the discard, so it counts only the kept quad's lanes) and `(1, 0, 0, 0)` for every
//! `counted_*` shape (the sum counts the dropped quads that share the kept quad's simdgroup).
//!
//! The compile layer moves the sum where Metal's compiler moves it and keeps it ahead of the
//! discard where Metal does, so a device whose fragment subgroup holds the kept quad with at
//! least one dropped quad gives Metal's answer for every shape whose placement is known. Two are
//! not: `counted_call_use` (arithmetic reads the sum between) and `hoisted_nested` (a nested
//! discard, which Metal placed both ways), where the module keeps the sum ahead of the discard
//! and records `QuadsCountedBeforeDiscard`: `hoisted_nested` answers `(1, 0, 0, 0)` there, the
//! noted residue. A device whose fragment subgroups hold fewer than two quads
//! (`AnswerGap::FragmentSubgroupUnderTwoQuads`, decided from its description as C3 decides it)
//! cannot place a dropped quad beside the kept one, so every shape answers `(0, 0, 0, 0)` there:
//! a valid Metal answer for a partly occupied simdgroup, and that gap. The answer is announced:
//! the sums run over the device's own fragment subgroups, so every shape records
//! `PrecisionNote::FragmentSimdgroupGrouping` and the runtime announces
//! `Degradation::FragmentSimdgroupGrouping` wherever the driver does not run the fragment on
//! Metal, which covers every device with the gap; the test asserts it. `RECORDED` holds a measured
//! venue's answers where they differ from these.
//!
//! Measured: RADV (Radeon 780M, subgroup 64), RTX 3090 and RTX 5080 (32), ANV
//! (Iris Xe RPL-P, 32), and lavapipe x86 (8) give these answers, the noted residue included;
//! lavapipe arm64 (fragment subgroup 4, the gap) answers `(0, 0, 0, 0)` for all thirteen; MoltenVK
//! on Apple M4 answers `hoisted_nested` as Metal does (`RECORDED`).

use super::buffer_with;
use super::degrade_notices::counts;
use super::resolves::{lane, quiet, Lane};
use crate::air_library::AirLibrary;
use crate::c3::AnswerGap;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::DriverId;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, StoreAction, Texture, TextureDescriptor,
};

/// The target's width and height.
const W: u64 = 8;
const H: u64 = 4;

/// The clear value of every channel.
const CLEAR: i32 = -7;

/// The quads the draw keeps: bit `(y / 2) * 4 + x / 2`, the quad of pixels (0..2, 0..2) alone.
const KEPT_QUADS: u32 = 1;

/// What a kept thread's simdgroup counted ahead of the discard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answer {
    /// `(0, 0, 0, 0)`: the lanes that survive the discard alone.
    Survivors,
    /// `(1, 0, 0, 0)`: the dropped quads that share the simdgroup too.
    DroppedQuads,
}

/// Each fragment function and Metal's answer (the module docs).
const SHAPES: [(&str, Answer); 13] = [
    ("hoisted_known", Answer::Survivors),
    ("counted_dependent", Answer::DroppedQuads),
    ("hoisted_const", Answer::Survivors),
    ("hoisted_device_read", Answer::Survivors),
    ("hoisted_written_read", Answer::Survivors),
    ("hoisted_sample", Answer::Survivors),
    ("hoisted_texture_read", Answer::Survivors),
    ("hoisted_loop", Answer::Survivors),
    ("hoisted_call", Answer::Survivors),
    ("counted_call_use", Answer::DroppedQuads),
    ("hoisted_nested", Answer::Survivors),
    ("counted_store", Answer::DroppedQuads),
    ("counted_atomic", Answer::DroppedQuads),
];

/// The shapes whose placement by Metal's compiler the module does not know: it keeps the sum
/// ahead of the discard and records `QuadsCountedBeforeDiscard`.
const NOTED: [&str; 2] = ["counted_call_use", "hoisted_nested"];

/// A measured venue's answers where they differ from [`expected`]'s, by its driver and reported
/// subgroup size. MoltenVK on Apple M4 answers the nested shape as Metal
/// does: the module keeps the sum ahead of the discard, and Metal's compiler, which compiles
/// MoltenVK's MSL, moves it past the discard as it does from the fixture's own AIR.
const RECORDED: &[(DriverId, u32, &str, Answer)] =
    &[(DriverId::MOLTENVK, 32, "hoisted_nested", Answer::Survivors)];

/// The answer the module gives for `shape` (Metal's answer `metals`) on a device with or without
/// the subgroup gap.
fn expected(shape: &str, metals: Answer, gap: bool) -> Answer {
    if gap {
        Answer::Survivors
    } else if NOTED.contains(&shape) {
        Answer::DroppedQuads
    } else {
        metals
    }
}

/// The fixture's pipeline of fragment function `fragment` on `lane`'s device, drawing an
/// `RGBA32Sint` target.
fn pipeline(lane: &Lane, fragment: &str) -> RenderPipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("fragment_discard_hoist.ll");
    let library = lane
        .device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("discard_hoist_full")));
    descriptor.fragment = Some(StageSource::Function(function(fragment)));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Sint));
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A texture of `pf`, `width` by `height`, of `usage`, private or shared by `options`.
fn texture(
    lane: &Lane,
    pf: PixelFormat,
    (width, height): (u64, u64),
    usage: TextureUsage,
    options: ResourceOptions,
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = options;
    lane.device.new_texture(&descriptor).unwrap()
}

/// The texels, row by row, after one draw of `pipeline`.
fn drawn(lane: &Lane, pipeline: &RenderPipelineState) -> Vec<[i32; 4]> {
    let target = texture(
        lane,
        PixelFormat::RGBA32Sint,
        (W, H),
        TextureUsage::RENDER_TARGET,
        ResourceOptions::private(),
    );
    // The sampled and read texture: one texel of zeros, written by the host.
    let sampled = texture(
        lane,
        PixelFormat::RGBA32Float,
        (1, 1),
        TextureUsage::SHADER_READ,
        ResourceOptions::shared(),
    );
    sampled
        .replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[0; 16], 16, 0)
        .unwrap();
    let sampler = lane
        .device
        .new_sampler_state(&SamplerDescriptor::new())
        .unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    let clear = f64::from(CLEAR);
    attachment.clear_color = ClearColor::new(clear, clear, clear, clear);
    attachment.store_action = StoreAction::Store;
    lane.run(&descriptor, |lane, encoder, held| {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let buffers = [
            KEPT_QUADS.to_le_bytes().to_vec(),
            0i32.to_le_bytes().to_vec(),
            vec![0; 256],
            vec![0; 1024],
            vec![0; 64],
        ];
        for (index, bytes) in buffers.iter().enumerate() {
            let buffer = buffer_with(&lane.device, bytes).unwrap();
            encoder
                .set_fragment_buffer(Some(&buffer), 0, index as u32)
                .unwrap();
            held.push(buffer);
        }
        encoder.set_fragment_texture(Some(&sampled), 0).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&sampler), 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap();
    lane.read(&target, (0, 0), 16, BlitOption::None)
        .as_chunks::<16>()
        .0
        .iter()
        .map(|texel| {
            let word = |i: usize| i32::from_le_bytes(texel[4 * i..4 * i + 4].try_into().unwrap());
            [word(0), word(1), word(2), word(3)]
        })
        .collect()
}

/// The answer of a draw's texels: the kept quad's four agree on one, and every other texel holds
/// the clear value. `Err`: a texel outside both.
fn answer(texels: &[[i32; 4]]) -> Result<Answer, String> {
    let mut found = None;
    for (index, &texel) in texels.iter().enumerate() {
        let (x, y) = (index as u64 % W, index as u64 / W);
        let kept = (KEPT_QUADS >> ((y / 2) * 4 + x / 2)) & 1 != 0;
        let this = match texel {
            [0, 0, 0, 0] if kept => Answer::Survivors,
            [1, 0, 0, 0] if kept => Answer::DroppedQuads,
            [CLEAR, CLEAR, CLEAR, CLEAR] if !kept => continue,
            _ => return Err(format!("({x}, {y}) wrote {texel:?}")),
        };
        if found.is_some_and(|f| f != this) {
            return Err(format!(
                "({x}, {y}) answers {this:?}, another kept texel {found:?}"
            ));
        }
        found = Some(this);
    }
    found.ok_or_else(|| "no kept texel was written".into())
}

/// The texels of a draw, one line a row.
fn rows(texels: &[[i32; 4]]) -> String {
    texels
        .as_chunks::<{ W as usize }>()
        .0
        .iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>()
        .join("\n  ")
}

/// Each shape answers as the module places its sum on the device: Metal's answer where its
/// placement is known and the device's fragment subgroup holds two quads, the noted residue
/// where it is not known, the survivors alone under the subgroup gap (`AnswerGap`).
///
/// The gap's answer is announced, not silent: the sums run over the device's fragment subgroups,
/// so every one of these draws records `PrecisionNote::FragmentSimdgroupGrouping` and the runtime
/// announces `Degradation::FragmentSimdgroupGrouping` on every device whose driver does not run
/// the fragment on Metal — which every device with the gap is, since a device that does
/// (`DriverDeviation::MetalFragmentSimdgroups`) runs Metal's 32-lane simdgroups. The test asserts
/// the notice where the facts predict it and its absence where they do not, so the gap and the
/// notice are tied by this run and not by two descriptions of one fact.
#[test]
fn a_sum_before_a_discard_answers_as_metal_places_it() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let description = &lane.device.properties().description;
    let size = description.subgroup.size;
    let gap = AnswerGap::FragmentSubgroupUnderTwoQuads
        .has(description, lane.device.compile_device().profile());
    for (shape, metals) in SHAPES {
        let texels = drawn(&lane, &pipeline(&lane, shape));
        let answer = answer(&texels)
            .unwrap_or_else(|why| panic!("{shape} on {name}: {why}\n  {}", rows(&texels)));
        let recorded = description.driver_id.and_then(|driver| {
            RECORDED
                .iter()
                .find(|&&(d, s, r, _)| d == driver && s == size && r == shape)
                .map(|&(_, _, _, answer)| answer)
        });
        let expected = recorded.unwrap_or_else(|| expected(shape, metals, gap));
        eprintln!(
            "discard hoist {shape} on {name} (subgroup {size}, gap {gap}): {answer:?}, Metal's \
             {metals:?}, expected {expected:?}"
        );
        assert_eq!(
            answer,
            expected,
            "{shape} on {name} (subgroup {size}, gap {gap})\n  {}",
            rows(&texels)
        );
    }
    // The notice the draws' sums announce, and the gap it covers: a device whose driver runs the
    // fragment on Metal has Metal's own simdgroups and announces none, and no such device has the
    // gap, so the gap's answer always carries the notice.
    let metal_simdgroups = lane
        .device
        .compile_device()
        .profile()
        .takes_driver_deviation(DriverDeviation::MetalFragmentSimdgroups);
    let announced = counts(&messages.take_notices());
    let grouping = announced.contains_key(&Degradation::FragmentSimdgroupGrouping);
    assert_eq!(
        grouping, !metal_simdgroups,
        "{name} (subgroup {size}, gap {gap}, Metal's simdgroups {metal_simdgroups}) announced \
         {announced:?}"
    );
    assert!(
        !gap || grouping,
        "{name} answers the subgroup gap with no grouping notice: {announced:?}"
    );
    quiet(lane, &messages);
}
