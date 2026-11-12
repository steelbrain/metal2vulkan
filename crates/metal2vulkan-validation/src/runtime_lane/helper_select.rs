//! A texture write after a pick only quad helpers make, with no branch, on the device the cfg
//! selects, under the validation layer: a write is lost only where its fragment records
//! `PrecisionNote::HelperBranchDropsSimdgroupStores`.
//!
//! The owned fixture `rch_helper_select` (without fast math: every float operation is
//! `NoContraction`, which SPIRV-Cross writes through its `optnone` `spvFAdd`) draws the triangle
//! of `runtime_lane::guarded_effects` (its corners in pixels there; no pixel centre near an edge)
//! with no attachment over a 32x16 pass: its edges cut quads, whose helper invocations run beside
//! the covered pixels. Each fragment reads a 32x16 coverage texture, 1 at a covered pixel and 0
//! elsewhere, samples an 8x8 texture of `(0.0625, 0.0625, 0.0625, 0)` at LOD 0, and picks a sum
//! where its coverage is not 1, which only the helpers do, as a select: SPIRV-Cross writes it as a
//! `?:` whose arm calls `spvFAdd`. After it the fragment writes a 32x16 texture of [`SENTINEL`]
//! words at the pixel. A covered pixel writes what Metal does: `(1, 0.5, 0.25, 1) + (2^-11,
//! 2^-11, 2^-11, 0)`, the texel where the function samples at the picked coordinate, and the sum
//! in red where a constant block's flag picks it for every invocation. A helper's write has no
//! effect. Apple's compiler loses a simdgroup's texture writes after an `optnone` call only its
//! helpers make: on MoltenVK on an Apple GPU the `?:` arm and the right side of the `&&` are
//! fenced where the device's fragment stage shuffles (an identity shuffle follows the call,
//! census `fragment.helper-fence`), else record the note, their lost writes reported. The pick a
//! sample reads after it and the pick on a constant block's flag are left as they are. No write
//! may be lost where the note is not recorded, and none written outside the triangle.
//!
//! The fence's GPU time is printed for a 1024x1024 pass whose every fragment picks the sum (each
//! invocation, live ones too, runs the call and its fence), 32 draws a command buffer.

use super::buffer_with;
use super::guarded_effects::{bytes, covered, texture, unpack, words, CORNERS, H, SENTINEL, W};
use super::resolves::{lane, quiet, Lane};
use super::WAIT;
use crate::air_library::AirLibrary;
use metal2vulkan::device::DriverWorkaround;
use metal2vulkan::device::OpBits;
use metal2vulkan::metal::{PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{PrecisionNote, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk;
use metal2vulkan::Budget;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Library, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
};
use std::path::Path;

/// The sampled texture's side.
const TILE: u64 = 8;

/// Every texel of the sampled texture.
const NOISE: [f32; 4] = [0.0625, 0.0625, 0.0625, 0.0];

/// The fragment functions, each with whether its pick leaves the call where only helpers make it
/// with nothing tangled reading the sum after it (fenced on a device with the defect).
const CASES: [(&str, bool); 4] = [
    ("rch_select_arm", true),
    ("rch_select_and", true),
    ("rch_select_sampled", false),
    ("rch_select_uniform", false),
];

/// The timed pass's side.
const SIDE: u64 = 1024;

/// Draws in each timed command buffer, and timed command buffers.
const DRAWS: usize = 32;
const RUNS: usize = 5;

fn library(lane: &Lane) -> Library {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/rch_helper_select.ll");
    lane.device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// The pipeline of `fragment` after `rch_select_vertex`.
fn descriptor(library: &Library, fragment: &str) -> RenderPipelineDescriptor {
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("rch_select_vertex"));
    descriptor.fragment = Some(function(fragment));
    descriptor
}

fn pipeline(lane: &Lane, library: &Library, fragment: &str) -> RenderPipelineState {
    lane.device
        .new_render_pipeline_state(&descriptor(library, fragment), &PipelineOptions::default())
        .unwrap()
}

/// Whether the fragment stage of `fragment`'s pipeline records
/// `PrecisionNote::HelperBranchDropsSimdgroupStores`, and the calls it fences (its census's
/// `fragment.helper-fence`).
fn noted(lane: &Lane, library: &Library, fragment: &str) -> (bool, u32) {
    let compiled = lane
        .device
        .compile_device()
        .compile_render_pipeline(&descriptor(library, fragment), &Budget::standard())
        .unwrap();
    let fragment = compiled
        .stages
        .iter()
        .filter(|s| s.stage == vk::ShaderStage::FRAGMENT);
    let note = fragment.clone().any(|s| {
        s.precision_notes
            .contains(&PrecisionNote::HelperBranchDropsSimdgroupStores)
    });
    let fences = fragment
        .filter_map(|s| s.strategies.entries.get("fragment.helper-fence"))
        .map(|u| u.uses)
        .sum();
    (note, fences)
}

/// The vertex buffer of `corners`, in clip space.
fn corners(lane: &Lane, corners: &[[f32; 2]]) -> metal2vulkan_runtime::Buffer {
    let flat: Vec<f32> = corners.iter().flatten().copied().collect();
    buffer_with(&lane.device, &bytes(&words(&flat))).unwrap()
}

/// One draw of `fragment` over the 32x16 pass; the written texture's words.
fn draw(lane: &Lane, library: &Library, fragment: &str) -> Vec<u32> {
    let pipeline = pipeline(lane, library, fragment);
    let noise: Vec<u32> = (0..TILE * TILE).flat_map(|_| words(&NOISE)).collect();
    let noise = texture(lane, (TILE, TILE), TextureUsage::SHADER_READ, &noise);
    let coverage: Vec<u32> = (0..H)
        .flat_map(|y| (0..W).map(move |x| covered(x, y)))
        .flat_map(|c| words(&[if c { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0]))
        .collect();
    let coverage = texture(lane, (W, H), TextureUsage::SHADER_READ, &coverage);
    let written = texture(
        lane,
        (W, H),
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        &vec![SENTINEL; (4 * W * H) as usize],
    );
    let clip: Vec<[f32; 2]> = CORNERS
        .iter()
        .map(|[x, y]| {
            [
                (x / W as f64 * 2.0 - 1.0) as f32,
                (1.0 - y / H as f64 * 2.0) as f32,
            ]
        })
        .collect();
    let corners = corners(lane, &clip);
    let flag = buffer_with(&lane.device, &bytes(&[1])).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = W;
    pass.render_target_height = H;
    pass.default_raster_sample_count = 1;
    lane.run(&pass, |_, encoder, _| {
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&noise), 0).unwrap();
        encoder.set_fragment_texture(Some(&coverage), 1).unwrap();
        encoder.set_fragment_texture(Some(&written), 2).unwrap();
        encoder.set_fragment_buffer(Some(&flag), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap();
    unpack(&lane.read(&written, (0, 0), 16, BlitOption::None))
}

/// What every fragment computes before its pick.
fn base() -> [f32; 4] {
    let n = 0.0078125 * NOISE[0];
    [1.0 + n, 0.5 + n, 0.25 + n, 1.0]
}

/// The words a covered pixel of `fragment` writes.
fn expected(fragment: &str) -> Vec<u32> {
    let mut out = base();
    match fragment {
        "rch_select_sampled" => out = NOISE,
        "rch_select_uniform" => out[0] += out[1],
        _ => {}
    }
    words(&out)
}

/// The covered pixels whose write is not `want` (lost), and the others written (extra), with the
/// first difference.
fn compare(got: &[u32], want: &[u32]) -> (usize, usize, Option<String>) {
    let (mut lost, mut extra, mut first) = (0, 0, None);
    for y in 0..H {
        for x in 0..W {
            let at = ((y * W + x) * 4) as usize;
            let here = &got[at..at + 4];
            let kept = covered(x, y);
            let wrong = if kept {
                here != want
            } else {
                here != [SENTINEL; 4]
            };
            if wrong {
                if kept {
                    lost += 1;
                } else {
                    extra += 1;
                }
                first.get_or_insert_with(|| format!("({x}, {y}) covered {kept}: {here:x?}"));
            }
        }
    }
    (lost, extra, first)
}

/// A write after a pick only helpers make is lost only where the fragment records the note; on a
/// device with the defect, the pick's call is fenced where the fragment stage shuffles, and the
/// picks a sample reads after them or every invocation makes alike are left as they are.
#[test]
fn a_write_after_a_helper_pick_is_lost_only_where_noted() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let profile = lane.device.compile_device().profile();
    let defect = profile
        .workarounds()
        .contains(&DriverWorkaround::DiscardDropsSimdgroupStores);
    let fragment = &profile.subgroups().fragment;
    let shuffles = fragment.supported && fragment.ops.contains(OpBits::SHUFFLE);
    let library = library(&lane);
    let mut failures = Vec::new();
    for (entry, bare) in CASES {
        let (note, fences) = noted(&lane, &library, entry);
        let (lost, extra, first) = compare(&draw(&lane, &library, entry), &expected(entry));
        eprintln!(
            "helper select {entry} on {name} (defect {defect}, fragment shuffle {shuffles}): \
             note {note}, fences {fences}, {lost} lost, {extra} extra {first:?}"
        );
        let bare = defect && bare;
        if note != (bare && !shuffles) {
            failures.push(format!(
                "{entry}: note {note}, expected {}",
                bare && !shuffles
            ));
        }
        if (fences > 0) != (bare && shuffles) {
            failures.push(format!(
                "{entry}: {fences} fences, expected {}",
                if bare && shuffles { "some" } else { "none" }
            ));
        }
        if (lost > 0 && !note) || extra > 0 {
            failures.push(format!(
                "{entry}: note {note}, {lost} lost, {extra} extra, first {first:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "on {name}:\n  {}",
        failures.join("\n  ")
    );
    quiet(lane, &messages);
}

/// The GPU time of `rch_select_arm` where every fragment picks the sum: a 1024x1024 pass, 32
/// draws a command buffer, the least of five command buffers printed beside the fences the
/// fragment takes. Every pixel must hold the sum.
#[test]
fn a_fenced_pick_s_gpu_time() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let library = library(&lane);
    let entry = "rch_select_arm";
    let (note, fences) = noted(&lane, &library, entry);
    let pipeline = pipeline(&lane, &library, entry);
    let noise: Vec<u32> = (0..TILE * TILE).flat_map(|_| words(&NOISE)).collect();
    let noise = texture(&lane, (TILE, TILE), TextureUsage::SHADER_READ, &noise);
    let coverage = texture(
        &lane,
        (SIDE, SIDE),
        TextureUsage::SHADER_READ,
        &vec![0; (4 * SIDE * SIDE) as usize],
    );
    let written = texture(
        &lane,
        (SIDE, SIDE),
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        &vec![SENTINEL; (4 * SIDE * SIDE) as usize],
    );
    let corners = corners(&lane, &[[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]);
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = SIDE;
    pass.render_target_height = SIDE;
    pass.default_raster_sample_count = 1;
    let mut times = Vec::new();
    for _ in 0..RUNS {
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&noise), 0).unwrap();
        encoder.set_fragment_texture(Some(&coverage), 1).unwrap();
        encoder.set_fragment_texture(Some(&written), 2).unwrap();
        for _ in 0..DRAWS {
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let ms = (command_buffer.gpu_end_time().unwrap()
            - command_buffer.gpu_start_time().unwrap())
            * 1e3;
        times.push(ms);
    }
    let got = unpack(&lane.read(&written, (0, 0), 16, BlitOption::None));
    let mut want = base();
    want[0] += want[1];
    let want = words(&want);
    let wrong = got
        .chunks(4)
        .filter(|texel| *texel != want.as_slice())
        .count();
    let least = times.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!(
        "helper select GPU time on {name}: {entry}, fences {fences}, note {note}: \
         {DRAWS} draws of {SIDE}x{SIDE} in {least:.3} ms (least of {times:.3?}), {wrong} texels wrong"
    );
    assert_eq!(wrong, 0, "on {name}: texels that do not hold the sum");
    quiet(lane, &messages);
}
