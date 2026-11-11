//! A texture write after a branch only quad helpers take, whose arm takes the high words of
//! products beside a tangled instruction, on the device the cfg selects, under the validation
//! layer: a write is lost only where its fragment records
//! `PrecisionNote::HelperBranchDropsSimdgroupStores`.
//!
//! The owned fixture `rch_helper_tangle` draws the triangle of `runtime_lane::guarded_effects`
//! (its corners in pixels there; no pixel centre near an edge) with no attachment over a 32x16
//! pass: its edges cut quads, whose helper invocations run beside the covered pixels. Each
//! fragment reads a 32x16 coverage texture, 1 at a covered pixel and 0 elsewhere, samples an 8x8
//! texture of `(0.0625, 0.0625, 0.0625, 0)` at LOD 0, and takes a branch where its coverage is not
//! 1, which only the helpers take; the arm takes the high words of products (`mulhi`, an
//! `OpUMulExtended`, which SPIRV-Cross writes through its `optnone` `spvMulExtended`) beside a
//! tangled instruction. After the branch it writes a 32x16 texture of [`SENTINEL`] words at the
//! pixel. A covered pixel writes `(1, 0.5, 0.25, 1) + (2^-11, 2^-11, 2^-11, 0)`; a helper's write
//! has no effect. Metal's answer is that. Apple's compiler loses a simdgroup's texture writes after
//! an `optnone` call only its helpers make, but not where they make an implicit-LOD sample, a
//! derivative or a simdgroup ballot and shuffle after it that reads what the call computes: on
//! MoltenVK on an Apple GPU the `after` functions, whose tangled instruction reads the products,
//! are left as they are, and the `before` functions and `rch_tangle_unread`, whose tangled
//! instruction reads none of them, are fenced where the device's fragment stage shuffles (an
//! identity shuffle of the products follows the call, census `fragment.helper-fence`), else
//! record the note, their lost writes reported. No write may be lost where the note is not
//! recorded, and none written outside the triangle.

use super::buffer_with;
use super::guarded_effects::{
    bytes, covered, texture, unpack, words, CORNERS, H, ROW, SENTINEL, W,
};
use super::resolves::{lane, quiet, Lane};
use crate::air_library::AirLibrary;
use metal2vulkan::device::DriverWorkaround;
use metal2vulkan::device::OpBits;
use metal2vulkan::metal::{PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{PrecisionNote, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk;
use metal2vulkan::Budget;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{BlitOption, Library, PipelineOptions, RenderPassDescriptor};
use std::path::Path;

/// The sampled texture's side.
const TILE: u64 = 8;

/// Every texel of the sampled texture.
const NOISE: [f32; 4] = [0.0625, 0.0625, 0.0625, 0.0];

/// The fragment functions, each with whether its tangled instruction reads the products after
/// they are taken (in the arm, after the branch's merge, or after a loop only helpers repeat).
const CASES: [(&str, bool); 9] = [
    ("rch_tangle_sample_after", true),
    ("rch_tangle_dfdx_after", true),
    ("rch_tangle_quad_after", true),
    ("rch_tangle_lod_after", true),
    ("rch_tangle_merge_after", true),
    ("rch_tangle_loop_after", true),
    ("rch_tangle_quad_before", false),
    ("rch_tangle_sample_before", false),
    ("rch_tangle_unread", false),
];

/// The pipeline of `fragment` after `rch_tangle_vertex`.
fn descriptor(library: &Library, fragment: &str) -> RenderPipelineDescriptor {
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("rch_tangle_vertex"));
    descriptor.fragment = Some(function(fragment));
    descriptor
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

/// One draw of `fragment`; the written texture's words.
fn draw(lane: &Lane, library: &Library, fragment: &str) -> Vec<u32> {
    let pipeline = lane
        .device
        .new_render_pipeline_state(&descriptor(library, fragment), &PipelineOptions::default())
        .unwrap();
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
    let corners: Vec<f32> = CORNERS
        .iter()
        .flat_map(|[x, y]| {
            [
                (x / W as f64 * 2.0 - 1.0) as f32,
                (1.0 - y / H as f64 * 2.0) as f32,
            ]
        })
        .collect();
    let corners = buffer_with(&lane.device, &bytes(&words(&corners))).unwrap();
    let src = buffer_with(&lane.device, &vec![0; (16 * ROW * H) as usize]).unwrap();
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
        encoder.set_fragment_buffer(Some(&src), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap();
    unpack(&lane.read(&written, (0, 0), 16, BlitOption::None))
}

/// The words a covered pixel writes.
fn expected() -> Vec<u32> {
    let n = 0.0078125 * NOISE[0];
    words(&[1.0 + n, 0.5 + n, 0.25 + n, 1.0])
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

/// A write after a branch only helpers take is lost only where the fragment records the note;
/// on a device with the defect, a tangled instruction reading the arm's products after them keeps
/// the note away, and elsewhere the call is fenced where the fragment stage shuffles.
#[test]
fn a_write_after_a_helper_branch_is_lost_only_where_noted() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let profile = lane.device.compile_device().profile();
    let defect = profile
        .workarounds()
        .contains(&DriverWorkaround::DiscardDropsSimdgroupStores);
    let fragment = &profile.subgroups().fragment;
    let shuffles = fragment.supported && fragment.ops.contains(OpBits::SHUFFLE);
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/rch_helper_tangle.ll");
    let library = lane
        .device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let mut failures = Vec::new();
    for (entry, reads) in CASES {
        let (note, fences) = noted(&lane, &library, entry);
        let (lost, extra, first) = compare(&draw(&lane, &library, entry), &expected());
        eprintln!(
            "helper tangle {entry} on {name} (defect {defect}, fragment shuffle {shuffles}): \
             note {note}, fences {fences}, {lost} lost, {extra} extra {first:?}"
        );
        let bare = defect && !reads;
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
