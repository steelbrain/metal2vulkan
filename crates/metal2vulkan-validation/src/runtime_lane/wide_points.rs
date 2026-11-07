//! A point against its viewport on the device the cfg selects, under the validation layer.
//!
//! Metal probes (on an M3 Max and an M4; alike, each plain and under Metal's API validation, 2,810
//! draws each) drew one point a draw, of sizes 1 to 8, through viewports with fractional, whole,
//! mirrored and flipped edges, its center inside, at and past the viewport's edges, into an 8x8
//! target. Every stored pixel is one rule's ([`metal_pixels`]): Metal keeps a point while its
//! center lies within 1.5 `w` of the viewport's center on each axis (`|x|` and `|y|` at most
//! `1.5 w`; 1.5 drew, the next `float` above culled) and `0 <= z <= w`, and draws the pixels whose
//! centers its square covers, `[c - s/2, c + s/2)` on each axis, inside the pixels its viewport
//! overlaps (`floor` to `ceil` of its edges).
//!
//! Vulkan culls a point whose center leaves the clip volume (`|x| > w`), and clips a wide one at
//! fractional viewport edges by the device's own rule. The runtime draws a point through Vulkan
//! viewports widened by powers of two about their centers, which hold Metal's 1.5 and every
//! pixel the viewport overlaps with one more beside it, the vertex stage scaling its position
//! back and culling at Metal's 1.5 by four clip distances (`render_state::guard_points`). Where it
//! cannot, in a rate-mapped pass among others, the draw keeps Vulkan's clipping of points,
//! announced once (`point-guard-band`).
//!
//! Here `rt6_restart`'s vertex function draws a point of one pixel at each position below (its
//! `z` 0.25, `w` 1) through each of probe 9's viewports, with `rt6_render_state`'s white
//! fragment function, and the lane compares the pixel it covers with probe 9's ([`PROBE_9`]); at
//! 1.5 and the next `float` past it, with the rule's.

use super::buffer_with;
use super::degrade_notices::{
    assert_made_once, made_render_kinds, point_draw_kinds, shared_in_system_memory,
};
use super::rate_maps::descriptor;
use super::render_state::{quiet, texture, validated, Probe, PASS};
use super::WAIT;
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, PrimitiveTopologyClass, PrimitiveType};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, Degradation, LoadAction, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, StoreAction, Viewport,
};
use std::collections::BTreeSet;

/// The kinds `wide_points` takes: a draw of points that keeps Vulkan's clipping of points.
pub(super) const POINT_GUARD_BAND: [Degradation; 1] = [Degradation::PointGuardBand];

/// Probe 9's viewports: `(x, y, width, height)`, depths 0 to 1.
const VIEWPORTS: [[f64; 4]; 7] = [
    [4.5, 0.0, 1.0, 8.0],
    [4.25, 0.0, 1.5, 8.0],
    [4.7, 0.0, 1.0, 8.0],
    [4.2, 0.0, 1.0, 8.0],
    [0.0, 2.75, 8.0, 2.5],
    [3.25, 2.75, 1.5, 2.5],
    [4.0, 0.0, 1.0, 8.0],
];

/// Probe 9's positions along each axis, in the viewport's NDC: the center, then each offset.
const OFFSETS: [f32; 17] = [
    0.0, 0.5, -0.5, 0.9, -0.9, 0.99, -0.99, 1.01, -1.01, 1.1, -1.1, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0,
];

/// No pixel drawn.
const NONE: u8 = u8::MAX;

/// The pixel probe 9's point of one pixel stored, `8 x + y`, through each viewport ([`VIEWPORTS`])
/// at each offset ([`OFFSETS`]) along `x`, then along `y`; [`NONE`] where it drew none. The M3
/// Max and the M4 stored the same.
const PROBE_9: [[[u8; 17]; 2]; 7] = [
    [
        [
            35, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, NONE, NONE, NONE, NONE,
        ],
        [
            35, 33, 37, 32, 39, 32, 39, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
    ],
    [
        [
            35, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
        [
            35, 33, 37, 32, 39, 32, 39, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
    ],
    [
        [
            43, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, NONE, NONE, NONE, NONE,
        ],
        [
            43, 41, 45, 40, 47, 40, 47, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
    ],
    [
        [
            35, 35, 35, 43, 35, 43, 35, 43, 35, 43, 35, 43, NONE, NONE, NONE, NONE, NONE,
        ],
        [
            35, 33, 37, 32, 39, 32, 39, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
    ],
    [
        [
            27, 43, 11, 59, 3, 59, 3, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
        [
            27, 27, 28, 26, 29, 26, 29, 26, 29, 26, 29, 26, 29, NONE, NONE, NONE, NONE,
        ],
    ],
    [
        [
            27, 35, 27, 35, 27, 35, 27, 35, 27, 35, 27, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
        [
            27, 27, 28, 26, 29, 26, 29, 26, 29, 26, 29, 26, 29, NONE, NONE, NONE, NONE,
        ],
    ],
    [
        [
            35, 35, 35, 35, 35, 35, 35, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
        [
            35, 33, 37, 32, 39, 32, 39, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
        ],
    ],
];

/// The pixels `(x, y)` Metal's rule (the module docs) keeps of a point of `size` at the
/// clip-space `(x, y, z, w)` through `viewport` in an 8x8 target.
fn metal_pixels(viewport: [f64; 4], [x, y, z, w]: [f32; 4], size: f64) -> Vec<(u32, u32)> {
    let guard = 1.5 * f64::from(w);
    let kept = w > 0.0
        && (0.0..=w).contains(&z)
        && f64::from(x).abs() <= guard
        && f64::from(y).abs() <= guard;
    if !kept {
        return Vec::new();
    }
    let [ox, oy, width, height] = viewport;
    let center = [
        ox + (f64::from(x / w) + 1.0) / 2.0 * width,
        oy + (1.0 - f64::from(y / w)) / 2.0 * height,
    ];
    let pixels = |origin: f64, extent: f64| {
        let (lo, hi) = (origin.min(origin + extent), origin.max(origin + extent));
        (lo.floor().max(0.0) as u32)..(hi.ceil().min(8.0) as u32)
    };
    let covers = |pixel: u32, c: f64| {
        let p = f64::from(pixel) + 0.5;
        c - size / 2.0 <= p && p < c + size / 2.0
    };
    let mut kept = Vec::new();
    for py in pixels(oy, height) {
        for px in pixels(ox, width) {
            if covers(px, center[0]) && covers(py, center[1]) {
                kept.push((px, py));
            }
        }
    }
    kept
}

/// A pipeline drawing `rt6_restart`'s points of one pixel, white (`rt6_render_state`'s).
fn points(probe: &Probe) -> RenderPipelineState {
    probe
        .device
        .new_render_pipeline_state(&points_descriptor(probe), &PipelineOptions::default())
        .unwrap()
}

/// The kinds a test of [`points`] forces where the device has not announced them: its making's
/// (`made_render_kinds`), its draws of points' (`point_draw_kinds`) and the `Shared` buffers'
/// system memory.
fn forced_kinds(probe: &Probe) -> BTreeSet<Degradation> {
    let descriptor = points_descriptor(probe);
    let mut kinds = made_render_kinds(&probe.device, &descriptor);
    kinds.extend(point_draw_kinds(&probe.device, &descriptor));
    if shared_in_system_memory(&probe.device) {
        kinds.insert(Degradation::SharedInSystemMemory);
    }
    kinds
}

/// The descriptor of [`points`]'s state.
fn points_descriptor(probe: &Probe) -> RenderPipelineDescriptor {
    let function = |stem: &str, name: &str| {
        let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
        probe
            .device
            .new_library_with_air(&bitcode)
            .unwrap()
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function(
        "rt6_restart",
        "rt6_restart_vertex",
    )));
    descriptor.fragment = Some(StageSource::Function(function(
        "rt6_render_state",
        "rt6_white",
    )));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.input_primitive_topology = PrimitiveTopologyClass::Point;
    descriptor
}

/// The pixels `(x, y)` a point of one pixel at `(x, y)` stores through `viewport` on the device.
fn stored(
    probe: &Probe,
    pipeline: &RenderPipelineState,
    viewport: [f64; 4],
    [x, y]: [f32; 2],
) -> Vec<(u32, u32)> {
    let [ox, oy, width, height] = viewport;
    let position: Vec<u8> = [x, y].iter().flat_map(|v| v.to_le_bytes()).collect();
    let position = buffer_with(&probe.device, &position).unwrap();
    let stored = probe.run(PASS.size(8), |_, e, _| {
        e.set_viewport(Viewport::new(ox, oy, width, height, 0.0, 1.0))
            .unwrap();
        e.set_render_pipeline_state(pipeline).unwrap();
        e.set_vertex_buffer(Some(&position), 0, 0).unwrap();
        e.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0).unwrap();
    });
    stored
        .color
        .chunks(4)
        .enumerate()
        .filter(|(_, texel)| texel[0] != 0)
        .map(|(i, _)| ((i % 8) as u32, (i / 8) as u32))
        .collect()
}

/// Every point of probe 9 ([`PROBE_9`]) is Metal's rule's ([`metal_pixels`]); the device stores
/// probe 9's pixels, and the rule's at 1.5 and the next `float` past it on each axis, taking
/// Metal's guard band without a notice.
#[test]
fn points_past_their_viewport_store_metals_pixels() {
    // A code is `8 x + y`.
    let pixel = |code: u8| (code != NONE).then(|| (u32::from(code / 8), u32::from(code % 8)));
    let mut cases = Vec::new();
    for (v, viewport) in VIEWPORTS.iter().enumerate() {
        for (axis, codes) in PROBE_9[v].iter().enumerate() {
            for (&offset, &code) in OFFSETS.iter().zip(codes) {
                let mut at = [0.0f32; 2];
                at[axis] = offset;
                let metal: Vec<_> = pixel(code).into_iter().collect();
                assert_eq!(
                    metal_pixels(*viewport, [at[0], at[1], 0.25, 1.0], 1.0),
                    metal,
                    "probe 9 off the rule: {viewport:?} at {at:?}"
                );
                cases.push((*viewport, at, metal));
            }
            for offset in [1.5f32, -1.5, 1.5f32.next_up(), (-1.5f32).next_down()] {
                let mut at = [0.0f32; 2];
                at[axis] = offset;
                let metal = metal_pixels(*viewport, [at[0], at[1], 0.25, 1.0], 1.0);
                cases.push((*viewport, at, metal));
            }
        }
    }
    let (messages, _cache, probe) = validated();
    let pipeline = points(&probe);
    let differ: Vec<_> = cases
        .iter()
        .filter_map(|(viewport, at, metal)| {
            let stored = stored(&probe, &pipeline, *viewport, *at);
            (stored != *metal)
                .then(|| format!("{viewport:?} at {at:?}: {stored:?}, Metal {metal:?}"))
        })
        .collect();
    eprintln!(
        "WIDE-POINTS {}: {} points, {} not Metal's",
        probe.device.name(),
        cases.len(),
        differ.len()
    );
    assert_made_once(&messages, &[], &forced_kinds(&probe));
    quiet(probe, &messages);
    assert!(differ.is_empty(), "points not Metal's: {differ:#?}");
}

/// A draw of points in a rate-mapped pass keeps Vulkan's clipping of points, announced once
/// (`point-guard-band`): the runtime widens no region's viewport.
#[test]
fn points_in_a_rate_mapped_pass_announce_the_guard_band() {
    let (messages, _cache, probe) = validated();
    let pipeline = points(&probe);
    let map = probe
        .device
        .new_rasterization_rate_map(&descriptor(8, 8, &[(1.0, 1.0)]))
        .unwrap();
    let target = texture(&probe.device, PixelFormat::RGBA8Unorm, 8);
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    attachment.store_action = StoreAction::Store;
    pass.rasterization_rate_map = Some(map);
    let position = buffer_with(&probe.device, &[0u8; 8]).unwrap();
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut e = command_buffer.render_command_encoder(&pass).unwrap();
    e.set_render_pipeline_state(&pipeline).unwrap();
    e.set_vertex_buffer(Some(&position), 0, 0).unwrap();
    for _ in 0..2 {
        e.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0).unwrap();
    }
    e.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    assert_made_once(&messages, &POINT_GUARD_BAND, &forced_kinds(&probe));
    quiet(probe, &messages);
}
