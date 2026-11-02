//! Device tests of Apple's `setLineWidth:` (the runtime's `render/line_width.rs`) on the device the
//! cfg selects, under the validation layer, against what plain Metal drew for
//! `crates/metal2vulkan-conformance/probes/line_widths.m` on the Apple M3 Max, macOS 27.0.1: the
//! probe's output, [`MEASURED`], pixel for pixel.
//!
//! Each case is one of the probe's, drawn as the probe draws it: the owned fixture
//! `rt6_provoking`'s `pv_vertex_smooth` places each point at the clip-space position the probe's
//! vertex function computes from its pixel coordinates (`(x / (S / 2) - 1, 1 - y / (S / 2))`, the
//! same two `f32` operations), and `a2_front_facing`'s fragment function writes one color for a
//! front-facing fragment and another for a back-facing one, into an `RGBA8Unorm` target: white,
//! or `1/255` added for each fragment where the probe counts them. The map of what drew is printed
//! as the probe prints it and compared with its map, and every fragment must be front-facing, as
//! a line's is. The cases: the lane's 16x16 list of two lines and three-vertex strip at widths 0,
//! 0.5, 1, 2, 3.5 and 8; the widths Metal quantizes; the width's encoder state; the band's edges
//! and ends; width 0, on the centres and off them (lines offset by steps of 1/512 pixel, scattered
//! endpoints as lists, reversed, instanced and a strip, clip-space positions), which Metal draws
//! as 1/16; strips' overlaps counted; filled triangles; and 4 samples at widths 0.5 and 2.
//!
//! The work counts show which route each draw took (`WorkCounts::line_band_draws`,
//! `device_wide_line_draws`, `nearest_width_line_draws`): every draw at a width other than 1 takes
//! one of the three, and on a device without `wideLines` a line takes the band. A draw the
//! device's own lines draw otherwise than Metal is announced (`device-wide-lines`, at every
//! occurrence here) and left unchecked; under `DevicePolicy::exact_answers()` no line draw is,
//! where the device has `tessellationShader`. What neither the device's lines nor the band draw (a
//! triangle's edges in fill mode `Lines` the device's lines do not draw; a line without
//! `tessellationShader` the device's lines do not draw) runs the device's lines at the nearest
//! width they draw, announced (`lines-at-nearest-width`) and counted at each draw; where that
//! width is 1 it is compared with Metal's map of the same geometry at width 1. Nothing is refused.
//! A draw of no complete primitive (the consumer's triangles of 1 and 2 vertices, by vertices and
//! by indices, under the last provoking vertex) draws nothing and takes no route. Not covered:
//! Metal's width-1 lines on 4 samples (the probe's `msaa` width 1 and diagonal cases) where the
//! device's lines are not Metal's own.

use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DevicePolicy, VkFeature};
use metal2vulkan::metal::{BlendFactor, IndexType, TriangleFillMode};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, Error, Library, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    StoreAction, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeMap;

/// What plain Metal drew for the probe, case by case (its output, as the probe prints it).
const MEASURED: &str = include_str!("line_widths_m3_max.txt");

/// The kinds the lane's runs take: a draw of lines at a width other than 1 that the device's own
/// lines draw otherwise than Metal's parallelogram, and one they draw at the nearest width they
/// draw where neither they nor the band draw Metal's.
pub(super) const WIDE_LINES: [Degradation; 2] = [
    Degradation::DeviceWideLines,
    Degradation::LinesAtNearestWidth,
];

/// The probe's widths of the lane's draws.
const LANE_WIDTHS: [f32; 6] = [0.0, 0.5, 1.0, 2.0, 3.5, 8.0];

/// How a case's fragments shade.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Shade {
    /// Written whole: a pixel any fragment covered.
    White,
    /// `1/255` added for each fragment.
    Count,
}

/// One draw of an encoder: its primitive type, its points in pixel coordinates (y down), or
/// clip-space `x` and `y` (`z` 0, `w` 1) where `clip`, the width set before it (`None`: none), its
/// triangle fill mode, whether it sets the other of the probe's two pipeline states first, its
/// instances, whether it draws through 16-bit indices `0..n` (`indexed`), and whether it sets
/// Apple's last provoking vertex first (`provoking_last`).
#[derive(Clone, Debug)]
struct Draw {
    primitive: PrimitiveType,
    points: Vec<[f32; 2]>,
    clip: bool,
    width: Option<f32>,
    fill: TriangleFillMode,
    other_pipeline: bool,
    instances: u32,
    indexed: bool,
    provoking_last: bool,
}

impl Draw {
    fn line(primitive: PrimitiveType, points: &[f32], width: f32) -> Draw {
        Draw {
            primitive,
            points: points.chunks(2).map(|p| [p[0], p[1]]).collect(),
            clip: false,
            width: Some(width),
            fill: TriangleFillMode::Fill,
            other_pipeline: false,
            instances: 1,
            indexed: false,
            provoking_last: false,
        }
    }

    /// Whether the draw draws lines at a width other than Metal's 1.
    fn wide(&self, width: f32) -> bool {
        let lines = matches!(
            self.primitive,
            PrimitiveType::Line | PrimitiveType::LineStrip
        ) || self.fill == TriangleFillMode::Lines;
        metal_width(width) != 1.0 && lines
    }
}

/// The width Metal draws for `setLineWidth:`'s `width`: `floor(16 w) / 16` in `[0, 16]`, the
/// probe's measured rule, written apart from the runtime's.
fn metal_width(width: f32) -> f32 {
    if width.is_nan() || width <= 0.0 {
        0.0
    } else {
        ((width * 16.0).floor() / 16.0).min(16.0)
    }
}

/// What the device's lines draw: whether it enables `wideLines` and `tessellationShader`, and its
/// `lineWidthRange`.
#[derive(Clone, Copy, Debug)]
struct LineFacts {
    wide_lines: bool,
    tessellates: bool,
    range: [f32; 2],
}

impl LineFacts {
    fn of(device: &Device) -> LineFacts {
        let compile = device.compile_device();
        let profile = compile.profile();
        let has = |feature| profile.enabled().features.contains(&feature);
        LineFacts {
            wide_lines: has(VkFeature::WideLines),
            tessellates: has(VkFeature::TessellationShader),
            range: profile.limits().line_width_range_bits.map(f32::from_bits),
        }
    }

    /// The width nearest the one Metal draws for `width` (1/16 at 0) that the device's lines
    /// draw: in their range, 1 without `wideLines` (the runtime's docs, written apart from it).
    fn nearest(self, width: f32) -> f32 {
        let w = metal_width(width).max(1.0 / 16.0);
        let [low, high] = self.range;
        match self.wide_lines {
            true => w.clamp(low, high),
            false => 1.0,
        }
    }

    /// Whether the device's own lines draw Metal's `width` (`wideLines`, in their range, above
    /// 0).
    fn draws(self, width: f32) -> bool {
        let w = metal_width(width);
        let [low, high] = self.range;
        self.wide_lines && w > 0.0 && (low..=high).contains(&w)
    }
}

/// One of the probe's renders: an `S x S` target of `samples` samples, cleared to zero, and the
/// draws of each of its encoders, in one command buffer.
#[derive(Clone, Debug)]
struct Shot {
    side: u64,
    samples: u32,
    shade: Shade,
    encoders: Vec<Vec<Draw>>,
}

impl Shot {
    fn one(side: u64, samples: u32, shade: Shade, draw: Draw) -> Shot {
        Shot {
            side,
            samples,
            shade,
            encoders: vec![vec![draw]],
        }
    }

    /// The draws at a width other than 1, the encoder's width carried from draw to draw and
    /// reset to 1 at each encoder (the probe's `state` case).
    fn wide_draws(&self) -> u64 {
        let mut wide = 0;
        for draws in &self.encoders {
            let mut width = 1.0;
            for draw in draws {
                width = draw.width.unwrap_or(width);
                wide += u64::from(draw.wide(width));
            }
        }
        wide
    }
}

/// The probe's cell for a pixel of red `v`: `.` none, `#` white, a digit the fragments counted or
/// the samples covered.
fn cell(v: u8, shade: Shade, samples: u32) -> char {
    if v == 0 {
        return '.';
    }
    if shade == Shade::Count {
        return match v {
            1..=9 => char::from(b'0' + v),
            _ => '*',
        };
    }
    if samples > 1 {
        let k = (f64::from(v) * f64::from(samples) / 255.0).round() as u8;
        return char::from(b'0' + k);
    }
    match v {
        255 => '#',
        _ => '?',
    }
}

/// The probe's `MAP` block of `red`: its title and covered pixels, then each row any pixel of
/// which is covered.
fn map(title: &str, side: usize, red: &[u8], shade: Shade, samples: u32) -> String {
    let covered = red.iter().filter(|&&v| v != 0).count();
    let mut out = format!("MAP {title} ({covered} pixels)\n");
    for (y, row) in red.chunks(side).enumerate() {
        if row.iter().all(|&v| v == 0) {
            continue;
        }
        out.push_str(&format!("{y:3} "));
        out.extend(row.iter().map(|&v| cell(v, shade, samples)));
        out.push('\n');
    }
    out
}

/// The probe's `LANE` block of `red`: every row.
fn lane_block(name: &str, width: f32, red: &[u8]) -> String {
    let mut out = format!("LANE {name} {width}\n");
    for row in red.chunks(16) {
        out.push_str("  \"");
        out.extend(row.iter().map(|&v| cell(v, Shade::White, 1)));
        out.push_str("\",\n");
    }
    out
}

/// [`MEASURED`]'s blocks: each `MAP` by its title (without its count), each `LANE` by its name and
/// width, each with its lines; and each `QUANTIZE` line's rows by the width's bits.
fn measured() -> (BTreeMap<String, String>, BTreeMap<u32, String>) {
    let mut blocks = BTreeMap::new();
    let mut quantized = BTreeMap::new();
    let mut open: Option<(String, String)> = None;
    for line in MEASURED.lines().filter(|l| !l.starts_with('#')) {
        if let Some(rest) = line.strip_prefix("QUANTIZE ") {
            let bits = rest.split("(bits ").nth(1).unwrap().get(..8).unwrap();
            let rows = rest.split("): ").nth(1).unwrap();
            quantized.insert(u32::from_str_radix(bits, 16).unwrap(), rows.to_owned());
            continue;
        }
        let key = match (line.strip_prefix("MAP "), line.strip_prefix("LANE ")) {
            (Some(title), _) => Some(title.rsplit_once(" (").unwrap().0.to_owned()),
            (_, Some(lane)) => Some(format!("LANE {lane}")),
            _ => None,
        };
        match key {
            Some(key) => {
                if let Some((k, text)) = open.take() {
                    blocks.insert(k, text);
                }
                open = Some((key, format!("{line}\n")));
            }
            None => {
                let (_, text) = open.as_mut().expect("a row inside a block");
                text.push_str(line);
                text.push('\n');
            }
        }
    }
    if let Some((k, text)) = open {
        blocks.insert(k, text);
    }
    (blocks, quantized)
}

/// The probe's `scatter`: `count` points of a `side x side` target, each coordinate in
/// `[1, side - 1)` on a grid of 1/4096 pixel, `x` then `y`, from its 64-bit linear congruential
/// generator seeded with `seed`.
fn scatter(count: usize, side: u32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..2 * count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let r = (state >> 33) as u32;
            1.0 + (r % ((side - 2) * 4096)) as f32 / 4096.0
        })
        .collect()
}

/// The probe's eight clip-space positions (`x`, `y`) of its `zero` case.
const CLIP_POSITIONS: [f32; 16] = [
    -0.32525787,
    0.4939026,
    0.38724214,
    -0.6796814,
    0.13544922,
    -0.1575119,
    0.21390076,
    0.4868866,
    0.9281403,
    -0.76723635,
    -0.45363313,
    0.2326004,
    -0.824118,
    -0.9427811,
    0.06479645,
    -0.2337021,
];

/// The probe's `MAP` cases, each by its title: `ties`, `zero`, `strips`, `fill`, `msaa` and
/// `state`, geometry for geometry.
fn map_cases() -> Vec<(String, Shot)> {
    use PrimitiveType::{Line, LineStrip, Triangle};
    let mut out = Vec::new();
    let mut one = |title: &str, side, samples, shade, draw| {
        out.push((title.to_owned(), Shot::one(side, samples, shade, draw)));
    };
    let ties: [(&str, u64, [f32; 4], f32); 14] = [
        ("horizontal y=16 width 3", 32, [4.0, 16.0, 28.0, 16.0], 3.0),
        (
            "horizontal y=16 width 0.5",
            32,
            [4.0, 16.0, 28.0, 16.0],
            0.5,
        ),
        (
            "horizontal y=16.5 width 0.5",
            32,
            [4.0, 16.5, 28.0, 16.5],
            0.5,
        ),
        (
            "horizontal y=16 width 3.5",
            32,
            [4.0, 16.0, 28.0, 16.0],
            3.5,
        ),
        (
            "horizontal y=16.5 width 3.5",
            32,
            [4.0, 16.5, 28.0, 16.5],
            3.5,
        ),
        (
            "ends on centres x 4.5 to 27.5 width 2",
            32,
            [4.5, 10.0, 27.5, 10.0],
            2.0,
        ),
        (
            "ends on centres x 27.5 to 4.5 width 2",
            32,
            [27.5, 10.0, 4.5, 10.0],
            2.0,
        ),
        (
            "ends on centres y 4.5 to 27.5 width 2",
            32,
            [10.0, 4.5, 10.0, 27.5],
            2.0,
        ),
        (
            "ends on centres y 27.5 to 4.5 width 2",
            32,
            [10.0, 27.5, 10.0, 4.5],
            2.0,
        ),
        (
            "45 degrees down-right width 4",
            24,
            [4.0, 4.0, 20.0, 20.0],
            4.0,
        ),
        (
            "45 degrees up-left width 4",
            24,
            [20.0, 20.0, 4.0, 4.0],
            4.0,
        ),
        (
            "45 degrees up-right width 4",
            24,
            [4.0, 20.0, 20.0, 4.0],
            4.0,
        ),
        ("x-major diagonal width 8", 32, [2.0, 6.0, 30.0, 20.0], 8.0),
        ("y-major width 3.5", 32, [8.0, 4.0, 20.0, 28.0], 3.5),
    ];
    for (title, side, points, width) in ties {
        one(
            title,
            side,
            1,
            Shade::White,
            Draw::line(Line, &points, width),
        );
    }
    let zero: [(&str, [f32; 4], f32); 11] = [
        (
            "horizontal y=16.5 x 4.5 to 27.5 width 0",
            [4.5, 16.5, 27.5, 16.5],
            0.0,
        ),
        (
            "horizontal y=16.5 x 27.5 to 4.5 width 0",
            [27.5, 16.5, 4.5, 16.5],
            0.0,
        ),
        (
            "vertical x=16.5 y 4.5 to 27.5 width 0",
            [16.5, 4.5, 16.5, 27.5],
            0.0,
        ),
        (
            "diagonal through centres width 0",
            [4.5, 4.5, 28.5, 16.5],
            0.0,
        ),
        (
            "diagonal through centres reversed width 0",
            [28.5, 16.5, 4.5, 4.5],
            0.0,
        ),
        (
            "45 degrees through centres width 0",
            [4.5, 4.5, 20.5, 20.5],
            0.0,
        ),
        (
            "diagonal through centres width -3",
            [4.5, 4.5, 28.5, 16.5],
            -3.0,
        ),
        (
            "diagonal through centres width NaN",
            [4.5, 4.5, 28.5, 16.5],
            f32::NAN,
        ),
        (
            "diagonal through centres width 0.0625",
            [4.5, 4.5, 28.5, 16.5],
            0.0625,
        ),
        (
            "horizontal y=16 width 0 (through no centre)",
            [4.0, 16.0, 28.0, 16.0],
            0.0,
        ),
        (
            "horizontal y=16.5 x 4 to 28 width 0",
            [4.0, 16.5, 28.0, 16.5],
            0.0,
        ),
    ];
    for (title, points, width) in zero {
        one(title, 32, 1, Shade::White, Draw::line(Line, &points, width));
    }
    // Lines offset from rows and columns of centres by `(r - 32) / 512` pixel.
    let (mut rows, mut columns) = (Vec::new(), Vec::new());
    for r in 4..60 {
        let at = r as f32 + 0.5 + (r - 32) as f32 / 512.0;
        rows.extend([2.25, at, 9.75, at]);
        columns.extend([at, 2.25, at, 9.75]);
    }
    for width in [0.0f32, 0.0625] {
        for (title, points) in [
            ("rows offset by (r - 32) / 512", &rows),
            ("columns offset by (c - 32) / 512", &columns),
        ] {
            let title = format!("{title} width {width}");
            one(&title, 64, 1, Shade::White, Draw::line(Line, points, width));
        }
    }
    // Scattered endpoints: 48 lines, the same reversed and as two instances, and a strip.
    let list = scatter(96, 32, 1);
    let reversed: Vec<f32> = list
        .chunks(4)
        .flat_map(|l| [l[2], l[3], l[0], l[1]])
        .collect();
    let strip = scatter(49, 32, 2);
    let scattered: [(&str, PrimitiveType, &[f32], f32, u32); 9] = [
        ("scattered lines width 0", Line, &list, 0.0, 1),
        ("scattered lines width 0.03125", Line, &list, 0.03125, 1),
        ("scattered lines width 0.0625", Line, &list, 0.0625, 1),
        ("scattered lines width -1", Line, &list, -1.0, 1),
        ("scattered lines width NaN", Line, &list, f32::NAN, 1),
        ("scattered lines width 2.5", Line, &list, 2.5, 1),
        ("scattered lines reversed width 0", Line, &reversed, 0.0, 1),
        ("scattered lines 2 instances width 0", Line, &list, 0.0, 2),
        ("scattered strip width 0", LineStrip, &strip, 0.0, 1),
    ];
    for (title, primitive, points, width, instances) in scattered {
        let draw = Draw {
            instances,
            ..Draw::line(primitive, points, width)
        };
        one(&format!("{title}, fragments"), 32, 1, Shade::Count, draw);
    }
    for (title, primitive) in [
        ("clip-space lines width 0, fragments", Line),
        ("clip-space strip width 0, fragments", LineStrip),
    ] {
        let draw = Draw {
            clip: true,
            ..Draw::line(primitive, &CLIP_POSITIONS, 0.0)
        };
        one(title, 16, 1, Shade::Count, draw);
    }
    let corner = [4.0, 8.5, 24.5, 8.5, 24.5, 28.0];
    let peak = [4.0, 28.0, 16.0, 4.0, 28.0, 28.0];
    for width in [1.0f32, 3.5, 8.0, 0.0] {
        for (name, points) in [("L", &corner), ("peak", &peak)] {
            let title = format!("strip {name} width {width}, fragments");
            one(
                &title,
                32,
                1,
                Shade::Count,
                Draw::line(LineStrip, points, width),
            );
        }
    }
    let triangle = [4.0, 4.0, 28.0, 4.0, 16.0, 28.0];
    for (fill, width, title) in [
        (
            TriangleFillMode::Lines,
            1.0,
            "triangle fill Lines width 1, fragments",
        ),
        (
            TriangleFillMode::Lines,
            4.0,
            "triangle fill Lines width 4, fragments",
        ),
        (
            TriangleFillMode::Lines,
            0.0,
            "triangle fill Lines width 0, fragments",
        ),
        (
            TriangleFillMode::Fill,
            4.0,
            "triangle fill Fill width 4, fragments",
        ),
    ] {
        let mut draw = Draw::line(Triangle, &triangle, width);
        draw.fill = fill;
        one(title, 32, 1, Shade::Count, draw);
    }
    let horizontal = [4.0, 16.0, 28.0, 16.0];
    for width in [0.5f32, 2.0] {
        let title = format!("4 samples horizontal y=16 width {width}, samples");
        one(
            &title,
            32,
            4,
            Shade::White,
            Draw::line(Line, &horizontal, width),
        );
    }
    // `state`: width 4 before the first draw, none before the second, of the other pipeline
    // state; then width 4 in a first encoder, none in a second.
    let upper = Draw::line(Line, &[4.0, 8.0, 28.0, 8.0], 4.0);
    let unset = |points: &[f32]| Draw {
        width: None,
        ..Draw::line(Line, points, 1.0)
    };
    let lower = Draw {
        other_pipeline: true,
        ..unset(&[4.0, 20.0, 28.0, 20.0])
    };
    out.push((
        "width 4, then a draw of another pipeline state with no width set".to_owned(),
        Shot {
            side: 32,
            samples: 1,
            shade: Shade::White,
            encoders: vec![vec![upper.clone(), lower]],
        },
    ));
    out.push((
        "width 4 in encoder 0; encoder 1 sets none".to_owned(),
        Shot {
            side: 32,
            samples: 1,
            shade: Shade::White,
            encoders: vec![vec![upper], vec![unset(&[4.0, 14.0, 28.0, 14.0])]],
        },
    ));
    out
}

/// The lane's device, queue and the probe's two pipeline states of each shade and sample count.
struct Lane {
    device: Device,
    queue: CommandQueue,
    vertex: Library,
    fragment: Library,
    pipelines: BTreeMap<(Shade, u32), [RenderPipelineState; 2]>,
}

/// What a shot drew: each pixel's red (front-facing fragments) and green (back-facing ones).
struct Drawn {
    red: Vec<u8>,
    green: Vec<u8>,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let library = |name: &str| {
            let bitcode = llvm_assemble(&engine_dir().join(name)).unwrap();
            device.new_library_with_air(&bitcode).unwrap()
        };
        Lane {
            queue: device.new_command_queue().unwrap(),
            vertex: library("rt6_provoking.ll"),
            fragment: library("a2_front_facing.ll"),
            pipelines: BTreeMap::new(),
            device,
        }
    }

    /// The probe's two pipeline states of `shade` and `samples` (alike, as the probe's `a` and
    /// `b` are).
    fn pipelines(&mut self, shade: Shade, samples: u32) -> [RenderPipelineState; 2] {
        let function = |library: &Library, name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(function(&self.vertex, "pv_vertex_smooth"));
        descriptor.fragment = Some(function(&self.fragment, "a2_front_facing"));
        descriptor.raster_sample_count = samples;
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        if shade == Shade::Count {
            color.blending_enabled = true;
            color.destination_rgb_blend_factor = BlendFactor::One;
            color.destination_alpha_blend_factor = BlendFactor::One;
        }
        descriptor.color_attachments[0] = Some(color);
        let device = &self.device;
        self.pipelines
            .entry((shade, samples))
            .or_insert_with(|| {
                [0, 1].map(|_| {
                    device
                        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                        .unwrap()
                })
            })
            .clone()
    }

    /// `shot` drawn, or the first refusal of its calls.
    fn draw(&mut self, shot: &Shot) -> Result<Drawn, Error> {
        let [a, b] = self.pipelines(shot.shade, shot.samples);
        let side = shot.side;
        let mut t = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
        t.usage = TextureUsage::RENDER_TARGET;
        t.resource_options = ResourceOptions::shared();
        let resolved = self.device.new_texture(&t).unwrap();
        let target = match shot.samples {
            1 => resolved.clone(),
            n => {
                t.texture_type = TextureType::Type2DMultisample;
                t.sample_count = u64::from(n);
                t.resource_options = ResourceOptions::private();
                self.device.new_texture(&t).unwrap()
            }
        };
        // A front-facing fragment writes red, a back-facing one green: white or one count each.
        let unit = match shot.shade {
            Shade::White => 1.0,
            Shade::Count => 1.0 / 255.0,
        };
        let colors: Vec<u8> = [unit, 0.0, 0.0, unit, 0.0, unit, 0.0, unit]
            .iter()
            .flat_map(|v: &f32| v.to_le_bytes())
            .collect();
        let half_side = side as f32 / 2.0;
        let command_buffer = self.queue.command_buffer()?;
        let mut index = 0;
        let mut index_buffers = Vec::new();
        for (e, draws) in shot.encoders.iter().enumerate() {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = match e {
                0 => LoadAction::Clear,
                _ => LoadAction::Load,
            };
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            match shot.samples {
                1 => color.store_action = StoreAction::Store,
                _ => {
                    color.resolve_texture = Some(resolved.clone());
                    color.store_action = StoreAction::MultisampleResolve;
                }
            }
            let mut encoder = command_buffer.render_command_encoder(&pass)?;
            let drawn = (|| {
                encoder.set_render_pipeline_state(&a)?;
                encoder.set_fragment_bytes(&colors, 0)?;
                for draw in draws {
                    if let Some(width) = draw.width {
                        encoder.set_line_width(width)?;
                    }
                    if draw.other_pipeline {
                        encoder.set_render_pipeline_state(if index & 1 == 1 { &b } else { &a })?;
                    }
                    encoder.set_triangle_fill_mode(draw.fill)?;
                    if draw.provoking_last {
                        encoder.set_provoking_vertex_mode(1)?;
                    }
                    // The probe's vertex function's arithmetic, in the same `f32` operations;
                    // clip-space positions as they are.
                    let positions: Vec<u8> = draw
                        .points
                        .iter()
                        .flat_map(|&[x, y]| match draw.clip {
                            true => [x, y, 0.0, 1.0],
                            false => [x / half_side - 1.0, 1.0 - y / half_side, 0.0, 1.0],
                        })
                        .flat_map(f32::to_le_bytes)
                        .collect();
                    encoder.set_vertex_bytes(&positions, 0)?;
                    let count = draw.points.len() as u32;
                    match draw.indexed {
                        false => {
                            encoder.draw_primitives(draw.primitive, 0, count, draw.instances, 0)?
                        }
                        true => {
                            let indices: Vec<u8> =
                                (0..count as u16).flat_map(u16::to_le_bytes).collect();
                            let buffer = buffer_with(&self.device, &indices).unwrap();
                            encoder.draw_indexed_primitives(
                                draw.primitive,
                                count,
                                IndexType::UInt16,
                                &buffer,
                                0,
                                draw.instances,
                                0,
                                0,
                            )?;
                            index_buffers.push(buffer);
                        }
                    }
                    index += 1;
                }
                Ok(())
            })();
            let ended = encoder.end_encoding();
            drawn.and(ended)?;
        }
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{shot:?}: {:?}",
            command_buffer.error()
        );
        let mut rgba = vec![0u8; (side * side * 4) as usize];
        resolved
            .get_bytes(
                &mut rgba,
                side * 4,
                0,
                Region::new_2d(0, 0, side, side),
                0,
                0,
            )
            .unwrap();
        Ok(Drawn {
            red: rgba.chunks(4).map(|p| p[0]).collect(),
            green: rgba.chunks(4).map(|p| p[1]).collect(),
        })
    }
}

/// What a run found.
#[derive(Debug, Default)]
struct Found {
    /// Cases run: the lane's, the maps' and the quantized widths'.
    cases: usize,
    /// Of them, the cases that draw a triangle's edges at a width other than 1.
    edges: usize,
    /// Cases drawn as Metal drew them, pixel for pixel.
    matched: usize,
    /// Cases a device line drew otherwise than Metal, announced: left unchecked.
    announced: usize,
    /// Of them, the cases of a triangle's edges, which the band does not draw.
    announced_edges: usize,
    /// Cases drawn at the nearest width the device's lines draw, announced.
    nearest: usize,
    /// Of them, the cases drawn at width 1 as Metal drew the same geometry at width 1, pixel for
    /// pixel.
    nearest_matched: usize,
    /// The device's lines.
    facts: Option<LineFacts>,
}

/// The `DeviceWideLines` and `LinesAtNearestWidth` notices among `messages`' since the last take.
fn announced_lines(messages: &Messages) -> (usize, usize) {
    let notices = messages.take_notices();
    let kinds: Vec<Degradation> = notices
        .iter()
        .filter_map(|m| m.degradation.as_ref())
        .map(|notice| notice.degradation)
        .collect();
    let count = |kind| kinds.iter().filter(|&&k| k == kind).count();
    (
        count(Degradation::DeviceWideLines),
        count(Degradation::LinesAtNearestWidth),
    )
}

/// The draws of each route between two work counts: the band's, the device's lines' at Metal's
/// width, and theirs at the nearest width.
fn routed(before: &WorkCounts, after: &WorkCounts) -> [u64; 3] {
    [
        after.line_band_draws - before.line_band_draws,
        after.device_wide_line_draws - before.device_wide_line_draws,
        after.nearest_width_line_draws - before.nearest_width_line_draws,
    ]
}

/// The key of Metal's map of `key`'s geometry at width 1, drawn at `width`, where the probe has
/// one: the lane's `LANE <name> 1`; a map's title with its one `width <w>` as `width 1`.
fn width_one_twin(key: &str, width: f32, blocks: &BTreeMap<String, String>) -> Option<String> {
    let twin = match key.strip_prefix("LANE ") {
        Some(lane) => format!("LANE {} 1", lane.split(' ').next()?),
        None => {
            let at = format!("width {width}");
            (key.matches(&at).count() == 1).then(|| key.replace(&at, "width 1"))?
        }
    };
    blocks.contains_key(&twin).then_some(twin)
}

/// Every case of [`MEASURED`] on the cfg's device under `policy`: drawn as Metal drew it, or
/// announced (where the device's lines draw it at another width than Metal's, compared with
/// Metal's map at width 1 where that width is 1); and each draw at a width other than 1 counted by
/// a route. Nothing is refused.
fn run(policy: DevicePolicy) -> Found {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut found = Found::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy = policy;
            options.announce_each = WIDE_LINES.into_iter().collect();
        })
        .unwrap();
        let facts = LineFacts::of(&device);
        let LineFacts {
            wide_lines,
            tessellates,
            ..
        } = facts;
        found.facts = Some(facts);
        let name = device.name().to_owned();
        let mut lane = Lane::new(device);
        let (blocks, quantized) = measured();
        let mut cases: Vec<(String, Shot, Option<f32>)> = Vec::new();
        let lines = [1.0, 3.0, 15.0, 3.0, 1.0, 8.0, 15.0, 15.0];
        let strip = [3.5, 1.0, 3.5, 9.5, 13.5, 14.5];
        for (lane_name, primitive, points) in [
            ("lines", PrimitiveType::Line, &lines[..]),
            ("strip", PrimitiveType::LineStrip, &strip[..]),
        ] {
            for width in LANE_WIDTHS {
                let draw = Draw::line(primitive, points, width);
                let key = format!("LANE {lane_name} {width}");
                cases.push((key, Shot::one(16, 1, Shade::White, draw), Some(width)));
            }
        }
        cases.extend(
            map_cases()
                .into_iter()
                .map(|(title, shot)| (title, shot, None)),
        );
        let mut bands = 0;
        let mut device_lines = 0;
        let mut nearest_lines = 0;
        let mut wide = 0;
        let mut checked = std::collections::BTreeSet::new();
        for (key, shot, lane_width) in &cases {
            let edges = shot.encoders[0][0].fill == TriangleFillMode::Lines;
            found.edges += usize::from(edges && shot.wide_draws() > 0);
            let before = lane.device.work_counts();
            let drawn = lane
                .draw(shot)
                .unwrap_or_else(|error| panic!("{name}: {key}: {error}"));
            let after = lane.device.work_counts();
            let [band, own, nearest] = routed(&before, &after);
            bands += band;
            device_lines += own;
            nearest_lines += nearest;
            let (announced, announced_nearest) = announced_lines(&messages);
            assert_eq!(
                band + own + nearest,
                shot.wide_draws(),
                "{name}: {key}: a wide draw took no route"
            );
            wide += shot.wide_draws();
            if !wide_lines {
                assert_eq!(own, 0, "{name}: {key}: device lines without wideLines");
            }
            // Each draw at the nearest width announces it here, and only those.
            assert_eq!(
                announced_nearest as u64, nearest,
                "{name}: {key}: nearest-width draws and their notices"
            );
            assert!(
                drawn.green.iter().all(|&g| g == 0),
                "{name}: {key}: back-facing fragments"
            );
            if announced > 0 {
                found.announced += 1;
                found.announced_edges += usize::from(edges);
                continue;
            }
            let side = shot.side as usize;
            if nearest > 0 {
                found.nearest += 1;
                // At width 1, Metal's own map of the geometry at width 1, where the probe has it.
                let draw = &shot.encoders[0][0];
                let width = draw.width.unwrap_or(1.0);
                let twin = (shot.encoders.len() == 1 && shot.encoders[0].len() == 1)
                    .then(|| width_one_twin(key, width, &blocks))
                    .flatten()
                    .filter(|_| facts.nearest(width) == 1.0);
                if let Some(twin) = twin {
                    let ours = match lane_width {
                        Some(_) => lane_block(key.split(' ').nth(1).unwrap(), 1.0, &drawn.red),
                        None => map(&twin, side, &drawn.red, shot.shade, shot.samples),
                    };
                    assert_eq!(
                        &ours, &blocks[&twin],
                        "{name}: {key}: drawn at width 1 otherwise than Metal's {twin}"
                    );
                    found.nearest_matched += 1;
                }
                continue;
            }
            let ours = match lane_width {
                Some(width) => lane_block(key.split(' ').nth(1).unwrap(), *width, &drawn.red),
                None => map(key, side, &drawn.red, shot.shade, shot.samples),
            };
            let metal = blocks
                .get(key)
                .unwrap_or_else(|| panic!("{key}: not among Metal's answers"));
            assert_eq!(&ours, metal, "{name}: {key}: drawn otherwise than Metal");
            checked.insert(key.clone());
            found.matched += 1;
        }
        // The quantized widths: the rows a horizontal line at y = 32 covers in column 32 of a
        // 64x64 target, and a vertical one at x = 16.5 in column 16 of a 32x32 one.
        for (bits, rows) in &quantized {
            let width = f32::from_bits(*bits);
            let mut first_last = Vec::new();
            for (side, points, column) in [
                (64, [4.0, 32.0, 60.0, 32.0], 32),
                (32, [16.5, 4.0, 16.5, 28.0], 16),
            ] {
                let shot = Shot::one(
                    side,
                    1,
                    Shade::White,
                    Draw::line(PrimitiveType::Line, &points, width),
                );
                let drawn = lane
                    .draw(&shot)
                    .unwrap_or_else(|e| panic!("{name}: width {width}: {e}"));
                let covered: Vec<usize> = (0..side as usize)
                    .filter(|y| drawn.red[y * side as usize + column] != 0)
                    .collect();
                let (first, last) = match (covered.first(), covered.last()) {
                    (Some(&f), Some(&l)) => (f as i64, l as i64),
                    _ => (-1, -1),
                };
                first_last.push((first, last));
            }
            match announced_lines(&messages) {
                (0, 0) => {}
                (_, 0) => {
                    found.announced += 1;
                    continue;
                }
                _ => {
                    found.nearest += 1;
                    continue;
                }
            }
            let [(hf, hl), (vf, vl)] = [first_last[0], first_last[1]];
            let ours = format!("horizontal y=32 rows {hf}..{hl}, vertical x=16.5 rows {vf}..{vl}");
            assert_eq!(&ours, rows, "{name}: width {width} ({bits:08x})");
            found.matched += 1;
        }
        found.cases = cases.len() + quantized.len();
        assert_eq!(
            found.matched + found.announced + found.nearest,
            found.cases,
            "{found:?}"
        );
        // Every measured map was drawn or accounted for.
        assert!(
            checked.len() + found.announced + found.nearest >= blocks.len() - 2,
            "{found:?}"
        );
        eprintln!(
            "{name} (wideLines {wide_lines}, tessellationShader {tessellates}, {policy:?}): {found:?}; \
             {bands} draws as bands, {device_lines} by the device's lines, {nearest_lines} by them \
             at the nearest width"
        );
        // The routes ran: every wide draw took one; wherever the device has no wide lines and
        // tessellates, a line the band and a triangle's edges the nearest width.
        assert!(
            wide > 0 && bands + device_lines + nearest_lines == wide,
            "{bands} {device_lines} {nearest_lines} {wide}"
        );
        if !wide_lines && tessellates {
            assert_eq!(nearest_lines as usize, found.edges, "{found:?}");
            assert_eq!(bands + nearest_lines, wide);
        }
    }
    layer_reported_nothing_at_all(&messages);
    found
}

/// Every case under the default policy: the device's own lines where they draw the width,
/// announced where they draw it otherwise than Metal, else the band.
#[test]
fn wide_lines_draw_metal_s_parallelograms() {
    let found = run(DevicePolicy::default());
    // Width 1 draws as Metal's on every device; on one whose own lines differ from Metal's at
    // no width drawn, every case but the edges does.
    assert!(found.matched > 0, "{found:?}");
    let facts = found.facts.unwrap();
    if facts.tessellates && found.announced == found.announced_edges {
        assert_eq!(found.matched, found.cases - found.edges, "{found:?}");
    }
    // Without `wideLines` (MoltenVK) the edges are drawn 1 pixel wide, as Metal's at width 1.
    if facts.tessellates && !facts.wide_lines {
        assert_eq!(found.nearest, found.edges, "{found:?}");
        assert_eq!(found.nearest_matched, found.edges, "{found:?}");
    }
}

/// Every case under `DevicePolicy::exact_answers()`: the band wherever the device's lines differ
/// from Metal's, so no draw of lines is announced on a device with `tessellationShader`.
#[test]
fn wide_lines_draw_metal_s_parallelograms_under_exact_answers() {
    let found = run(DevicePolicy::default().exact_answers());
    assert!(found.matched > 0, "{found:?}");
    // Edges aside, which the band does not draw: drawn by the device's lines (announced) where
    // they draw the width, else by them at the nearest width (announced).
    let facts = found.facts.unwrap();
    if facts.tessellates {
        assert_eq!(found.announced, found.announced_edges, "{found:?}");
        assert_eq!(found.matched, found.cases - found.edges, "{found:?}");
    }
    if facts.tessellates && !facts.wide_lines {
        assert_eq!(found.nearest_matched, found.edges, "{found:?}");
    }
}

/// The lane's maps Metal drew at each width, the list's and the strip's together, differ from
/// every other width's (the strip alone draws alike at 0 and 0.5, through pixel centres): the
/// comparison tells the widths apart, so a draw at another width than the case's fails it.
#[test]
fn the_lane_s_maps_tell_the_widths_apart() {
    let (blocks, _) = measured();
    let maps: Vec<[&str; 2]> = LANE_WIDTHS
        .iter()
        .map(|w| {
            ["lines", "strip"].map(|name| {
                let block = blocks[&format!("LANE {name} {w}")].as_str();
                block.split_once('\n').unwrap().1
            })
        })
        .collect();
    for (i, a) in maps.iter().enumerate() {
        for (j, b) in maps.iter().enumerate().skip(i + 1) {
            assert_ne!(a, b, "widths {} and {}", LANE_WIDTHS[i], LANE_WIDTHS[j]);
        }
    }
}

/// Without `tessellationShader` a line at a width the device's lines do not draw runs them at the
/// nearest width they draw, announced and counted at each draw, where Metal draws the width: at
/// width 1 as Metal's map of the same lines at width 1. At width 1 it draws as Metal's. Each
/// encoder starts at 1.
#[test]
fn without_tessellation_lines_draw_at_the_nearest_width() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options
                .withheld_features
                .insert(VkFeature::TessellationShader);
            options.announce_each = WIDE_LINES.into_iter().collect();
        })
        .unwrap();
        let name = device.name().to_owned();
        let facts = LineFacts::of(&device);
        assert!(!facts.tessellates, "{name}: tessellationShader withheld");
        let mut lane = Lane::new(device);
        let points = [1.0, 3.0, 15.0, 3.0, 1.0, 8.0, 15.0, 15.0];
        let (blocks, _) = measured();
        let mut nearest_draws = 0;
        // Width 0 no device's lines draw; 2 those without `wideLines` do not.
        for width in [0.0, 2.0, 1.0] {
            let shot = Shot::one(
                16,
                1,
                Shade::White,
                Draw::line(PrimitiveType::Line, &points, width),
            );
            let before = lane.device.work_counts();
            let drawn = lane
                .draw(&shot)
                .unwrap_or_else(|e| panic!("{name}: width {width}: {e}"));
            let [band, own, nearest] = routed(&before, &lane.device.work_counts());
            let (_, announced) = announced_lines(&messages);
            assert_eq!(band, 0, "{name}: a band without tessellationShader");
            let drawable = facts.draws(width);
            let (expected_own, expected_nearest) = if width == 1.0 {
                (0, 0)
            } else if drawable {
                (1, 0)
            } else {
                (0, 1)
            };
            assert_eq!(
                (own, nearest, announced as u64),
                (expected_own, expected_nearest, expected_nearest),
                "{name}: width {width}"
            );
            nearest_draws += nearest;
            let at_one = width == 1.0 || (nearest == 1 && facts.nearest(width) == 1.0);
            if at_one {
                assert_eq!(
                    lane_block("lines", 1.0, &drawn.red),
                    blocks["LANE lines 1"],
                    "{name}: width {width}"
                );
            }
        }
        assert!(nearest_draws > 0, "{name}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// The consumer's draws of no complete primitive at a width other than 1, which Metal draws as
/// nothing and goes on (measured on the M3 Max): triangles and a triangle strip of 1 and 2
/// vertices, by vertices and by indices, in fill mode `Lines` at width 2 under the last provoking
/// vertex. None is refused, none draws a pixel, none takes a route of the width's or announces
/// one.
#[test]
fn a_draw_of_no_complete_primitive_draws_nothing() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.announce_each = WIDE_LINES.into_iter().collect();
        })
        .unwrap();
        let name = device.name().to_owned();
        let mut lane = Lane::new(device);
        let points = [4.0, 4.0, 28.0, 4.0, 16.0, 28.0];
        for primitive in [PrimitiveType::Triangle, PrimitiveType::TriangleStrip] {
            for count in [1, 2] {
                for indexed in [false, true] {
                    let draw = Draw {
                        fill: TriangleFillMode::Lines,
                        indexed,
                        provoking_last: true,
                        ..Draw::line(primitive, &points[..2 * count], 2.0)
                    };
                    let shot = Shot::one(32, 1, Shade::Count, draw);
                    let before = lane.device.work_counts();
                    let drawn = lane
                        .draw(&shot)
                        .unwrap_or_else(|e| panic!("{name}: {shot:?}: {e}"));
                    let routes = routed(&before, &lane.device.work_counts());
                    assert!(
                        drawn.red.iter().chain(&drawn.green).all(|&v| v == 0),
                        "{name}: {shot:?}: drew"
                    );
                    assert_eq!(routes, [0, 0, 0], "{name}: {shot:?}: a route of the width");
                    assert_eq!(announced_lines(&messages), (0, 0), "{name}: {shot:?}");
                }
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A whole triangle's edges in fill mode `Lines` at width 2 draw: by the device's own lines where
/// they draw width 2, else by them at the nearest width they draw, announced once per device and
/// counted at each draw; at width 1 (no `wideLines`: MoltenVK) as Metal's edges of the same
/// triangle at width 1, pixel for pixel.
#[test]
fn edges_draw_at_the_nearest_width_the_device_s_lines_draw() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let name = device.name().to_owned();
        let facts = LineFacts::of(&device);
        let mut lane = Lane::new(device);
        let (blocks, _) = measured();
        let triangle = [4.0, 4.0, 28.0, 4.0, 16.0, 28.0];
        let draw = Draw {
            fill: TriangleFillMode::Lines,
            ..Draw::line(PrimitiveType::Triangle, &triangle, 2.0)
        };
        let shot = Shot::one(32, 1, Shade::Count, draw);
        let title = "triangle fill Lines width 1, fragments";
        for round in 0..2 {
            let before = lane.device.work_counts();
            let drawn = lane
                .draw(&shot)
                .unwrap_or_else(|e| panic!("{name}: round {round}: {e}"));
            let [band, own, nearest] = routed(&before, &lane.device.work_counts());
            let (_, announced) = announced_lines(&messages);
            assert!(drawn.red.iter().any(|&v| v != 0), "{name}: drew nothing");
            assert_eq!(band, 0, "{name}: the band does not draw edges");
            if facts.draws(2.0) {
                assert_eq!((own, nearest, announced), (1, 0, 0), "{name}");
                continue;
            }
            // Announced at the first draw alone, counted at each.
            let first = usize::from(round == 0);
            assert_eq!(
                (own, nearest, announced),
                (0, 1, first),
                "{name}: round {round}"
            );
            if facts.nearest(2.0) == 1.0 {
                assert_eq!(
                    map(title, 32, &drawn.red, Shade::Count, 1),
                    blocks[title],
                    "{name}: round {round}"
                );
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}
