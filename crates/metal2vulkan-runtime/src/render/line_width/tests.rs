//! The line width's quantization, each committed device's route, and the draws that draw no
//! primitive (`render/line_width.rs`).

use super::{drawn, draws_primitives, route, Assembly, Drawn, LineRoute, LineWidth, Unrouted};
use crate::error::{Error, RuntimeGap};
use metal2vulkan::bind::{DrawCall, IndexBufferRef, IndirectRef, RewriteProgram};
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile, VkFeature};
use metal2vulkan::device::{VkLimit, WideLinePolicy};
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::{PrimitiveType, TriangleFillMode};
use metal2vulkan::pipeline::BandPrimitive;
use metal2vulkan::vk;

/// A route as the table below writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    /// `P`: as every line, at width 1.
    Plain,
    /// `E`: the device's own lines, exactly Metal's parallelogram.
    Exact,
    /// `A`: the device's own lines, announced as differing.
    Announced,
    /// `B`: the band.
    Band,
    /// `T`, `U`, `V`, `R`: the device's own lines at the nearest width they draw, announced, the
    /// band lacking `tessellationShader`, not drawing the draw's lines (a mesh's lines, a patch's
    /// edges), not its amplified views, or refused by the translator.
    Nearest(Unrouted),
}

impl Cell {
    fn parse(row: &str) -> Vec<Cell> {
        row.split_whitespace()
            .map(|code| match code {
                "P" => Cell::Plain,
                "E" => Cell::Exact,
                "A" => Cell::Announced,
                "B" => Cell::Band,
                "T" => Cell::Nearest(Unrouted::Untessellated),
                "U" => Cell::Nearest(Unrouted::Unbanded),
                "V" => Cell::Nearest(Unrouted::Amplified),
                "R" => Cell::Nearest(Unrouted::Refused),
                other => panic!("no cell {other}"),
            })
            .collect()
    }
}

/// A profile of the repository's, read as the runtime reads its device
/// (`DeviceDescription::from_khronos_json`), so this runs from the workspace checkout.
fn device(stem: &str) -> Device {
    let path = format!(
        "{}/../metal2vulkan-validation/profiles/{stem}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap();
    let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
    Device::new(description, DevicePolicy::default()).unwrap()
}

/// The width nearest the one Metal draws for `w` (`w`, 1/16 at 0) the device's lines draw: that
/// clamped into `lineWidthRange` where the device enables `wideLines`, else 1 (the module docs,
/// written apart from the route).
fn nearest(profile: &DeviceProfile, w: f32) -> f32 {
    let [low, high] = profile.limits().line_width_range_bits.map(f32::from_bits);
    let w = if w == 0.0 { 1.0 / 16.0 } else { w };
    match profile.enabled().features.contains(&VkFeature::WideLines) {
        true if w < low => low,
        true if w > high => high,
        true => w,
        false => 1.0,
    }
}

/// The cell of `drawn` at `width` on `profile`'s device under `policy`, amplified where
/// `amplifies`, of a pipeline state whose band the translator refuses where `refused`; each
/// route's own value checked (the device's width, the band's half width and primitive, the
/// nearest width).
fn cell(
    profile: &DeviceProfile,
    policy: WideLinePolicy,
    width: LineWidth,
    drawn: Drawn,
    (amplifies, refused): (bool, bool),
) -> Cell {
    let w = width.pixels();
    match route(
        profile,
        policy,
        width,
        drawn,
        (amplifies, false, false),
        refused,
    ) {
        LineRoute::Plain => Cell::Plain,
        LineRoute::Device { width, differs } => {
            assert_eq!(width, w);
            match differs {
                false => Cell::Exact,
                true => Cell::Announced,
            }
        }
        LineRoute::Band { half, primitive } => {
            assert_eq!(half, if w == 0.0 { 1.0 / 32.0 } else { w / 2.0 });
            let banded = match drawn {
                Drawn::Lines => BandPrimitive::Lines,
                _ => BandPrimitive::Triangles,
            };
            assert_eq!(primitive, banded);
            Cell::Band
        }
        LineRoute::Nearest { width, why } => {
            assert_eq!(width, nearest(profile, w), "the nearest width to {w}");
            Cell::Nearest(why)
        }
    }
}

/// The route of every draw at a width other than 1 on each committed device profile the
/// table names, under both `WideLinePolicy` values: a list of lines and a triangle's edges in
/// fill mode `Lines` (of a list, a strip and a fan), alike; an emulated mesh's lines and a
/// tessellated patch's edges, which the band does not draw, under either value; both under
/// vertex amplification above 1; and both of a pipeline state whose band the translator refuses,
/// which take the nearest width where they would take the band. The widths: 0, 0.5, 1, 1.5, a
/// width on the device's `lineWidthGranularity` and one off it where Metal has one (every
/// sixteenth is on a granularity of 1/16 or finer), the top of `lineWidthRange` and a sixteenth
/// above it where Metal draws them, and 16. Where neither the device's lines nor the band draw
/// the width, the device's lines at the nearest width they draw: no draw is refused. A
/// rate-mapped pass's lines and an execution's command's, whose width Metal's answer is not
/// measured for, take the nearest width at every width other than 1.
#[test]
fn each_device_routes_each_width_as_its_lines_allow() {
    // (profile, widths, lines and edges Device, lines and edges Exact, mesh lines and patch
    // edges, amplified lines and edges)
    let table: [(&str, &[f32], [&str; 4]); 11] = [
        // wideLines, [0, 8] by 1/8, non-strict parallelograms, tessellation.
        (
            "amd-780m-radv",
            &[0.0, 0.5, 1.0, 1.5, 2.125, 2.0625, 8.0, 8.0625, 16.0],
            [
                "B A P E E A E B B",
                "B B P E E B E B B",
                "U A P E E A E U U",
                "V A P E E A E V V",
            ],
        ),
        (
            "amd-rdna3-radv",
            &[0.0, 0.5, 1.0, 1.5, 2.125, 2.0625, 8.0, 8.0625, 16.0],
            [
                "B A P E E A E B B",
                "B B P E E B E B B",
                "U A P E E A E U U",
                "V A P E E A E V V",
            ],
        ),
        // wideLines, [0, 8] by 1/128, not stated as parallelograms.
        (
            "intel-arl-anv",
            &[0.0, 0.5, 1.0, 1.5, 2.0625, 8.0, 8.0625, 16.0],
            [
                "B A P A A A B B",
                "B B P B B B B B",
                "U A P A A A U U",
                "V A P A A A V V",
            ],
        ),
        // wideLines, [1, 255] by 1/128, strict.
        (
            "lavapipe",
            &[0.0, 0.5, 1.0, 1.5, 2.0625, 16.0],
            ["B B P A A A", "B B P B B B", "U U P A A A", "V V P A A A"],
        ),
        // wideLines, [1, 64] by 1/16, strict.
        (
            "nvidia-rtx5080-laptop",
            &[0.0, 0.5, 1.0, 1.5, 2.0625, 16.0],
            ["B B P A A A", "B B P B B B", "U U P A A A", "V V P A A A"],
        ),
        // No wideLines ([1, 1]), tessellation.
        (
            "moltenvk-m3",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["B B P B B B", "B B P B B B", "U U P U U U", "V V P V V V"],
        ),
        // No wideLines, no tessellation.
        (
            "swiftshader",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["T T P T T T", "T T P T T T", "U U P U U U", "T T P T T T"],
        ),
        (
            "vk12-floor",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["T T P T T T", "T T P T T T", "U U P U U U", "T T P T T T"],
        ),
        (
            "roadmap-2022",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["T T P T T T", "T T P T T T", "U U P U U U", "T T P T T T"],
        ),
        (
            "lunarg-desktop-baseline-2023",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["B B P B B B", "B B P B B B", "U U P U U U", "V V P V V V"],
        ),
        (
            "amd-780m-hawkpoint-radv",
            &[0.0, 0.5, 1.0, 1.5, 2.125, 2.0625, 8.0, 8.0625, 16.0],
            [
                "B A P E E A E B B",
                "B B P E E B E B B",
                "U A P E E A E U U",
                "V A P E E A E V V",
            ],
        ),
    ];
    let lines = drawn(
        Assembly::Primitives(PrimitiveType::Line),
        TriangleFillMode::Fill,
        false,
        None,
        true,
    );
    let edges_of = |assembly| drawn(assembly, TriangleFillMode::Lines, false, None, true);
    let edges = edges_of(Assembly::Primitives(PrimitiveType::Triangle));
    for assembly in [
        Assembly::Primitives(PrimitiveType::TriangleStrip),
        Assembly::Private(RewriteProgram::TriangleFan),
        Assembly::Private(RewriteProgram::Quads),
    ] {
        assert_eq!(edges_of(assembly), Drawn::Edges);
    }
    let mesh_lines = drawn(
        Assembly::Meshes,
        TriangleFillMode::Fill,
        false,
        Some(2),
        true,
    );
    let patch_edges = edges_of(Assembly::Patches);
    assert_eq!(
        (lines, edges, mesh_lines, patch_edges),
        (Drawn::Lines, Drawn::Edges, Drawn::Other, Drawn::Other)
    );
    // A mesh of triangles or points draws no line filled, whether the device draws the mesh or
    // it is emulated (`GraphicsState::mesh_corners`); a mesh of triangles draws its edges in fill
    // mode `Lines`.
    let mesh = |fill, corners| drawn(Assembly::Meshes, fill, false, corners, true);
    assert_eq!(
        [
            mesh(TriangleFillMode::Fill, Some(3)),
            mesh(TriangleFillMode::Fill, Some(1)),
            mesh(TriangleFillMode::Lines, Some(3)),
        ],
        [Drawn::Nothing, Drawn::Nothing, Drawn::Other]
    );
    for (stem, widths, rows) in table {
        let device = device(stem);
        let profile = device.profile();
        let limits = profile.limits();
        let features = &profile.enabled().features;
        let facts = format!(
            "wideLines {}, tessellationShader {}, lineWidthRange {:?}, granularity {}, \
             strictLines {}, parallelogram {}",
            features.contains(&VkFeature::WideLines),
            features.contains(&VkFeature::TessellationShader),
            limits.line_width_range_bits.map(f32::from_bits),
            f32::from_bits(limits.line_width_granularity_bits),
            limits.strict_lines,
            limits.non_strict_wide_lines_use_parallelogram,
        );
        let at = |policy, drawn, (amplifies, refused)| -> Vec<Cell> {
            (widths.iter())
                .map(|&w| {
                    cell(
                        profile,
                        policy,
                        LineWidth::of(w),
                        drawn,
                        (amplifies, refused),
                    )
                })
                .collect()
        };
        let [device_lines, exact_lines, other, amplified] = rows.map(Cell::parse);
        // Refused, either policy routes as `Device` does without the band: its band's cells
        // are the nearest width's.
        let refused = |row: &[Cell]| -> Vec<Cell> {
            (row.iter())
                .map(|&c| match c {
                    Cell::Band => Cell::Nearest(Unrouted::Refused),
                    c => c,
                })
                .collect()
        };
        let mut found = Vec::new();
        for (policy, row) in [
            (WideLinePolicy::Device, &device_lines),
            (WideLinePolicy::Exact, &exact_lines),
        ] {
            for (what, drawn) in [("lines", lines), ("edges", edges)] {
                found.push((
                    what,
                    policy,
                    "",
                    at(policy, drawn, (false, false)),
                    row.clone(),
                ));
                found.push((
                    what,
                    policy,
                    ", amplified",
                    at(policy, drawn, (true, false)),
                    amplified.clone(),
                ));
                found.push((
                    what,
                    policy,
                    ", refused",
                    at(policy, drawn, (false, true)),
                    refused(&device_lines),
                ));
            }
            for (what, drawn) in [("mesh lines", mesh_lines), ("patch edges", patch_edges)] {
                for place in [(false, false), (true, false), (false, true)] {
                    found.push((what, policy, "", at(policy, drawn, place), other.clone()));
                }
            }
        }
        for (what, policy, how, cells, expected) in found {
            assert_eq!(
                cells, expected,
                "{stem}, {what}{how}, {policy:?}, widths {widths:?} ({facts})"
            );
        }
        // A rate-mapped pass and an execution's command draw every line at a width other than 1
        // at the nearest width, edges too, and every one at 1 as every line.
        for (place, why) in [
            ((false, true, false), Unrouted::RateMapped),
            ((false, false, true), Unrouted::Executed),
        ] {
            for drawn in [lines, edges] {
                for w in [0.0, 0.5, 2.0, 16.0] {
                    let routed = route(
                        profile,
                        WideLinePolicy::Exact,
                        LineWidth::of(w),
                        drawn,
                        place,
                        false,
                    );
                    let width = nearest(profile, w);
                    assert_eq!(routed, LineRoute::Nearest { width, why }, "{stem}: {w}");
                }
                let routed = route(
                    profile,
                    WideLinePolicy::Exact,
                    LineWidth::ONE,
                    drawn,
                    place,
                    false,
                );
                assert_eq!(routed, LineRoute::Plain, "{stem}");
            }
        }
    }
}

/// A draw whose own band variant the translator refuses (`Unsupported(WideLines)`: not
/// implemented, or a limit such as `maxTessellationEvaluationOutputComponents` a varying of the
/// draw's variant passes) takes the nearest width, routed again as a draw of a state whose band is
/// refused (`LineRoute::refused`), never failing: on every committed device profile, under both
/// `WideLinePolicy` values, for lines and a triangle's edges at each width the band draws. Routed
/// again it draws no band, so the draw settles again once at most. Any other answer stands, and a
/// route other than the band takes no refusal.
#[test]
fn a_draw_whose_band_the_translator_refuses_takes_the_nearest_width() {
    let refusals = [
        Error::Unsupported(Unsupported::not_implemented(Feature::WideLines)),
        Error::Unsupported(Unsupported::needs(
            Feature::WideLines,
            Capability::Limit {
                limit: VkLimit::MaxTessellationEvaluationOutputComponents,
                required: 128,
                available: 124,
            },
        )),
    ];
    let others = [
        Error::Unsupported(Unsupported::not_implemented(Feature::Tessellation)),
        Error::NotImplemented(RuntimeGap::WideLines),
    ];
    let edges = drawn(
        Assembly::Primitives(PrimitiveType::Triangle),
        TriangleFillMode::Lines,
        false,
        None,
        true,
    );
    let lines = drawn(
        Assembly::Primitives(PrimitiveType::Line),
        TriangleFillMode::Fill,
        false,
        None,
        true,
    );
    let mut bands = 0;
    for stem in [
        "amd-780m-radv",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "lavapipe",
        "nvidia-rtx5080-laptop",
        "moltenvk-m3",
        "swiftshader",
        "vk12-floor",
        "roadmap-2022",
        "lunarg-desktop-baseline-2023",
        "amd-780m-hawkpoint-radv",
    ] {
        let device = device(stem);
        let profile = device.profile();
        for policy in [WideLinePolicy::Device, WideLinePolicy::Exact] {
            for drawn in [lines, edges] {
                for w in [0.0, 0.5, 1.0, 1.5, 2.5, 8.0625, 16.0] {
                    let width = LineWidth::of(w);
                    let at =
                        |refused| route(profile, policy, width, drawn, Default::default(), refused);
                    let what = format!("{stem}, {policy:?}, {drawn:?} at {w}");
                    let first = at(false);
                    for other in &others {
                        assert_eq!(first.refused(other), None, "{what}: {other:?}");
                    }
                    let LineRoute::Band { primitive, .. } = first else {
                        for refusal in &refusals {
                            assert_eq!(first.refused(refusal), None, "{what}: {first:?}");
                        }
                        continue;
                    };
                    bands += 1;
                    for refusal in &refusals {
                        assert_eq!(first.refused(refusal), Some(primitive), "{what}");
                    }
                    // Routed again: the device's lines where they draw the width, else the
                    // nearest width, the translator's refusal the reason; no band.
                    let again = at(true);
                    let nearest = LineRoute::Nearest {
                        width: nearest(profile, width.pixels()),
                        why: Unrouted::Refused,
                    };
                    assert!(
                        again == nearest || matches!(again, LineRoute::Device { .. }),
                        "{what}: routed again {again:?}"
                    );
                    for refusal in &refusals {
                        assert_eq!(again.refused(refusal), None, "{what}: {again:?}");
                    }
                }
            }
        }
    }
    assert!(bands > 0, "no profile drew a band");
}

/// A direct draw whose vertices or indices complete no primitive of its type draws nothing at
/// any width, as Metal's: of every line and triangle type and each private one, list or strip,
/// indexed or not, of one instance or several, in either fill mode. One primitive's vertices
/// draw it; an indirect draw's count, a patch draw's and a mesh draw's are the GPU's, so each
/// may draw one. The route is the plain one on every device, where a draw of one primitive at
/// that width takes another.
#[test]
fn a_draw_of_no_complete_primitive_draws_nothing() {
    // (assembly, the vertices of its first primitive)
    let assemblies = [
        (Assembly::Primitives(PrimitiveType::Line), 2),
        (Assembly::Primitives(PrimitiveType::LineStrip), 2),
        (Assembly::Primitives(PrimitiveType::Triangle), 3),
        (Assembly::Primitives(PrimitiveType::TriangleStrip), 3),
        (Assembly::Private(RewriteProgram::LineLoop), 2),
        (Assembly::Private(RewriteProgram::TriangleFan), 3),
        (Assembly::Private(RewriteProgram::Quads), 4),
        (Assembly::Private(RewriteProgram::QuadStrip), 4),
    ];
    let direct = |count, instance_count| {
        [
            DrawCall::Primitives {
                vertex_start: 5,
                vertex_count: count,
                instance_count,
                base_instance: 0,
            },
            DrawCall::Indexed {
                index_type: vk::IndexType::UINT16,
                index_buffer: IndexBufferRef::new(0, 64),
                index_count: count,
                index_offset: 0,
                instance_count,
                base_vertex: 0,
                base_instance: 0,
            },
        ]
    };
    let indirect = [
        DrawCall::Indirect {
            indirect: IndirectRef::new(0, 0),
            indexed: None,
        },
        DrawCall::Indirect {
            indirect: IndirectRef::new(0, 0),
            indexed: Some((vk::IndexType::UINT32, IndexBufferRef::new(0, 64), 0)),
        },
    ];
    let devices = ["moltenvk-m3", "swiftshader", "amd-780m-radv", "lavapipe"].map(device);
    let wide = LineWidth::of(2.0);
    for (assembly, first) in assemblies {
        for instances in [1, 3] {
            for count in 1..first {
                for draw in direct(count, instances) {
                    assert!(!draws_primitives(assembly, &draw), "{draw:?}");
                    for fill in [TriangleFillMode::Fill, TriangleFillMode::Lines] {
                        let drawn = drawn(assembly, fill, false, None, false);
                        assert_eq!(drawn, Drawn::Nothing, "{draw:?}");
                        for device in &devices {
                            for policy in [WideLinePolicy::Device, WideLinePolicy::Exact] {
                                let routed = route(
                                    device.profile(),
                                    policy,
                                    wide,
                                    drawn,
                                    (true, false, false),
                                    true,
                                );
                                assert_eq!(routed, LineRoute::Plain, "{draw:?}");
                            }
                        }
                    }
                }
            }
            for draw in direct(first, instances) {
                assert!(draws_primitives(assembly, &draw), "{draw:?}");
                let edges = drawn(assembly, TriangleFillMode::Lines, false, None, true);
                assert_ne!(edges, Drawn::Nothing, "{draw:?}");
                for device in &devices {
                    let routed = route(
                        device.profile(),
                        WideLinePolicy::Device,
                        wide,
                        edges,
                        (false, false, false),
                        false,
                    );
                    assert_ne!(routed, LineRoute::Plain, "{draw:?}");
                }
            }
        }
        for draw in &indirect {
            assert!(draws_primitives(assembly, draw), "{draw:?}");
        }
    }
    // Points draw from one vertex; patches and meshes count on the GPU.
    for count in [1, 2] {
        for draw in direct(count, 1) {
            let points = Assembly::Primitives(PrimitiveType::Point);
            assert!(draws_primitives(points, &draw));
        }
    }
    let patches = DrawCall::Patches {
        control_points: 3,
        patch_start: 0,
        patch_count: 1,
        patch_index: None,
        instance_count: 1,
        base_instance: 0,
        control_point_index: None,
    };
    assert!(draws_primitives(Assembly::Patches, &patches));
}

/// The widths Metal draws for the widths measured (the module docs): floored to sixteenths,
/// clamped to `[0, 16]`, NaN and every width not above 0 as 0.
#[test]
fn the_widths_are_metal_s_measured_ones() {
    let cases: [(f32, f32); 22] = [
        (1.0, 1.0),
        (1.0625, 1.0625),
        (1.03125, 1.0),
        (1.000_061, 1.0),
        (0.999_939, 0.9375),
        (0.9375, 0.9375),
        (3.0, 3.0),
        (3.03125, 3.0),
        (3.0625, 3.0625),
        (2.96875, 2.9375),
        (15.9375, 15.9375),
        (15.96875, 15.9375),
        (16.0, 16.0),
        (17.0, 16.0),
        (1e30, 16.0),
        (f32::INFINITY, 16.0),
        (-0.0, 0.0),
        (-1.0, 0.0),
        (f32::NEG_INFINITY, 0.0),
        (f32::NAN, 0.0),
        (0.0625, 0.0625),
        (0.03125, 0.0),
    ];
    for (given, drawn) in cases {
        assert_eq!(LineWidth::of(given).pixels(), drawn, "{given}");
    }
    assert_eq!(LineWidth::of(1.0), LineWidth::ONE);
}
