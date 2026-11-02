//! The line width's quantization and each committed device's route (`render/line_width.rs`).

use super::{drawn, route, Assembly, Drawn, LineRoute, LineWidth};
use crate::error::{Error, RuntimeGap};
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy, VkFeature, WideLinePolicy};
use metal2vulkan::error::{Capability, Feature, Needs};
use metal2vulkan::metal::{PrimitiveType, TriangleFillMode};

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
    /// `T`: refused `Unsupported(WideLines)`, needing `tessellationShader`.
    NeedsTessellation,
    /// `N`: refused `NotImplemented(WideLines)`.
    Gap,
}

impl Cell {
    fn parse(row: &str) -> Vec<Cell> {
        row.split_whitespace()
            .map(|code| match code {
                "P" => Cell::Plain,
                "E" => Cell::Exact,
                "A" => Cell::Announced,
                "B" => Cell::Band,
                "T" => Cell::NeedsTessellation,
                "N" => Cell::Gap,
                other => panic!("no cell {other}"),
            })
            .collect()
    }
}

/// The cell of `drawn` at `width` on `profile`'s device under `policy`, amplified where
/// `amplifies`; each route's own value checked (the device's width, the band's half width).
fn cell(
    profile: &metal2vulkan::device::DeviceProfile,
    policy: WideLinePolicy,
    width: LineWidth,
    drawn: Drawn,
    amplifies: bool,
) -> Cell {
    let w = width.pixels();
    match route(profile, policy, width, drawn, (amplifies, false)) {
        Ok((LineRoute::Plain, false)) => Cell::Plain,
        Ok((LineRoute::Device { width }, differs)) => {
            assert_eq!(width, w);
            match differs {
                false => Cell::Exact,
                true => Cell::Announced,
            }
        }
        Ok((LineRoute::Band { half }, false)) => {
            assert_eq!(half, if w == 0.0 { 1.0 / 32.0 } else { w / 2.0 });
            Cell::Band
        }
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::WideLines);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::TessellationShader
                )]])
            );
            Cell::NeedsTessellation
        }
        Err(Error::NotImplemented(RuntimeGap::WideLines)) => Cell::Gap,
        other => panic!("{other:?}"),
    }
}

/// The route of every draw at a width other than 1 on each committed device profile the
/// table names, under both `WideLinePolicy` values: a list of lines, a triangle's edges in
/// fill mode `Lines` and an emulated mesh's lines (which the band does not draw, under either
/// value), and a list of lines under vertex amplification above 1. The widths: 0, 0.5, 1,
/// 1.5, a width on the device's `lineWidthGranularity` and one off it where Metal has one
/// (every sixteenth is on a granularity of 1/16 or finer), the top of `lineWidthRange` and a
/// sixteenth above it where Metal draws them, and 16. Read from the profiles as the runtime
/// reads its device (`DeviceDescription::from_khronos_json`); the repository's profiles, so
/// this runs from the workspace checkout.
#[test]
fn each_device_routes_each_width_as_its_lines_allow() {
    // (profile, widths, list Device, list Exact, edges and mesh lines, amplified list)
    let table: [(&str, &[f32], [&str; 4]); 11] = [
        // wideLines, [0, 8] by 1/8, non-strict parallelograms, tessellation.
        (
            "amd-780m-radv",
            &[0.0, 0.5, 1.0, 1.5, 2.125, 2.0625, 8.0, 8.0625, 16.0],
            [
                "B A P E E A E B B",
                "B B P E E B E B B",
                "N A P E E A E N N",
                "N A P E E A E N N",
            ],
        ),
        (
            "amd-rdna3-radv",
            &[0.0, 0.5, 1.0, 1.5, 2.125, 2.0625, 8.0, 8.0625, 16.0],
            [
                "B A P E E A E B B",
                "B B P E E B E B B",
                "N A P E E A E N N",
                "N A P E E A E N N",
            ],
        ),
        // wideLines, [0, 8] by 1/128, not stated as parallelograms.
        (
            "intel-arl-anv",
            &[0.0, 0.5, 1.0, 1.5, 2.0625, 8.0, 8.0625, 16.0],
            [
                "B A P A A A B B",
                "B B P B B B B B",
                "N A P A A A N N",
                "N A P A A A N N",
            ],
        ),
        // wideLines, [1, 255] by 1/128, strict.
        (
            "lavapipe",
            &[0.0, 0.5, 1.0, 1.5, 2.0625, 16.0],
            ["B B P A A A", "B B P B B B", "N N P A A A", "N N P A A A"],
        ),
        // wideLines, [1, 64] by 1/16, strict.
        (
            "nvidia-rtx5080-laptop",
            &[0.0, 0.5, 1.0, 1.5, 2.0625, 16.0],
            ["B B P A A A", "B B P B B B", "N N P A A A", "N N P A A A"],
        ),
        // No wideLines ([1, 1]), tessellation.
        (
            "moltenvk-m3",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["B B P B B B", "B B P B B B", "N N P N N N", "N N P N N N"],
        ),
        // No wideLines, no tessellation.
        (
            "swiftshader",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["T T P T T T", "T T P T T T", "N N P N N N", "T T P T T T"],
        ),
        (
            "vk12-floor",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["T T P T T T", "T T P T T T", "N N P N N N", "T T P T T T"],
        ),
        (
            "roadmap-2022",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["T T P T T T", "T T P T T T", "N N P N N N", "T T P T T T"],
        ),
        (
            "lunarg-desktop-baseline-2023",
            &[0.0, 0.5, 1.0, 1.0625, 1.5, 16.0],
            ["B B P B B B", "B B P B B B", "N N P N N N", "N N P N N N"],
        ),
        (
            "amd-780m-hawkpoint-radv",
            &[0.0, 0.5, 1.0, 1.5, 2.125, 2.0625, 8.0, 8.0625, 16.0],
            [
                "B A P E E A E B B",
                "B B P E E B E B B",
                "N A P E E A E N N",
                "N A P E E A E N N",
            ],
        ),
    ];
    let lines = drawn(
        Assembly::Primitives(PrimitiveType::Line),
        TriangleFillMode::Fill,
        false,
        None,
    );
    let edges = drawn(
        Assembly::Primitives(PrimitiveType::Triangle),
        TriangleFillMode::Lines,
        false,
        None,
    );
    let mesh_lines = drawn(Assembly::Meshes, TriangleFillMode::Fill, false, Some(2));
    assert_eq!(
        (lines, edges, mesh_lines),
        (Drawn::Lines, Drawn::Other, Drawn::Other)
    );
    for (stem, widths, rows) in table {
        let path = format!(
            "{}/../metal2vulkan-validation/profiles/{stem}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap();
        let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let device = Device::new(description, DevicePolicy::default()).unwrap();
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
        let at = |policy, drawn, amplifies| -> Vec<Cell> {
            (widths.iter())
                .map(|&w| cell(profile, policy, LineWidth::of(w), drawn, amplifies))
                .collect()
        };
        let [device_lines, exact_lines, other, amplified] = rows.map(Cell::parse);
        let found = [
            (
                "lines, Device",
                at(WideLinePolicy::Device, lines, false),
                &device_lines,
            ),
            (
                "lines, Exact",
                at(WideLinePolicy::Exact, lines, false),
                &exact_lines,
            ),
            (
                "edges, Device",
                at(WideLinePolicy::Device, edges, false),
                &other,
            ),
            (
                "edges, Exact",
                at(WideLinePolicy::Exact, edges, false),
                &other,
            ),
            (
                "mesh lines, Device",
                at(WideLinePolicy::Device, mesh_lines, false),
                &other,
            ),
            (
                "mesh lines, Exact",
                at(WideLinePolicy::Exact, mesh_lines, false),
                &other,
            ),
            (
                "amplified, Device",
                at(WideLinePolicy::Device, lines, true),
                &amplified,
            ),
            (
                "amplified, Exact",
                at(WideLinePolicy::Exact, lines, true),
                &amplified,
            ),
        ];
        for (what, cells, expected) in found {
            assert_eq!(
                &cells, expected,
                "{stem}, {what}, widths {widths:?} ({facts})"
            );
        }
        // A rate-mapped pass draws no line at a width other than 1, and every one at 1.
        for w in [0.0, 2.0] {
            let routed = route(
                profile,
                WideLinePolicy::Device,
                LineWidth::of(w),
                lines,
                (false, true),
            );
            assert!(
                matches!(routed, Err(Error::NotImplemented(RuntimeGap::WideLines))),
                "{stem}: {routed:?}"
            );
        }
        assert!(matches!(
            route(
                profile,
                WideLinePolicy::Device,
                LineWidth::ONE,
                lines,
                (false, true)
            ),
            Ok((LineRoute::Plain, false))
        ));
    }
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
