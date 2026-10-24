//! Tile threadgroups: which thread of a tile dispatch is which, and what a tile state answers.
//!
//! Ported from the runtime lane `runtime_lane/tile_threadgroups.rs`. That lane is the clearest case
//! in the survey for porting: its expectations are **formulas** rather than a table —
//! `position(simdgroup, lane)`, `thread_index(x, y)` and `quad_order(qw, qh)`, each a model of how
//! Metal places the threads of a tile threadgroup — and the only thing tying those formulas to
//! Metal is a comment saying a probe measured them on an M4 and on an M3 Max under macOS 26.7.
//! Both of those measurements are gone, and the second is of an older release.
//!
//! So Metal's answer here is **run**, not copied. Record mode runs each case on native Metal on the
//! chip and build the recording is keyed to, and the case records the words Metal's own threads
//! wrote. The lane's formulas stay exactly where they are, as the readable model, and this module's
//! own tests check the recorded words against them: that is the cross-check the port
//! owes the lane, and it is what turns a comment into evidence.
//!
//! What a case observes is typed: every call of the pass by the Metal selector it makes, the
//! command buffer's status and error, the records the tile function wrote as a buffer of 32-bit
//! words, and each attachment's whole extent as bytes under a layout that reads it as its own
//! element.
//!
//! The owned fixtures are `k10ts_tile_order` (one `R32Float` attachment; each thread adds one to
//! its cell and records its position, lane, simdgroup and its simdgroup's first lane's position at
//! its thread index), `k10ts_tile_grid` (three 32-bit attachments; each thread stages its 2x2 cells
//! and records twelve words of its builtins) and `ttg_tile_simd_blur` (one `RGBA16Float`
//! attachment; a SIMD blur of twelve `simd_shuffle_and_fill_down` taps).
//!
//! Nothing here can fault a GPU: every pass is one or four tiles of at most 32 by 32 threads over
//! an attachment of at most 64 by 64 texels, and every dispatch is inside Metal's own 1024-thread
//! threadgroup.
//!
//! Where a device holds fewer threads than Metal does, the dispatch Metal runs is refused typed as
//! the device's gap, and the run stops at that call, so the row reads as the refusal it is rather
//! than as one divergence per cell a refused dispatch never wrote.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{
    Abort, Case, Failure, Handle, Input, MetalSource, Side, TextureRole, TextureSpec,
    TilePipelineSpec, TileStep,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The tile's side, and the threadgroup's, as the probe drew them.
const SIDE: u32 = 32;

/// The fixture whose tile function records which thread is which.
const ORDER: &str = "k10ts_tile_order";

/// The fixture whose tile function stages a whole tile from fewer threads.
const GRID: &str = "k10ts_tile_grid";

/// The fixture whose tile function is a SIMD blur.
const BLUR: &str = "ttg_tile_simd_blur";

/// `k10ts_tile_order`'s one attachment.
const ORDER_ATTACHMENTS: [PixelFormat; 1] = [PixelFormat::R32Float];

/// `k10ts_tile_grid`'s three attachments: a float first, then two words.
const GRID_ATTACHMENTS: [PixelFormat; 3] = [
    PixelFormat::R32Float,
    PixelFormat::R32Uint,
    PixelFormat::R32Uint,
];

/// `ttg_tile_simd_blur`'s one attachment.
const BLUR_ATTACHMENTS: [PixelFormat; 1] = [PixelFormat::RGBA16Float];

/// The threadgroup shapes of even sides the lane dispatches in a 32x32 tile, each in a pass of its
/// own over its own attachment and record buffer, as the lane runs them: the ones whose simdgroup
/// is a whole row of quads (16x2, 32x2) and the ones whose columns, rows and blocks end partly.
const EVEN_SHAPES: &[&[[u32; 2]]] = &[
    &[[16, 2]],
    &[[32, 2]],
    &[[6, 6]],
    &[[10, 10]],
    &[[30, 30]],
    &[[24, 8]],
    &[[2, 18]],
    &[[14, 4]],
];

/// The degrade notices a tile render pass may announce, each of which a case of this family
/// loosens its notices site by. Metal announces none, so a notice our side announced is the one
/// field of these cases that may differ at all — every record and every cell compares exactly.
///
/// It is a list and not a device query, because a case never branches on the device: a case asks by
/// running and reads what came back. The list holds exactly the kinds measured on the two venues of
/// this round, MoltenVK 1.4.2 on this M3 Max and lavapipe in the local container. A third venue
/// that announces a kind not here fails the case until its kind is added, which is the model
/// working: an announced difference nobody named is not an announced difference.
pub const TILE_PASS_NOTICES: &[Degradation] = &[
    // The attachments a tile dispatch stages are copied into its scratch before the dispatch and
    // back after it (`render/tile_dispatch.rs`), which is what a tile dispatch costs on a device
    // whose imageblock is not Metal's.
    Degradation::TileDispatchScratch,
    // The pass is split at the barrier the tile dispatch's scratch needs.
    Degradation::RenderPassSplit,
    // A render target of a format whose NaN encoding the device copies its own way: the blur's
    // `RGBA16Float` attachment.
    Degradation::DeviceNanEncoding,
    // A device that splits a fragment quad at a clipped edge where Metal keeps it whole. Announced
    // on lavapipe. Nothing of these cases reads a quad's grouping -- the tile order is the answer,
    // and it compares exactly -- so the kind is declared and no value is loosened.
    Degradation::QuadsSplitByClipping,
];

/// The allowances a case of this family keeps: one over its notices site per kind of
/// [`TILE_PASS_NOTICES`], and nothing anywhere else.
pub fn tile_notice_allowances() -> Vec<Allowance> {
    TILE_PASS_NOTICES
        .iter()
        .map(|kind| {
            Allowance::new(
                format!("case/{NOTICES_SITE}/*"),
                Loosening::Unchecked,
                Justification::notice(*kind),
            )
        })
        .collect()
}

/// The blur's source extent, its origin in it, and its color scale, as the lane's probe set them.
const SOURCE: [u32; 2] = [160, 160];
const ORIGIN: [u32; 2] = [3, 5];
const SCALE: f32 = 0.5;

/// Which family a case is of. Each is one shape of pass, so a refusal of one costs no other its
/// sites.
enum Shape {
    /// The states a tile descriptor makes and what each answers.
    ThreadLimits,
    /// `k10ts_tile_order` under a state of `limit` threads and `tile_sized`: one pass per element
    /// of `passes`, each over its own attachment and record buffer, dispatching that element's
    /// threadgroups per 32x32 tile in order.
    Order {
        passes: &'static [&'static [[u32; 2]]],
        limit: Option<u32>,
        tile_sized: bool,
    },
    /// `k10ts_tile_grid` dispatched 8x8 per 16x16 tile over a 40x24 pass, tile-sized.
    Grid,
    /// `ttg_tile_simd_blur` over 2x2 tiles of 32x32, the simdgroups the tile's rows or, transposed,
    /// its columns.
    Blur { transpose: bool },
}

/// A case of the family.
pub struct TileThreadgroups {
    name: &'static str,
    shape: Shape,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<TileThreadgroups> {
    vec![
        TileThreadgroups {
            name: "tile-state-thread-limits",
            shape: Shape::ThreadLimits,
        },
        TileThreadgroups {
            name: "tile-order-32x32",
            shape: Shape::Order {
                passes: &[&[[SIDE, SIDE]]],
                limit: None,
                tile_sized: false,
            },
        },
        TileThreadgroups {
            name: "tile-order-past-the-states-limit",
            shape: Shape::Order {
                passes: &[&[[SIDE, SIDE], [16, 16]]],
                limit: Some(256),
                tile_sized: false,
            },
        },
        TileThreadgroups {
            name: "tile-order-even-threadgroups",
            shape: Shape::Order {
                passes: EVEN_SHAPES,
                limit: None,
                tile_sized: false,
            },
        },
        TileThreadgroups {
            name: "tile-order-tile-sized-16x16",
            shape: Shape::Order {
                passes: &[&[[16, 16]]],
                limit: None,
                tile_sized: true,
            },
        },
        TileThreadgroups {
            name: "tile-grid-threadgroup-is-the-tile",
            shape: Shape::Grid,
        },
        TileThreadgroups {
            name: "tile-simd-blur-rows",
            shape: Shape::Blur { transpose: false },
        },
        TileThreadgroups {
            name: "tile-simd-blur-columns",
            shape: Shape::Blur { transpose: true },
        },
    ]
}

/// The site name one `k10ts_tile_order` pass takes: the threadgroups it dispatches, in order.
fn pass_name(dispatches: &[[u32; 2]]) -> String {
    dispatches
        .iter()
        .map(|[w, h]| format!("{w}x{h}"))
        .collect::<Vec<_>>()
        .join("-then-")
}

/// The bytes `k10ts_tile_order`'s 32x32 `R32Float` attachment starts at: cell `i` holds `i`.
fn order_start() -> Vec<u8> {
    (0..SIDE * SIDE)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect()
}

/// The records `k10ts_tile_order` writes: two 32-bit words a thread, at its thread index.
const ORDER_RECORD_BYTES: usize = (SIDE * SIDE * 8) as usize;

/// The slots `k10ts_tile_grid` writes: 512 records of twelve 32-bit words.
const GRID_RECORD_BYTES: usize = 512 * 48;

/// `k10ts_tile_grid`'s pass extent, which is three tiles across and two down of its 16x16 tile,
/// the last partial on both axes.
const GRID_AREA: [u32; 2] = [40, 24];

/// The tile `k10ts_tile_grid`'s pass is in.
const GRID_TILE: u32 = 16;

/// The blur's twelve weights: 1, 2, 3 repeating, as the lane's `weight`.
fn weight(tap: u32) -> u32 {
    tap % 3 + 1
}

/// The blur's source texel at (`x`, `y`): small integers, so every sum is exact in `half`.
fn source_texel(x: u32, y: u32) -> [u32; 4] {
    [(x + 2 * y) % 8, (3 * x + y) % 8, (x * y) % 8, (x ^ y) % 8]
}

/// `value`'s bits as a `half`, for the values the blur's fixtures hold (small integers, halves and
/// 9) which are all exact.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & 0x007f_ffff;
    if value == 0.0 {
        return sign;
    }
    let half_exponent = exponent + 15;
    debug_assert!(
        (1..=30).contains(&half_exponent) && mantissa & 0x1fff == 0,
        "{value} is not exact as a half"
    );
    sign | ((half_exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

/// The blur's parameter words: the origin, the texels to fill past the simdgroup's end, whether the
/// simdgroups are the tile's columns, and the color scale.
fn blur_params(transpose: bool) -> Vec<u8> {
    let fill = if transpose { [0u16, 32] } else { [32, 0] };
    [
        ORIGIN[0] as u16,
        ORIGIN[1] as u16,
        fill[0],
        fill[1],
        u16::from(transpose),
        half_bits(SCALE),
    ]
    .iter()
    .flat_map(|word| word.to_le_bytes())
    .collect()
}

/// The blur's twelve weights as `half`s.
fn blur_weights() -> Vec<u8> {
    (0..12)
        .flat_map(|tap| half_bits(weight(tap) as f32).to_le_bytes())
        .collect()
}

/// The blur's source bytes: every texel of a 160x160 `RGBA16Float` texture.
fn blur_source() -> Vec<u8> {
    (0..SOURCE[1])
        .flat_map(|y| (0..SOURCE[0]).map(move |x| (x, y)))
        .flat_map(|(x, y)| source_texel(x, y))
        .flat_map(|channel| half_bits(channel as f32).to_le_bytes())
        .collect()
}

/// The blur's target bytes before the pass: 9 in every channel, which no blurred pixel holds.
fn blur_start() -> Vec<u8> {
    std::iter::repeat_n(half_bits(9.0).to_le_bytes(), (4 * 4 * SIDE * SIDE) as usize)
        .flatten()
        .collect()
}

/// The layout a buffer of 32-bit records is read under.
fn record_layout(bytes: usize) -> ByteLayout {
    ByteLayout::buffer(0, bytes as u64, Element::U32)
}

/// The layout an attachment's whole level 0 is read under: `channels` elements of `element` a texel,
/// packed, so a diff names the row and the element inside it.
fn attachment_layout(size: [u32; 2], element: Element, channels: u32) -> ByteLayout {
    let row_pitch = size[0] * channels * element.size();
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
        },
        element,
        row_offset: 0,
        row_elements: size[0] * channels,
        row_pitch,
        rows: size[1],
        image_pitch: row_pitch * size[1],
        images: 1,
        gaps: Gaps::Compared,
    }
}

impl TileThreadgroups {
    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        Fixture::new(match self.shape {
            Shape::ThreadLimits | Shape::Order { .. } => ORDER,
            Shape::Grid => GRID,
            Shape::Blur { .. } => BLUR,
        })
    }

    /// The tile pipeline descriptors the case makes, each under the name its site takes.
    fn specs(&self) -> Vec<(String, TilePipelineSpec<'static>)> {
        match self.shape {
            // The three states the lane makes of one descriptor: Metal's own maximum, a limit
            // below it, and a limit that is not a threadgroup of even sides.
            Shape::ThreadLimits => vec![
                (
                    "state-unlimited".to_owned(),
                    TilePipelineSpec::new(ORDER, &ORDER_ATTACHMENTS),
                ),
                (
                    "state-limited-512".to_owned(),
                    TilePipelineSpec::new(ORDER, &ORDER_ATTACHMENTS).limited(512),
                ),
                (
                    "state-limited-1000".to_owned(),
                    TilePipelineSpec::new(ORDER, &ORDER_ATTACHMENTS).limited(1000),
                ),
            ],
            Shape::Order {
                limit, tile_sized, ..
            } => {
                let mut spec = TilePipelineSpec::new(ORDER, &ORDER_ATTACHMENTS);
                if let Some(limit) = limit {
                    spec = spec.limited(limit);
                }
                if tile_sized {
                    spec = spec.tile_sized();
                }
                vec![("state".to_owned(), spec)]
            }
            Shape::Grid => vec![(
                "state".to_owned(),
                TilePipelineSpec::new(GRID, &GRID_ATTACHMENTS).tile_sized(),
            )],
            Shape::Blur { .. } => vec![(
                "state".to_owned(),
                TilePipelineSpec::new(BLUR, &BLUR_ATTACHMENTS),
            )],
        }
    }

    /// Run the case's pass or passes on `side`.
    fn passes(&self, side: &mut dyn Side, state: Handle, sites: &mut Sites) -> Result<(), Abort> {
        match self.shape {
            Shape::ThreadLimits => Ok(()),
            Shape::Order { passes, .. } => {
                // Every shape is its own pass over its own attachment and its own record buffer,
                // as the lane runs them, so one shape's records cannot reach another's.
                for dispatches in passes {
                    self.order_pass(side, state, &pass_name(dispatches), dispatches, sites)?;
                }
                Ok(())
            }
            Shape::Grid => self.grid_pass(side, state, sites),
            Shape::Blur { transpose } => self.blur_pass(side, state, transpose, sites),
        }
    }

    /// One `k10ts_tile_order` pass: a fresh 32x32 `R32Float` attachment holding its cell indices, a
    /// zeroed record buffer, and `dispatches` dispatched per 32x32 tile in order.
    fn order_pass(
        &self,
        side: &mut dyn Side,
        state: Handle,
        at: &str,
        dispatches: &[[u32; 2]],
        sites: &mut Sites,
    ) -> Result<(), Abort> {
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let start = order_start();
        let target = side
            .texture_2d_in_role(
                &TextureSpec::new(
                    PixelFormat::R32Float,
                    [SIDE, SIDE],
                    SIDE * 4,
                    TextureRole::RenderTarget,
                ),
                &start,
            )
            .map_err(|failure| stop(format!("{at}/target"), failure))?;
        let records = side
            .zeroed_buffer(ORDER_RECORD_BYTES)
            .map_err(|failure| stop(format!("{at}/records"), failure))?;
        let mut steps = vec![
            TileStep::Pipeline(state),
            TileStep::TileBuffer {
                buffer: records,
                offset: 0,
                index: 0,
            },
        ];
        for [w, h] in dispatches {
            steps.push(TileStep::DispatchThreadsPerTile([*w, *h, 1]));
        }
        let node = self.pass(side, at, &[target], [SIDE, SIDE], &steps, sites)?;
        let node = node.with(Node::leaf(
            "records",
            Observed::bytes(
                side.read_buffer(records, 0, ORDER_RECORD_BYTES)
                    .map_err(|failure| stop(format!("{at}/records-after"), failure))?,
                record_layout(ORDER_RECORD_BYTES),
            ),
        ));
        let node = node.with(Node::leaf(
            "cells",
            Observed::bytes(
                side.read_texture(target, [SIDE, SIDE], SIDE * 4)
                    .map_err(|failure| stop(format!("{at}/cells"), failure))?,
                attachment_layout([SIDE, SIDE], Element::F32, 1),
            ),
        ));
        sites.push(node);
        Ok(())
    }

    /// The `k10ts_tile_grid` pass: three 40x24 attachments in 16x16 tiles, 8x8 threads a tile.
    fn grid_pass(
        &self,
        side: &mut dyn Side,
        state: Handle,
        sites: &mut Sites,
    ) -> Result<(), Abort> {
        let at = "grid";
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let [w, h] = GRID_AREA;
        let mut targets = Vec::new();
        for (index, (format, bytes)) in GRID_ATTACHMENTS.iter().zip(grid_starts()).enumerate() {
            targets.push(
                side.texture_2d_in_role(
                    &TextureSpec::new(*format, GRID_AREA, w * 4, TextureRole::RenderTarget),
                    &bytes,
                )
                .map_err(|failure| stop(format!("{at}/target-{index}"), failure))?,
            );
        }
        let records = side
            .zeroed_buffer(GRID_RECORD_BYTES)
            .map_err(|failure| stop(format!("{at}/records"), failure))?;
        let steps = vec![
            TileStep::Pipeline(state),
            TileStep::TileBuffer {
                buffer: records,
                offset: 0,
                index: 0,
            },
            TileStep::DispatchThreadsPerTile([8, 8, 1]),
        ];
        let node = self.pass(side, at, &targets, [GRID_TILE, GRID_TILE], &steps, sites)?;
        let mut node = node.with(Node::leaf(
            "records",
            Observed::bytes(
                side.read_buffer(records, 0, GRID_RECORD_BYTES)
                    .map_err(|failure| stop(format!("{at}/records-after"), failure))?,
                record_layout(GRID_RECORD_BYTES),
            ),
        ));
        for (index, target) in targets.iter().enumerate() {
            let element = if index == 0 {
                Element::F32
            } else {
                Element::U32
            };
            node = node.with(Node::leaf(
                format!("attachment-{index}"),
                Observed::bytes(
                    side.read_texture(*target, GRID_AREA, w * 4)
                        .map_err(|failure| stop(format!("{at}/attachment-{index}"), failure))?,
                    attachment_layout([w, h], element, 1),
                ),
            ));
        }
        sites.push(node);
        Ok(())
    }

    /// The `ttg_tile_simd_blur` pass: a 64x64 `RGBA16Float` attachment of 2x2 tiles of 32x32, every
    /// pixel 9 before the pass, dispatched 32x32 per tile — the dispatch Metal runs.
    fn blur_pass(
        &self,
        side: &mut dyn Side,
        state: Handle,
        transpose: bool,
        sites: &mut Sites,
    ) -> Result<(), Abort> {
        let at = if transpose { "columns" } else { "rows" };
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let area = [2 * SIDE, 2 * SIDE];
        let source = side
            .texture_2d_in_role(
                &TextureSpec::new(
                    PixelFormat::RGBA16Float,
                    SOURCE,
                    SOURCE[0] * 8,
                    TextureRole::Read,
                ),
                &blur_source(),
            )
            .map_err(|failure| stop(format!("{at}/source"), failure))?;
        let target = side
            .texture_2d_in_role(
                &TextureSpec::new(
                    PixelFormat::RGBA16Float,
                    area,
                    area[0] * 8,
                    TextureRole::RenderTarget,
                ),
                &blur_start(),
            )
            .map_err(|failure| stop(format!("{at}/target"), failure))?;
        let steps = vec![
            TileStep::Pipeline(state),
            TileStep::TileTexture {
                texture: source,
                index: 0,
            },
            TileStep::TileBytes {
                bytes: blur_params(transpose),
                index: 0,
            },
            TileStep::TileBytes {
                bytes: blur_weights(),
                index: 1,
            },
            TileStep::DispatchThreadsPerTile([SIDE, SIDE, 1]),
        ];
        let node = self.pass(side, at, &[target], [SIDE, SIDE], &steps, sites)?;
        let node = node.with(Node::leaf(
            "pixels",
            Observed::bytes(
                side.read_texture(target, area, area[0] * 8)
                    .map_err(|failure| stop(format!("{at}/pixels"), failure))?,
                attachment_layout(area, Element::F16, 4),
            ),
        ));
        sites.push(node);
        Ok(())
    }

    /// Encode and run one tile pass, and emit the node holding every call's answer, the command
    /// buffer's status and its error.
    ///
    /// A step the side refused stops the run at that call: the refusal is the fact, and the cells a
    /// refused dispatch never wrote are not divergences of their own.
    fn pass(
        &self,
        side: &mut dyn Side,
        at: &str,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
        sites: &mut Sites,
    ) -> Result<Node, Abort> {
        let outcome = side
            .tile_pass(targets, tile, steps)
            .map_err(|failure| Abort {
                at: format!("{at}/pass"),
                failure,
            })?;
        let mut calls = Node::branch("calls", Vec::new());
        for (index, (step, refused)) in steps.iter().zip(&outcome.refused).enumerate() {
            calls = calls.with(Node::leaf(
                format!("{index}-{}", step.call()),
                Observed::value(match refused {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        let mut node = Node::branch(at, Vec::new()).with(calls);
        if let Some((site, failure)) =
            steps
                .iter()
                .zip(&outcome.refused)
                .enumerate()
                .find_map(|(index, (step, refused))| {
                    refused.as_ref().map(|failure| {
                        (
                            format!("{at}/calls/{index}-{}", step.call()),
                            failure.clone(),
                        )
                    })
                })
        {
            sites.push(node);
            return Err(Abort { at: site, failure });
        }
        node = node.with(Node::leaf(
            "status",
            Observed::value(Value::ok(outcome.status.clone())),
        ));
        node = node.with(Node::leaf(
            "command-buffer-error",
            Observed::value(match &outcome.error {
                Some(error) => Value::Variant {
                    name: "Some".to_owned(),
                    payload: Box::new(error.clone()),
                },
                None => Value::unit_variant("None"),
            }),
        ));
        Ok(node)
    }
}

/// The bytes `k10ts_tile_grid`'s three attachments start at, in attachment order: `y * 64 + x` as a
/// float, `y * 1000 + x + 7` as a word, and `0x5555_5555` everywhere.
fn grid_starts() -> Vec<Vec<u8>> {
    let [w, h] = GRID_AREA;
    let pixels = || (0..h).flat_map(move |y| (0..w).map(move |x| (x, y)));
    vec![
        pixels()
            .flat_map(|(x, y)| ((y * 64 + x) as f32).to_le_bytes())
            .collect(),
        pixels()
            .flat_map(|(x, y)| (y * 1000 + x + 7).to_le_bytes())
            .collect(),
        std::iter::repeat_n(0x5555_5555u32.to_le_bytes(), (w * h) as usize)
            .flatten()
            .collect(),
    ]
}

impl Case for TileThreadgroups {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        let mut allowances = tile_notice_allowances();
        // The record buffer a tile-order or tile-grid pass writes its order into is `Shared`
        // ([`crate::replay::shared_memory_notice`]), and the RTX 5080 was measured announcing it
        // for each of those five rows. The thread-limit case answers getters and the two blur
        // cases were measured announcing nothing of the kind on either venue, so neither declares
        // it: a declaration no run justifies is one nobody is checking.
        if matches!(self.shape, Shape::Order { .. } | Shape::Grid) {
            allowances.push(crate::replay::shared_memory_notice());
        }
        allowances
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: fixture.entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        for (name, spec) in self.specs() {
            inputs.push(Input::bytes(format!("tile-pipeline:{name}"), &spec.words()));
        }
        match self.shape {
            Shape::ThreadLimits => {}
            Shape::Order { passes, .. } => {
                let mut dispatches = Vec::new();
                for one in passes {
                    dispatches.extend(pass_name(one).as_bytes());
                    dispatches.push(b'\n');
                }
                inputs.push(Input::bytes("dispatches", &dispatches));
                inputs.push(Input::bytes("attachment-0", &order_start()));
                inputs.push(Input::bytes(
                    "buffer:0",
                    format!("{ORDER_RECORD_BYTES} zeroed bytes").as_bytes(),
                ));
            }
            Shape::Grid => {
                inputs.push(Input::bytes("dispatches", b"8x8\n"));
                inputs.push(Input::bytes(
                    "tile",
                    format!(
                        "{GRID_TILE}x{GRID_TILE} over {}x{}",
                        GRID_AREA[0], GRID_AREA[1]
                    )
                    .as_bytes(),
                ));
                for (index, bytes) in grid_starts().into_iter().enumerate() {
                    inputs.push(Input::bytes(format!("attachment-{index}"), &bytes));
                }
            }
            Shape::Blur { transpose } => {
                inputs.push(Input::bytes(
                    "dispatches",
                    format!("{SIDE}x{SIDE}\n").as_bytes(),
                ));
                inputs.push(Input::bytes("texture:0", &blur_source()));
                inputs.push(Input::bytes("bytes:0", &blur_params(transpose)));
                inputs.push(Input::bytes("bytes:1", &blur_weights()));
                inputs.push(Input::bytes("attachment-0", &blur_start()));
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let mut first = None;
        for (name, spec) in self.specs() {
            let state = crate::case::made(sites, &name, side.tile_pipeline(library, &spec))?;
            let facts = side.tile_facts(state).map_err(|failure| Abort {
                at: format!("{name}-facts"),
                failure,
            })?;
            sites.push(facts.node(format!("{name}-facts")));
            first = first.or(Some(state));
        }
        let state = first.ok_or_else(|| Abort {
            at: "state".to_owned(),
            failure: Failure::named("NoState", "the case makes no tile pipeline"),
        })?;
        self.passes(side, state, sites)
    }
}

#[cfg(test)]
mod tests {
    //! The cross-check the port owes the lane: the words Metal's own threads wrote, as the
    //! recordings hold them, against the lane's formulas for where Metal places a tile
    //! threadgroup's threads.
    //!
    //! The formulas are the lane's (`runtime_lane/tile_threadgroups.rs`), restated here because a
    //! conformance case may not depend on the validation crate. They are the claim under test, not
    //! a helper: if a recorded word disagrees with one, either Apple moved or the lane's model was
    //! never Metal's, and the test says which thread.

    use super::*;
    use crate::recording::Recording;
    use crate::transcript::{Node, Transcript};

    /// The lane's `position`: Metal's place of lane `lane` of simdgroup `simdgroup` in a 32x32 tile
    /// threadgroup. Two columns 16 wide; a column's simdgroup `s` is its 8x4 block
    /// `(s mod 2, s / 2)`, and lane `l` is thread `l mod 4` of the block's quad `l / 4`.
    fn position(simdgroup: u32, lane: u32) -> (u32, u32) {
        let (column, s) = (simdgroup / 16, simdgroup % 16);
        let (quad, within) = (lane / 4, lane % 4);
        (
            column * 16 + (s % 2) * 8 + (quad % 4) * 2 + within % 2,
            (s / 2) * 4 + (quad / 4) * 2 + within / 2,
        )
    }

    /// The lane's `thread_index`: Metal's `thread_index_in_threadgroup` of (`x`, `y`) in a 32-wide
    /// tile threadgroup.
    fn thread_index(x: u32, y: u32) -> u32 {
        ((y / 2) * (SIDE / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2
    }

    /// The lane's `quad_order`: Metal's quad order of a threadgroup of `qw` by `qh` 2x2 quads.
    fn quad_order(qw: u32, qh: u32) -> Vec<(u32, u32)> {
        let mut order = Vec::new();
        for c0 in (0..qw).step_by(8) {
            let c1 = (c0 + 8).min(qw);
            for r0 in (0..qh).step_by(2) {
                let r1 = (r0 + 2).min(qh);
                for b0 in (c0..c1).step_by(4) {
                    let b1 = (b0 + 4).min(c1);
                    for qy in r0..r1 {
                        for qx in b0..b1 {
                            order.push((qx, qy));
                        }
                    }
                }
            }
        }
        order
    }

    /// The place (`x`, `y`) packs into a record word as.
    fn packed((x, y): (u32, u32)) -> u32 {
        x | (y << 8)
    }

    /// The one outcome of `name`'s recorded Metal answer. A case of this family is settled, so its
    /// runs folded into one.
    fn only_outcome(name: &str) -> Transcript {
        let recording: Recording = crate::recording::read(name)
            .unwrap_or_else(|why| panic!("{name}: no recording ({why})"));
        let answer = recording
            .answers
            .first()
            .unwrap_or_else(|| panic!("{name}: no answer"));
        assert_eq!(answer.outcomes.len(), 1, "{name}: more than one outcome");
        answer.outcomes[0].transcript.clone()
    }

    /// The node at `path` (names under the root, `/`-separated) of `transcript`.
    fn at<'a>(transcript: &'a Transcript, path: &str) -> &'a Node {
        let mut node = &transcript.root;
        for name in path.trim_start_matches("case/").split('/') {
            node = node
                .child(name)
                .unwrap_or_else(|| panic!("no site {path} (stopped at {name})"));
        }
        node
    }

    /// The 32-bit words a bytes site holds.
    fn words(node: &Node) -> Vec<u32> {
        let crate::transcript::Value::Bytes(bytes) = &node.observed.value else {
            panic!("{} holds no bytes", node.name);
        };
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }

    /// The floats a bytes site holds.
    fn floats(node: &Node) -> Vec<f32> {
        words(node).into_iter().map(f32::from_bits).collect()
    }

    /// Metal's own 32x32 tile threadgroup, as the recording holds it, is the lane's formula: every
    /// thread's record at the index the formula gives, and every cell added one once.
    #[test]
    fn the_recorded_32x32_threadgroup_is_the_lanes_formula() {
        let recorded = only_outcome("tile-order-32x32");
        let records = words(at(&recorded, "case/32x32/records"));
        for simdgroup in 0..SIDE {
            for lane in 0..32 {
                let (x, y) = position(simdgroup, lane);
                let index = thread_index(x, y) as usize;
                assert_eq!(
                    records[2 * index],
                    packed((x, y)) | (lane << 16) | (simdgroup << 24),
                    "simdgroup {simdgroup} lane {lane}"
                );
                assert_eq!(
                    records[2 * index + 1],
                    packed(position(simdgroup, 0)),
                    "simdgroup {simdgroup} lane {lane}: its simdgroup's first lane"
                );
            }
        }
        for (i, cell) in floats(at(&recorded, "case/32x32/cells")).iter().enumerate() {
            assert_eq!(*cell, i as f32 + 1.0, "cell {i}");
        }
    }

    /// Metal's threadgroups of even sides, as the recording holds them, are the lane's quad order,
    /// and the cells outside a threadgroup keep what they held.
    #[test]
    fn the_recorded_even_threadgroups_are_the_lanes_quad_order() {
        let recorded = only_outcome("tile-order-even-threadgroups");
        for shape in EVEN_SHAPES {
            let [w, h] = shape[0];
            let name = pass_name(shape);
            let records = words(at(&recorded, &format!("case/{name}/records")));
            let cells = floats(at(&recorded, &format!("case/{name}/cells")));
            let order = quad_order(w / 2, h / 2);
            let place = |k: usize, within: u32| {
                let (qx, qy) = order[k];
                (qx * 2 + within % 2, qy * 2 + within / 2)
            };
            for (k, &(qx, qy)) in order.iter().enumerate() {
                let (simdgroup, first) = (k / 8, place(k / 8 * 8, 0));
                for within in 0..4 {
                    let (x, y) = place(k, within);
                    let lane = (k % 8) as u32 * 4 + within;
                    let index = ((qy * (w / 2) + qx) * 4 + within) as usize;
                    assert_eq!(
                        records[2 * index],
                        packed((x, y)) | (lane << 16) | ((simdgroup as u32) << 24),
                        "{w}x{h}: thread ({x}, {y})"
                    );
                    assert_eq!(
                        records[2 * index + 1],
                        packed(first),
                        "{w}x{h}: thread ({x}, {y}), its simdgroup's first lane"
                    );
                }
            }
            // Every record past the threadgroup's threads stays zero, and every cell outside it
            // keeps the index it held.
            for (at, word) in records.iter().enumerate().skip((w * h * 2) as usize) {
                assert_eq!(*word, 0, "{w}x{h}: word {at} past the threadgroup");
            }
            for (i, cell) in cells.iter().enumerate() {
                let (x, y) = (i as u32 % SIDE, i as u32 / SIDE);
                let added = if x < w && y < h { 1.0 } else { 0.0 };
                assert_eq!(*cell, i as f32 + added, "{w}x{h}: cell ({x}, {y})");
            }
        }
    }

    /// Metal's tile-sized threadgroup, as the recording holds it, answers the tile as its
    /// threadgroup and stages every cell of it, which is the lane's claim.
    #[test]
    fn the_recorded_tile_sized_threadgroup_answers_the_tile() {
        let recorded = only_outcome("tile-grid-threadgroup-is-the-tile");
        let records = words(at(&recorded, "case/grid/records"));
        let cell_tag = |tg: [u32; 2], c: [u32; 2]| {
            0xA000_0000 | (tg[1] << 20) | (tg[0] << 16) | (c[1] << 8) | c[0]
        };
        for tile in 0..6u32 {
            let tg = [tile % 3, tile / 3];
            for tidx in 0..64u32 {
                let slot = ((tg[1] * 4 + tg[0]) * 64 + tidx) as usize;
                let record = &records[slot * 12..slot * 12 + 12];
                // An 8-wide threadgroup's index is its quads' order, which is
                // `simdgroup * 32 + lane` too.
                let (quad, within) = (tidx / 4, tidx % 4);
                let tid = [(quad % 4) * 2 + within % 2, (quad / 4) * 2 + within / 2];
                let want = [
                    tid[0] | (tid[1] << 16),
                    tg[0] | (tg[1] << 16),
                    tg[0] * 16 + tid[0],
                    tg[1] * 16 + tid[1],
                    16 | (16 << 16),
                    48,
                    32,
                    tidx,
                    0,
                    16 | (16 << 16),
                ];
                assert_eq!(record[..10], want, "tile {tg:?} thread {tidx}");
                assert_eq!(
                    record[10],
                    cell_tag(tg, [15 - 2 * tid[0], 15 - 2 * tid[1]]),
                    "tile {tg:?} thread {tidx}: the mirrored cell's tag"
                );
                assert_eq!(record[11], 0xC0DE_0000 | slot as u32);
            }
        }
        let [w, h] = GRID_AREA;
        let starts = grid_starts();
        let read = |bytes: &[u8]| -> Vec<u32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect()
        };
        let (f0, v0) = (read(&starts[0]), read(&starts[1]));
        let planes: Vec<Vec<u32>> = (0..3)
            .map(|index| words(at(&recorded, &format!("case/grid/attachment-{index}"))))
            .collect();
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let (tg, c) = ([x / 16, y / 16], [x % 16, y % 16]);
                let k = 1 + (c[0] % 2) + 2 * (c[1] % 2);
                assert_eq!(
                    planes[0][i],
                    (f32::from_bits(f0[i]) * 2.0 + 1.0).to_bits(),
                    "f ({x}, {y})"
                );
                assert_eq!(planes[1][i], v0[i] * 3 + k, "v ({x}, {y})");
                assert_eq!(planes[2][i], cell_tag(tg, c), "tag ({x}, {y})");
            }
        }
    }

    /// A dispatch the **bind stage** refuses because the device's threadgroup is too small is the
    /// device's gap, not ours, so the verdict is `LacksFeature`.
    ///
    /// This is the shape lavapipe gives the SIMD blur: its emulated 32-lane simdgroups take 960
    /// threads of that pipeline, so the 32x32 tile threadgroup Metal runs is refused by
    /// `check_local` with `Feature::LocalSize` and a `Capability::Limit`. Before the classifier
    /// read `Error::Bind`, that refusal was neither announced nor counted as a device gap, which
    /// is exactly what the `LacksFeature` verdict exists to prevent: a venue that honestly cannot
    /// run a case must be its own outcome rather than a failure or a pass.
    ///
    /// The other direction is the half that keeps the fix honest: what the bind stage has **not
    /// built** stays ours to answer for, so the two mesh rows of `known-gaps.tsv`, whose
    /// errors are `Error::Bind(Unsupported(MeshShading, NotImplemented))`, remain refusals.
    #[test]
    fn a_dispatch_the_device_cannot_take_is_the_devices_gap() {
        use metal2vulkan::error::{BindError, Feature, RuntimeDriverDefect, Unsupported};
        use metal2vulkan_runtime::Error;
        // `Capability`'s payload vocabulary is private to the translator, so a `Needs::AnyOf`
        // cannot be built from here; the test states the other device-gap `Needs` instead. What
        // the classifier reads is the `Needs` and the variant that carries it, not which
        // capability, so this is the same claim.
        let limited = Unsupported::runtime_driver_defect(
            Feature::LocalSize { axis: None },
            RuntimeDriverDefect::NormalizedDepthPastUnitRange,
        );
        let lacks = crate::verdict::device_gap(&Error::Bind(BindError::Unsupported(limited)));
        assert!(
            lacks
                .as_deref()
                .is_some_and(|why| why.contains("LocalSize")),
            "{lacks:?}"
        );
        // Not built is not the device's gap, whichever variant carries it.
        let not_built = Unsupported::not_implemented(Feature::MeshShading);
        assert_eq!(
            crate::verdict::device_gap(&Error::Bind(BindError::Unsupported(not_built.clone()))),
            None
        );
        assert_eq!(
            crate::verdict::device_gap(&Error::Unsupported(not_built)),
            None
        );
    }

    /// Every case of the family states its inputs and its tile descriptors, and loosens **only**
    /// its notices site, by a degrade notice it names.
    ///
    /// This is the family's whole tolerance story in one assertion: a tile dispatch's thread
    /// layout, its records and its cells are exact or they are a bug, and the one field that may
    /// differ from Metal is a notice our side announced. A case that ever loosened a record or a
    /// cell would fail here.
    #[test]
    fn every_case_loosens_its_notices_alone() {
        for case in all() {
            let inputs = case.inputs().unwrap();
            assert!(
                inputs
                    .iter()
                    .any(|input| input.name.starts_with("tile-pipeline:")),
                "{}: no tile pipeline in its inputs",
                case.name()
            );
            assert_eq!(case.metal_source(), MetalSource::Run);
            case.shaders().unwrap();
            let allowances = case.allowances();
            // Every case declares the family's own list, and the five that read a record buffer
            // back declare the notice a device's own memory model provokes besides. Nothing else
            // is named and nothing is loosened, so a value tolerance added later fails here.
            let reads_a_buffer = matches!(case.shape, Shape::Order { .. } | Shape::Grid);
            assert_eq!(
                allowances.len(),
                TILE_PASS_NOTICES.len() + usize::from(reads_a_buffer),
                "{}",
                case.name()
            );
            for allowance in &allowances {
                assert_eq!(
                    allowance.path,
                    format!("case/{NOTICES_SITE}/*"),
                    "{}: an allowance outside the notices site",
                    case.name()
                );
                assert_eq!(allowance.loosening, Loosening::Unchecked, "{allowance:?}");
                match &allowance.justified_by {
                    Justification::Degradation(code) => assert!(
                        TILE_PASS_NOTICES.iter().any(|kind| kind.code() == code)
                            || code == Degradation::SharedInSystemMemory.code(),
                        "{}: {code} is neither a tile-pass notice nor the one a device's own \
                         memory model provokes",
                        case.name()
                    ),
                    other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                }
            }
        }
    }
}
