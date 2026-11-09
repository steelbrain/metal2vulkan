//! A coverage chain through framebuffer fetch, as a path renderer builds one (the consumer's icon
//! notch), on the device the cfg selects, under the validation layer.
//!
//! `dsx_coverage_chain` draws, over an 81x48 `RGBA16Float` target cleared to 0 beside an
//! `RG16Float` coverage attachment cleared to 0, a third `RGBA16Float` attachment and a
//! `Depth32Float_Stencil8` one, six cells (the consumer's layout: two of them at x = 0, the
//! others off the 8-pixel grid), each scissored to its cell, in three runs of six draws (each run
//! in the consumer's order of the cells):
//!
//! - the coverage: additive blending sums weights in the coverage's green, either
//!   - `chain_edges`, instanced rectangles: each cell's shape is its columns but the first and
//!     the last, every row: one rectangle of weight 1, or (`overlap`) four that sum to 1, the top
//!     ten rows and the rest of weight 1 each and two over the whole shape of weight 0.5 and -0.5;
//!   - or `chain_spans` and `chain_span_edges`, span quads: each cell's shape is a disc in pixel
//!     steps ([`disc`]), whose widest rows reach the cell's left edge (x = 0 for two cells), its
//!     outline cut into vertical edges, each a quad from the edge to the cell's right edge, of
//!     weight 1 on the left and -1 on the right. The quads of 13 edges are one instance, drawn as
//!     indexed triangle strips that restart after each quad (64 indices), four instances a cell:
//!     the edges past the outline's collapse to (-2, -2). Each quad reaches half a pixel past its
//!     edge to the left, above and below, so the quads of two edges meeting at a row overlap there
//!     (each fragment adds its share of the pixel, 0 outside its edge's rows);
//! - `chain_coverage`: a rectangle over the cell reads the sum back by framebuffer fetch under
//!   early fragment tests and leaves its absolute value, saturated, in red;
//! - `chain_color_discarding` (or `chain_color`): a rectangle over the cell reads that coverage by
//!   framebuffer fetch, discards where it is 0, and writes its colour times it to the target;
//!   or a gradient's fill (`chain_gradient`), as a path renderer's gradient ends the chain: it
//!   reads both the target and the coverage by framebuffer fetch, discards where the coverage is
//!   0, adds a 32x32 `RGBA16Float` noise texture's texel (0.0625, 0.0625, 0.0625, 0), sampled at
//!   the pixel, discards there again, and writes the sum where the coverage is 1 (else mixed over
//!   the target by the coverage). Its variants sample at LOD 0 (`chain_gradient_lod0`) or before
//!   the first discard (`chain_gradient_hoisted`).
//!
//! Metal's answer is the colour (0.25, 0.5, 0.75, 1) in each shape, or the gradient's (0.3125,
//! 0.5625, 0.8125, 1), and the clear elsewhere: every value is a dyadic fraction a `half` holds
//! exactly, the same with or without the discard. The
//! cases switch off, one at a time, what may lose a block of it (the consumer's r24 pass 266 lost
//! one 8x4 block of each cell at x = 0, at its shape's leftmost point): the coverage attachment
//! memoryless or not, the overlapping blended rectangles, the discard, and for the span quads the
//! outset, the strips' restarts (a triangle list instead), the shape's reach to the cell's left
//! edge (inset by a pixel), and the collapsed edges (only the instances the outline needs). Three
//! cases draw to a target over a managed no-copy buffer that stands for a whole IOSurface (48
//! KiB), its plane at offset 0, rows 768 bytes apart (`Buffer::new_texture_with_iosurface_plane`,
//! usage `ShaderRead | RenderTarget`), as the consumer makes its target. The gradient's cases
//! (the consumer's notch is a gradient's fill of this shape, ending a chain like this one, losing
//! 8x4 blocks through the pixel interlock on MoltenVK) run under the default depth state and, as
//! the consumer draws, under `GreaterEqual` without writes over a depth cleared to 0.

use super::{form, library};
use crate::runtime_lane::buffer_with;
use crate::runtime_lane::host_memory::{no_copy, Mapping};
use crate::runtime_lane::memoryless::memoryless_options;
use crate::runtime_lane::resolves::{lane, quiet, Lane};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, IndexType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, Deallocator, Error, IoSurfacePlane, Library, LoadAction,
    PipelineOptions, Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ScissorRect, StoreAction, Texture, TextureDescriptor, WorkCounts,
};

/// The target's width and height.
const W: u64 = 81;
const H: u64 = 48;

/// The cells, `(x0, y0, x1, y1)`: the consumer's layout.
const CELLS: [[u32; 4]; 6] = [
    [0, 0, 21, 21],
    [30, 0, 51, 21],
    [61, 0, 81, 21],
    [0, 27, 21, 48],
    [31, 27, 52, 48],
    [61, 28, 81, 48],
];

/// The order of [`CELLS`] in each run: the coverage, its fetch, the colour.
const ORDER: [[usize; 6]; 3] = [[1, 3, 4, 5, 0, 2], [4, 1, 2, 3, 0, 5], [4, 0, 1, 2, 3, 5]];

/// The colour, as `half` bits: (0.25, 0.5, 0.75, 1).
const COLOR: [u16; 4] = [0x3400, 0x3800, 0x3A00, 0x3C00];

/// The noise texture's every texel, as `half` bits: (0.0625, 0.0625, 0.0625, 0).
const NOISE: [u16; 4] = [0x2C00, 0x2C00, 0x2C00, 0];

/// The noise texture's width and height.
const NOISE_SIZE: u64 = 32;

/// A gradient's fill's colour, [`COLOR`] plus [`NOISE`] times its alpha, as `half` bits:
/// (0.3125, 0.5625, 0.8125, 1).
const GRADIENT: [u16; 4] = [0x3500, 0x3880, 0x3A80, 0x3C00];

/// The target's rows over the surface stand-in, in bytes.
const SURFACE_ROW: u64 = 768;

/// The surface's pages: three of 16 KiB, past the plane's 36,864 bytes.
const SURFACE_PAGES: usize = 48 << 10;

/// `MTLResourceStorageModeManaged`.
const MANAGED: u64 = 1 << 4;

/// The edges an instance of the span quads draws.
const EDGES_PER_INSTANCE: u32 = 13;

/// The instances of a cell's span quads, unless only those the outline needs.
const INSTANCES: u32 = 4;

/// How a case sums its coverage.
#[derive(Clone, Copy)]
enum Coverage {
    /// Instanced rectangles (`chain_edges`), four overlapping where `overlap`, else one.
    Rectangles { overlap: bool },
    /// Span quads (`chain_spans`, `chain_span_edges`).
    Spans(Spans),
}

/// What the span quads switch on.
#[derive(Clone, Copy)]
struct Spans {
    /// Each quad reaches half a pixel past its edge (else none).
    outset: bool,
    /// Indexed triangle strips that restart after each quad (else a triangle list).
    restart: bool,
    /// The disc reaches the cell's left edge (else it is a pixel narrower on each side).
    edge: bool,
    /// Every instance of [`INSTANCES`] is drawn (else only those the outline needs).
    collapsed: bool,
}

/// The span quads with everything on.
const SPANS: Spans = Spans {
    outset: true,
    restart: true,
    edge: true,
    collapsed: true,
};

/// The chain's last pass.
#[derive(Clone, Copy)]
enum Colour {
    /// The colour times the coverage, reading only the coverage, discarding where it is 0 if
    /// `discard` (`chain_color_discarding`, else `chain_color`).
    Fill { discard: bool },
    /// A gradient's fill, reading the colour and the coverage, discarding where the coverage is 0
    /// and sampling the noise ([`Sample`]), its colour the colour plus [`NOISE`]
    /// ([`GRADIENT`]).
    Gradient(Sample),
}

/// The fill that discards.
const FILL: Colour = Colour::Fill { discard: true };

/// Where and how a gradient's fill samples the noise.
#[derive(Clone, Copy)]
enum Sample {
    /// At an implicit LOD, between its two discards (`chain_gradient`).
    BetweenDiscards,
    /// At LOD 0, between its two discards (`chain_gradient_lod0`).
    Lod0,
    /// At an implicit LOD, before its first discard (`chain_gradient_hoisted`).
    Hoisted,
}

/// The depth test of every draw.
#[derive(Clone, Copy)]
enum Tests {
    /// The default state (always passing, no writes), the depth cleared to 1.
    Default,
    /// `GreaterEqual` without writes, the depth cleared to 0, as the consumer's pass draws.
    GreaterEqual,
}

/// One case: what it switches on.
struct Case {
    name: &'static str,
    coverage: Coverage,
    /// The coverage attachment is memoryless (else private, stored `DontCare`).
    memoryless: bool,
    /// The colour pass.
    colour: Colour,
    /// The depth test.
    tests: Tests,
    /// The target lies over the surface stand-in (else it is private).
    surface: bool,
}

const CASES: [Case; 21] = [
    Case {
        name: "rectangles, memoryless, overlapping, discarding",
        coverage: Coverage::Rectangles { overlap: true },
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "rectangles, stored coverage, overlapping, discarding",
        coverage: Coverage::Rectangles { overlap: true },
        memoryless: false,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "rectangles, memoryless, one rectangle, discarding",
        coverage: Coverage::Rectangles { overlap: false },
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "rectangles, memoryless, overlapping, no discard",
        coverage: Coverage::Rectangles { overlap: true },
        memoryless: true,
        colour: Colour::Fill { discard: false },
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "rectangles, stored coverage, one rectangle, no discard",
        coverage: Coverage::Rectangles { overlap: false },
        memoryless: false,
        colour: Colour::Fill { discard: false },
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, everything on",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, no outset",
        coverage: Coverage::Spans(Spans {
            outset: false,
            ..SPANS
        }),
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, a triangle list",
        coverage: Coverage::Spans(Spans {
            restart: false,
            ..SPANS
        }),
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, inset from the left edge",
        coverage: Coverage::Spans(Spans {
            edge: false,
            ..SPANS
        }),
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, only the instances needed",
        coverage: Coverage::Spans(Spans {
            collapsed: false,
            ..SPANS
        }),
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, stored coverage",
        coverage: Coverage::Spans(SPANS),
        memoryless: false,
        colour: FILL,
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, no discard",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: Colour::Fill { discard: false },
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "rectangles, memoryless, overlapping, discarding, over a surface",
        coverage: Coverage::Rectangles { overlap: true },
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: true,
    },
    Case {
        name: "spans, everything on, over a surface",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: FILL,
        tests: Tests::Default,
        surface: true,
    },
    Case {
        name: "spans, no discard, over a surface",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: Colour::Fill { discard: false },
        tests: Tests::Default,
        surface: true,
    },
    Case {
        name: "spans, a gradient sampling between its discards",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: Colour::Gradient(Sample::BetweenDiscards),
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, a gradient sampling between its discards, depth GreaterEqual",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: Colour::Gradient(Sample::BetweenDiscards),
        tests: Tests::GreaterEqual,
        surface: false,
    },
    Case {
        name: "spans, a gradient sampling at LOD 0, depth GreaterEqual",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: Colour::Gradient(Sample::Lod0),
        tests: Tests::GreaterEqual,
        surface: false,
    },
    Case {
        name: "spans, a gradient sampling before its discards",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: Colour::Gradient(Sample::Hoisted),
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "rectangles, overlapping, a gradient sampling between its discards",
        coverage: Coverage::Rectangles { overlap: true },
        memoryless: true,
        colour: Colour::Gradient(Sample::BetweenDiscards),
        tests: Tests::Default,
        surface: false,
    },
    Case {
        name: "spans, a gradient sampling between its discards, depth GreaterEqual, over a surface",
        coverage: Coverage::Spans(SPANS),
        memoryless: true,
        colour: Colour::Gradient(Sample::BetweenDiscards),
        tests: Tests::GreaterEqual,
        surface: true,
    },
];

/// The shape's rectangles of `cell`, with their weights (the module docs).
fn rectangles(cell: [u32; 4], overlap: bool) -> Vec<([f32; 4], f32)> {
    let [x0, y0, x1, y1] = cell.map(|v| v as f32);
    let (left, right) = (x0 + 1.0, x1 - 1.0);
    let whole = [left, y0, right, y1];
    match overlap {
        false => vec![(whole, 1.0)],
        true => vec![
            ([left, y0, right, y0 + 10.0], 1.0),
            ([left, y0 + 10.0, right, y1], 1.0),
            (whole, 0.5),
            (whole, -0.5),
        ],
    }
}

/// The disc of `cell` (inset by a pixel unless `edge`): each row's first pixel and the pixel past
/// its last, or `None` for an empty row. A pixel is in it where its centre lies within the
/// radius, half the cell's shorter side, of the cell's centre.
fn disc(cell: [u32; 4], edge: bool) -> Vec<Option<(u32, u32)>> {
    let [x0, y0, x1, y1] = cell;
    let (w, h) = (f64::from(x1 - x0), f64::from(y1 - y0));
    let (cx, cy) = (f64::from(x0) + w / 2.0, f64::from(y0) + h / 2.0);
    let radius = w.min(h) / 2.0 - if edge { 0.0 } else { 1.0 };
    (y0..y1)
        .map(|y| {
            let d = f64::from(y) + 0.5 - cy;
            if d.abs() >= radius {
                return None;
            }
            let half = (radius * radius - d * d).sqrt();
            let (left, right) = ((cx - half - 0.5).ceil(), (cx + half - 0.5).floor() + 1.0);
            (left < right).then_some((left as u32, right as u32))
        })
        .collect()
}

/// The outline's vertical edges of `cell`'s disc, `(x, top, bottom, weight)`: the left side's runs
/// of rows starting at one pixel, of weight 1, then the right side's runs ending at one, of
/// weight -1, each from the top.
fn edges(cell: [u32; 4], edge: bool) -> Vec<[f32; 4]> {
    let rows = disc(cell, edge);
    let mut out = Vec::new();
    for (side, weight) in [(0, 1.0f32), (1, -1.0)] {
        // The run's x, first row and the row past its last.
        let mut run: Option<(u32, u32, u32)> = None;
        for (r, row) in rows.iter().enumerate() {
            let y = cell[1] + r as u32;
            let x = row.map(|(left, right)| if side == 0 { left } else { right });
            match (run, x) {
                (Some((rx, top, _)), Some(x)) if rx == x => run = Some((rx, top, y + 1)),
                _ => {
                    if let Some((rx, top, bottom)) = run {
                        out.push([rx as f32, top as f32, bottom as f32, weight]);
                    }
                    run = x.map(|x| (x, y, y + 1));
                }
            }
        }
        if let Some((rx, top, bottom)) = run {
            out.push([rx as f32, top as f32, bottom as f32, weight]);
        }
    }
    out
}

/// The span quads' indices of an instance: a strip of four a quad, restarted after each (64), or
/// two triangles a quad (78).
fn indices(restart: bool) -> Vec<u16> {
    let quads = EDGES_PER_INSTANCE as u16;
    let mut out: Vec<u16> = (0..quads)
        .flat_map(|k| {
            let v = 4 * k;
            match restart {
                true => vec![v, v + 1, v + 3, v + 2, 0xFFFF],
                false => vec![v, v + 1, v + 3, v + 1, v + 2, v + 3],
            }
        })
        .collect();
    if restart {
        out.pop();
    }
    out
}

/// Whether pixel `(x, y)` is in a cell's shape under `coverage`.
fn in_shape(coverage: Coverage, x: u32, y: u32) -> bool {
    CELLS.iter().any(|&cell| {
        let [x0, y0, x1, y1] = cell;
        let inside = x >= x0 && x < x1 && y >= y0 && y < y1;
        inside
            && match coverage {
                Coverage::Rectangles { .. } => x > x0 && x + 1 < x1,
                Coverage::Spans(spans) => disc(cell, spans.edge)[(y - y0) as usize]
                    .is_some_and(|(left, right)| x >= left && x < right),
            }
    })
}

/// A pipeline of `vertex` and `fragment`, drawing the pass's attachments, blending the coverage
/// additively where `additive`.
fn pipeline(
    lane: &Lane,
    chain: &Library,
    (vertex, fragment): (&str, &str),
    additive: bool,
) -> RenderPipelineState {
    let function = |name| {
        StageSource::Function(
            chain
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    let mut coverage = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
    if additive {
        coverage.blending_enabled = true;
        coverage.source_rgb_blend_factor = BlendFactor::One;
        coverage.destination_rgb_blend_factor = BlendFactor::One;
        coverage.source_alpha_blend_factor = BlendFactor::One;
        coverage.destination_alpha_blend_factor = BlendFactor::One;
    }
    descriptor.color_attachments[1] = Some(coverage);
    descriptor.color_attachments[2] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    descriptor.depth_format = Some(PixelFormat::Depth32FloatStencil8);
    descriptor.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    let device = lane.device.name();
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap_or_else(|error| panic!("{fragment} on {device}: {error}"))
}

/// A `W` by `H` render target of `pf`, private or of `options`.
fn target(lane: &Lane, pf: PixelFormat, options: ResourceOptions) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, W, H, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options;
    lane.device.new_texture(&descriptor).unwrap()
}

/// The bytes of `values`, little-endian.
fn bytes_of<const N: usize>(values: impl IntoIterator<Item = [u8; N]>) -> Vec<u8> {
    values.into_iter().flatten().collect()
}

/// The noise texture: [`NOISE_SIZE`] square, every texel [`NOISE`].
fn noise(lane: &Lane) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, NOISE_SIZE, NOISE_SIZE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = lane.device.new_texture(&descriptor).unwrap();
    let texels = bytes_of((0..NOISE_SIZE * NOISE_SIZE).flat_map(|_| NOISE.map(u16::to_le_bytes)));
    let region = Region::new_2d(0, 0, NOISE_SIZE, NOISE_SIZE);
    texture
        .replace_region(region, 0, 0, &texels, 8 * NOISE_SIZE, 0)
        .unwrap();
    texture
}

/// Draw the span quads of `cell` under `spans` (the module docs), holding their buffers in `held`.
fn draw_spans(
    lane: &Lane,
    encoder: &mut RenderCommandEncoder,
    held: &mut Vec<Buffer>,
    cell: [u32; 4],
    spans: Spans,
) -> Result<(), Error> {
    let edges = edges(cell, spans.edge);
    let count = edges.len() as u32;
    assert!(count <= INSTANCES * EDGES_PER_INSTANCE, "{count} edges");
    let instances = match spans.collapsed {
        true => INSTANCES,
        false => count.div_ceil(EDGES_PER_INSTANCE),
    };
    let outset: f32 = if spans.outset { 0.5 } else { 0.0 };
    // The frame: size, right bound, outset, count, padded to the struct's 24 bytes.
    let mut frame = bytes_of([W as f32, H as f32, cell[2] as f32, outset].map(f32::to_le_bytes));
    frame.extend(count.to_le_bytes());
    frame.resize(24, 0);
    let indices = indices(spans.restart);
    let edge_buffer = buffer_with(
        &lane.device,
        &bytes_of(edges.iter().flat_map(|e| e.map(f32::to_le_bytes))),
    )
    .unwrap();
    let index_buffer = buffer_with(
        &lane.device,
        &bytes_of(indices.iter().map(|i| i.to_le_bytes())),
    )
    .unwrap();
    encoder.set_vertex_buffer(Some(&edge_buffer), 0, 0)?;
    encoder.set_vertex_bytes(&frame, 1)?;
    let primitive = match spans.restart {
        true => PrimitiveType::TriangleStrip,
        false => PrimitiveType::Triangle,
    };
    encoder.draw_indexed_primitives(
        primitive,
        indices.len() as u32,
        IndexType::UInt16,
        &index_buffer,
        0,
        instances,
        0,
        0,
    )?;
    held.push(edge_buffer);
    held.push(index_buffer);
    Ok(())
}

/// The target's texels, row by row, after one pass of `case`.
fn drawn(lane: &Lane, case: &Case, pipelines: &[RenderPipelineState; 3]) -> Vec<u8> {
    let (color, _surface) = match case.surface {
        false => (
            target(lane, PixelFormat::RGBA16Float, ResourceOptions::private()),
            None,
        ),
        true => {
            // A no-copy buffer over the whole surface's pages, as the consumer makes it.
            let mapping = Mapping::anonymous(SURFACE_PAGES);
            let pointer = mapping.pointer();
            let release: Deallocator = Box::new(move |_, _| drop(mapping));
            let options = ResourceOptions::from_raw(MANAGED).unwrap();
            // SAFETY: the mapping stays mapped until the deallocator, which owns it, runs.
            let surface =
                unsafe { no_copy(&lane.device, pointer, SURFACE_PAGES, options, Some(release)) }
                    .unwrap_or_else(|error| panic!("{}: the surface's buffer: {error}", case.name));
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, W, H, false);
            descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            descriptor.resource_options = surface.resource_options();
            let texture = surface
                .new_texture_with_iosurface_plane(
                    &descriptor,
                    IoSurfacePlane::new(0, SURFACE_ROW, false),
                )
                .unwrap_or_else(|error| panic!("{}: the surface's target: {error}", case.name));
            (texture, Some(surface))
        }
    };
    let coverage_options = match case.memoryless {
        true => memoryless_options(),
        false => ResourceOptions::private(),
    };
    let coverage = target(lane, PixelFormat::RG16Float, coverage_options);
    let third = target(lane, PixelFormat::RGBA16Float, memoryless_options());
    let depth = target(
        lane,
        PixelFormat::Depth32FloatStencil8,
        memoryless_options(),
    );
    let mut pass = RenderPassDescriptor::new();
    for (index, (texture, load, store)) in [
        (&color, LoadAction::Clear, StoreAction::Store),
        (&coverage, LoadAction::Clear, StoreAction::DontCare),
        (&third, LoadAction::DontCare, StoreAction::DontCare),
    ]
    .into_iter()
    .enumerate()
    {
        let attachment = &mut pass.color_attachments[index];
        attachment.texture = Some(texture.clone());
        attachment.load_action = load;
        attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        attachment.store_action = store;
    }
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = match case.tests {
        Tests::Default => 1.0,
        Tests::GreaterEqual => 0.0,
    };
    pass.depth_attachment.store_action = StoreAction::DontCare;
    pass.stencil_attachment.texture = Some(depth.clone());
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.store_action = StoreAction::DontCare;
    let size = bytes_of([W as f32, H as f32].map(f32::to_le_bytes));
    let colour = bytes_of(COLOR.map(u16::to_le_bytes));
    let noise = noise(lane);
    let tests = match case.tests {
        Tests::Default => None,
        Tests::GreaterEqual => {
            let mut descriptor = DepthStencilDescriptor::default();
            descriptor.depth_compare_function = CompareFunction::GreaterEqual;
            Some(lane.device.new_depth_stencil_state(&descriptor).unwrap())
        }
    };
    let name = case.name;
    let device = lane.device.name().to_owned();
    lane.run(&pass, |lane, encoder, held: &mut Vec<Buffer>| {
        if let Some(tests) = &tests {
            encoder.set_depth_stencil_state(tests).unwrap();
        }
        encoder.set_fragment_texture(Some(&noise), 0).unwrap();
        for (step, pipeline) in pipelines.iter().enumerate() {
            encoder.set_render_pipeline_state(pipeline).unwrap();
            for cell in ORDER[step].map(|i| CELLS[i]) {
                let [x0, y0, x1, y1] = cell;
                encoder
                    .set_scissor_rect(ScissorRect::new(
                        x0.into(),
                        y0.into(),
                        (x1 - x0).into(),
                        (y1 - y0).into(),
                    ))
                    .unwrap();
                encoder.set_fragment_bytes(&colour, 0).unwrap();
                let rects = match (step, case.coverage) {
                    (0, Coverage::Spans(spans)) => {
                        draw_spans(lane, encoder, held, cell, spans)
                            .unwrap_or_else(|error| panic!("{name}: a draw on {device}: {error}"));
                        continue;
                    }
                    (0, Coverage::Rectangles { overlap }) => rectangles(cell, overlap),
                    _ => vec![(cell.map(|v| v as f32), 0.0)],
                };
                let corners = buffer_with(
                    &lane.device,
                    &bytes_of(rects.iter().flat_map(|(r, _)| r.map(f32::to_le_bytes))),
                )
                .unwrap();
                let weights = buffer_with(
                    &lane.device,
                    &bytes_of(rects.iter().map(|(_, w)| w.to_le_bytes())),
                )
                .unwrap();
                encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                encoder.set_vertex_buffer(Some(&weights), 0, 1).unwrap();
                encoder.set_vertex_bytes(&size, 2).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::TriangleStrip, 0, 4, rects.len() as u32, 0)
                    .unwrap_or_else(|error| panic!("{name}: a draw on {device}: {error}"));
                held.push(corners);
                held.push(weights);
            }
        }
    })
    .unwrap_or_else(|error| panic!("{name}: the pass on {}: {error}", lane.device.name()));
    lane.read(&color, (0, 0), 8, BlitOption::None)
}

/// The pixels whose texel differs from Metal's under `coverage`, `color` in the shape, as a count
/// and the 8x4 blocks they fall in.
fn lost(coverage: Coverage, color: [u16; 4], texels: &[u8]) -> (usize, Vec<(u32, u32)>) {
    let mut count = 0;
    let mut blocks = Vec::new();
    for (index, texel) in texels.as_chunks::<8>().0.iter().enumerate() {
        let (x, y) = ((index as u64 % W) as u32, (index as u64 / W) as u32);
        let want = match in_shape(coverage, x, y) {
            true => bytes_of(color.map(u16::to_le_bytes)),
            false => vec![0; 8],
        };
        if texel != want.as_slice() {
            count += 1;
            let block = (x / 8 * 8, y / 4 * 4);
            if !blocks.contains(&block) {
                blocks.push(block);
            }
        }
    }
    (count, blocks)
}

/// Metal's answer in every case (the module docs), on every device.
#[test]
fn a_coverage_chain_through_fetch_keeps_every_block() {
    let (messages, _cache, lane) = lane();
    let chain = library(&lane, "dsx_coverage_chain");
    let device = lane.device.name().to_owned();
    let mut failed = Vec::new();
    for case in &CASES {
        let (colour, color) = match case.colour {
            Colour::Fill { discard: true } => ("chain_color_discarding", COLOR),
            Colour::Fill { discard: false } => ("chain_color", COLOR),
            Colour::Gradient(Sample::BetweenDiscards) => ("chain_gradient", GRADIENT),
            Colour::Gradient(Sample::Lod0) => ("chain_gradient_lod0", GRADIENT),
            Colour::Gradient(Sample::Hoisted) => ("chain_gradient_hoisted", GRADIENT),
        };
        let sum = match case.coverage {
            Coverage::Rectangles { .. } => ("chain_rects", "chain_edges"),
            Coverage::Spans(_) => ("chain_spans", "chain_span_edges"),
        };
        let pipelines = [
            pipeline(&lane, &chain, sum, true),
            pipeline(&lane, &chain, ("chain_rects", "chain_coverage"), false),
            pipeline(&lane, &chain, ("chain_rects", colour), false),
        ];
        let before = lane.device.work_counts();
        let texels = drawn(&lane, case, &pipelines);
        let after = lane.device.work_counts();
        let (count, blocks) = lost(case.coverage, color, &texels);
        let ran = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
        eprintln!(
            "coverage chain, {}, on {device} through {}: {count} texels from Metal's, 8x4 blocks \
             (x, y) {blocks:?}; {} instances begun, split for interlocked writes {}, for a \
             barrier {}, for a memoryless write {}; draws dropped {}",
            case.name,
            form(&lane),
            ran(|c| c.render_passes_begun_instances),
            ran(|c| c.instance_splits_interlocked_writes),
            ran(|c| c.instance_splits_barrier),
            ran(|c| c.instance_splits_memoryless_write),
            ran(|c| c.draws_dropped_still_making + c.draws_dropped_unmade),
        );
        if count != 0 {
            failed.push(case.name);
        }
    }
    assert!(failed.is_empty(), "on {device}: {failed:?}");
    quiet(lane, &messages);
}

/// The span quads' geometry as the module docs describe it: each disc reaches its cell's left
/// edge in its widest rows (a pixel short of it when inset), its edges sum to 1 in the shape and 0
/// elsewhere in the cell and fit the instances, and the strips restart after each quad.
#[test]
fn the_span_quads_cover_each_disc_once() {
    for cell in CELLS {
        for edge in [true, false] {
            let rows = disc(cell, edge);
            let reach = rows.iter().flatten().map(|(left, _)| *left).min();
            assert_eq!(reach, Some(cell[0] + u32::from(!edge)), "{cell:?}");
            let edges = edges(cell, edge);
            assert!(
                edges.len() as u32 <= INSTANCES * EDGES_PER_INSTANCE,
                "{cell:?}"
            );
            for y in cell[1]..cell[3] {
                for x in cell[0]..cell[2] {
                    let sum: f32 = edges
                        .iter()
                        .filter(|e| y as f32 >= e[1] && (y as f32) < e[2] && x as f32 >= e[0])
                        .map(|e| e[3])
                        .sum();
                    let inside = rows[(y - cell[1]) as usize]
                        .is_some_and(|(left, right)| x >= left && x < right);
                    assert_eq!(sum, f32::from(u8::from(inside)), "{cell:?} at ({x}, {y})");
                }
            }
        }
    }
    let strips = indices(true);
    assert_eq!(strips.len(), 64);
    assert_eq!(strips.iter().filter(|&&i| i == 0xFFFF).count(), 12);
    assert_eq!(indices(false).len(), 78);
}
