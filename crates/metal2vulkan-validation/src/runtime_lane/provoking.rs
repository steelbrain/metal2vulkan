//! Device tests of Apple's provoking vertex mode (`setProvokingVertexMode:`) on the device the cfg
//! selects, under the validation layer, with the device's last provoking vertex
//! (`VK_EXT_provoking_vertex`'s `provokingVertexLast`) and with it withheld, each also with
//! `fragmentShaderBarycentric` withheld, so a generated geometry stage writes the weights (on a
//! device with `geometryShader`; skipped elsewhere). The owned fixture
//! `rt6_provoking` places vertex `v` at the clip-space position buffer 0 holds at
//! `min(v, 65535)`, and its fragment functions write the barycentric weights, a smooth value of the
//! vertex ID, the flat vertex ID and the primitive ID, or some of them.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1, every fragment's weights solved for
//! its primitive's vertices): at mode 1 a primitive's flat inputs read its last vertex in the order
//! [`metal_primitives`] states (a line list's `2j + 1`, a line strip's `j + 1`, a triangle list's
//! `3j + 2`, a strip's `j + 2`, a private type's list primitive's last), and its weights follow
//! that order, mode 0's but for a triangle strip's odd triangle, `(j + 1, j, j + 2)` where mode 0
//! gives `(j, j + 2, j + 1)`; indexed draws (a base vertex, the all-ones index and the encoder's
//! restart index, each piece's parity its own), indirect draws and indexed indirect draws follow
//! the mode alike, and so do direct draws from a vertex start past `i32::MAX` (vertex IDs
//! wrapping past `u32::MAX`), which the device without the last vertex draws in the pulled list
//! form (`RuntimeFact::PrivateList`): an indexed list's base vertex cannot carry the start.
//!
//! So each case draws once at mode 0 and once at mode 1, and checks every pixel the mode-1 draw
//! covers against Metal's mode-1 primitive its primitive ID names (or, without one, some primitive
//! of the draw): the flat vertex ID is the primitive's last vertex, and the smooth value is the
//! weights' sum over its vertices in Metal's order. The two draws cover the same pixels with the
//! same primitive IDs (but for lines the device without the last vertex draws from their last
//! vertex, which it announces), and a pipeline that reads neither flat inputs nor weights draws
//! alike, bit for bit, at both modes. The mode-0 draw fails the mode-1 checks at some pixels: the
//! checks tell the modes apart. On the device without the last vertex, a line's direction, which
//! the runtime announces it draws otherwise, is left unchecked; the weights of a triangle that
//! reads flat inputs too, which the rotated-weights variant takes back to Metal's order but for
//! rounding, are checked. The work counts show which path ran.

use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage, TriangleFillMode,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, RuntimeFact, StageSource,
};
use metal2vulkan::vk;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, DrawPrimitiveType,
    Error, ExecutedPipeline, IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions,
    Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    StoreAction, TextureDescriptor,
};
use std::cell::Cell;

/// The kinds the lane's tests and `provoking_mesh`'s take: draws under `setProvokingVertexMode:`
/// 1 on a device without its last provoking vertex, of lines drawn from their last vertex to
/// their first, and of what no form draws as Metal does (a listed triangle's weights, a mode
/// past 1).
pub(super) const PROVOKING: [Degradation; 2] = [
    Degradation::ProvokingVertexLines,
    Degradation::ProvokingVertexUnmatched,
];

/// The target's side, in pixels.
const SIDE: u64 = 32;

/// The vertices the positions scatter over the target.
const PLACED: usize = 4096;

/// The vertices the short walks of indices name.
const SCATTERED: u32 = 64;

/// The attachments' formats: the weights, the smooth value, the flat vertex ID and primitive ID.
const FORMATS: [PixelFormat; 3] = [
    PixelFormat::RGBA32Float,
    PixelFormat::RGBA32Float,
    PixelFormat::RGBA32Uint,
];

/// The types the cases draw: the public line and triangle types and the private ones.
const TYPES: [DrawPrimitiveType; 8] = [
    DrawPrimitiveType::Public(PrimitiveType::Line),
    DrawPrimitiveType::Public(PrimitiveType::LineStrip),
    DrawPrimitiveType::Public(PrimitiveType::Triangle),
    DrawPrimitiveType::Public(PrimitiveType::TriangleStrip),
    DrawPrimitiveType::LineLoop,
    DrawPrimitiveType::TriangleFan,
    DrawPrimitiveType::Quads,
    DrawPrimitiveType::QuadStrip,
];

/// Whether `primitive` draws lines.
fn lines(primitive: DrawPrimitiveType) -> bool {
    matches!(
        primitive,
        DrawPrimitiveType::Public(PrimitiveType::Line | PrimitiveType::LineStrip)
            | DrawPrimitiveType::LineLoop
    )
}

/// Whether `primitive` counts one `[[primitive_id]]` for two triangles (refused under a fragment
/// function that reads it).
fn quads(primitive: DrawPrimitiveType) -> bool {
    matches!(
        primitive,
        DrawPrimitiveType::Quads | DrawPrimitiveType::QuadStrip
    )
}

/// The primitives Metal draws of `values` (each position's index, as Metal reads it) as
/// `primitive`, restarting a strip at `restart`, each in Metal's order at mode 1 where `last`, else
/// at mode 0 (the probes' model, written apart from the runtime's): the flat vertex is the last
/// one at mode 1, the first at mode 0.
fn metal_primitives(
    primitive: DrawPrimitiveType,
    values: &[u32],
    restart: Option<u32>,
    last: bool,
) -> Vec<Vec<u32>> {
    let pieces = |values: &[u32]| -> Vec<Vec<u32>> {
        match restart {
            Some(r) => values.split(|&v| v == r).map(<[u32]>::to_vec).collect(),
            None => vec![values.to_vec()],
        }
    };
    let mut out = Vec::new();
    match primitive {
        DrawPrimitiveType::Public(PrimitiveType::Line) => {
            out.extend(values.as_chunks::<2>().0.iter().map(|l| l.to_vec()));
        }
        DrawPrimitiveType::Public(PrimitiveType::LineStrip) => {
            for piece in pieces(values) {
                out.extend(piece.windows(2).map(<[u32]>::to_vec));
            }
        }
        DrawPrimitiveType::Public(PrimitiveType::Triangle) => {
            out.extend(values.as_chunks::<3>().0.iter().map(|t| t.to_vec()));
        }
        DrawPrimitiveType::Public(PrimitiveType::TriangleStrip) => {
            for piece in pieces(values) {
                for (j, t) in piece.windows(3).enumerate() {
                    out.push(match (j % 2, last) {
                        (0, _) => vec![t[0], t[1], t[2]],
                        (_, true) => vec![t[1], t[0], t[2]],
                        (_, false) => vec![t[0], t[2], t[1]],
                    });
                }
            }
        }
        DrawPrimitiveType::LineLoop => {
            if values.len() >= 2 {
                out.extend(values.windows(2).map(<[u32]>::to_vec));
                out.push(vec![values[values.len() - 1], values[0]]);
            }
        }
        DrawPrimitiveType::TriangleFan => {
            for j in 0..values.len().saturating_sub(2) {
                out.push(vec![values[0], values[j + 1], values[j + 2]]);
            }
        }
        DrawPrimitiveType::Quads => {
            for q in values.as_chunks::<4>().0 {
                out.push(vec![q[0], q[1], q[3]]);
                out.push(vec![q[1], q[2], q[3]]);
            }
        }
        DrawPrimitiveType::QuadStrip => {
            for piece in pieces(values) {
                for k in 0..piece.len().saturating_sub(2) / 2 {
                    let p = &piece[2 * k..2 * k + 4];
                    out.push(vec![p[0], p[1], p[3]]);
                    out.push(vec![p[2], p[0], p[3]]);
                }
            }
        }
        _ => unreachable!("a type the cases do not draw"),
    }
    out
}

/// The fixture's 65536 positions, `(x, y, z, w)` each: vertices 0 to [`PLACED`] less 1 and 65535
/// at points of a fixed pseudo-random walk over the target and a little past it, at depths and `w`
/// that make the weights perspective-correct; every other vertex off the target.
fn positions() -> Vec<u8> {
    let mut all = vec![[5.0f32, 5.0, 0.0, 1.0]; 65536];
    let mut state = 0x9e37_79b9_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for vertex in (0..PLACED).chain([65535]) {
        let unit = |n: u32| (n % 10_000) as f32 / 10_000.0;
        let (x, y) = (unit(next()) * 2.4 - 1.2, unit(next()) * 2.4 - 1.2);
        let w = 1.0 + 2.0 * unit(next());
        all[vertex] = [x * w, y * w, 0.5 * w * unit(next()), w];
    }
    all.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

/// `values` as indices of `index_type`.
fn bytes_of(index_type: IndexType, values: &[u32]) -> Vec<u8> {
    match index_type {
        IndexType::UInt16 => values
            .iter()
            .map(|&i| u16::try_from(i).unwrap())
            .flat_map(u16::to_le_bytes)
            .collect(),
        _ => values.iter().flat_map(|i| i.to_le_bytes()).collect(),
    }
}

/// How a case draws.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    /// `draw_primitives`.
    Direct,
    /// `draw_primitives_indirect`.
    Indirect,
    /// `draw_indexed_primitives`, of indices of this type.
    Indexed(IndexType),
    /// `draw_indexed_primitives_indirect`, of indices of this type.
    IndexedIndirect(IndexType),
}

impl Form {
    fn index_type(self) -> Option<IndexType> {
        match self {
            Form::Direct | Form::Indirect => None,
            Form::Indexed(t) | Form::IndexedIndirect(t) => Some(t),
        }
    }
}

/// A case: one draw.
#[derive(Clone, Debug)]
struct Case {
    primitive: DrawPrimitiveType,
    form: Form,
    /// The positions it draws.
    count: u32,
    /// A draw without indices: its vertex start; with: its first index.
    start: u32,
    base_vertex: i32,
    /// The index buffer's indices (empty for a draw without indices).
    indices: Vec<u32>,
    /// The encoder's `set_primitive_restart_enabled` index before the draw, if any.
    k: Option<u64>,
}

impl Case {
    fn direct(primitive: DrawPrimitiveType, form: Form, count: u32, start: u32) -> Case {
        Case {
            primitive,
            form,
            count,
            start,
            base_vertex: 0,
            indices: Vec::new(),
            k: None,
        }
    }

    fn indexed(primitive: DrawPrimitiveType, form: Form, indices: Vec<u32>) -> Case {
        Case {
            primitive,
            form,
            count: indices.len() as u32,
            start: 0,
            base_vertex: 2,
            indices,
            k: None,
        }
    }

    /// Each position's index, as Metal reads it (0 past the buffer).
    fn values(&self) -> Vec<u32> {
        match self.form.index_type() {
            None => (0..self.count).collect(),
            Some(_) => (0..self.count)
                .map(|p| {
                    let at = (self.start + p) as usize;
                    self.indices.get(at).copied().unwrap_or(0)
                })
                .collect(),
        }
    }

    /// The index a strip restarts at: `k'` after the encoder's call, else the index type's
    /// all-ones index; none without indices, and none of a type that never restarts.
    fn restart(&self) -> Option<u32> {
        let index_type = self.form.index_type()?;
        let strips = matches!(
            self.primitive,
            DrawPrimitiveType::Public(PrimitiveType::LineStrip | PrimitiveType::TriangleStrip)
                | DrawPrimitiveType::QuadStrip
        );
        if !strips {
            return None;
        }
        Some(match (self.k, index_type) {
            (Some(k), IndexType::UInt16) => (k as u32).min(0xffff),
            (Some(k), _) => k as u32,
            (None, IndexType::UInt16) => 0xffff,
            (None, _) => u32::MAX,
        })
    }

    /// The vertex ID of a position's index: the index plus the base vertex, or a draw without
    /// indices' position plus its vertex start.
    fn vertex(&self, value: u32) -> u32 {
        match self.form.index_type() {
            None => value.wrapping_add(self.start),
            Some(_) => value.wrapping_add(self.base_vertex as u32),
        }
    }

    /// Metal's primitives of the case at mode 1 (`last`) or 0, as vertex IDs.
    fn primitives(&self, last: bool) -> Vec<Vec<u32>> {
        metal_primitives(self.primitive, &self.values(), self.restart(), last)
            .into_iter()
            .map(|p| p.into_iter().map(|v| self.vertex(v)).collect())
            .collect()
    }
}

/// The cases of `primitive`.
fn cases(primitive: DrawPrimitiveType) -> Vec<Case> {
    use IndexType::{UInt16, UInt32};
    let walk =
        |n: u32, step: u32| -> Vec<u32> { (0..n).map(|i| (i * step + 1) % SCATTERED).collect() };
    // A strip's restart, 32-bit: elsewhere a vertex past which no other index runs.
    let strips = matches!(
        primitive,
        DrawPrimitiveType::Public(PrimitiveType::LineStrip | PrimitiveType::TriangleStrip)
            | DrawPrimitiveType::QuadStrip
    );
    let ones = match strips {
        true => u32::MAX,
        false => 0xffff,
    };
    let mut out = vec![
        Case::direct(primitive, Form::Direct, 13, 3),
        Case::direct(primitive, Form::Indirect, 12, 5),
        // From a vertex start past `i32::MAX`, its vertex IDs wrapping to 0 at the fourth: those
        // past 65535 all at vertex 65535's place.
        Case::direct(primitive, Form::Direct, 10, 0xffff_fffd),
    ];
    // The all-ones index, a restart of the strips, and 7 among the indices.
    let mut with = walk(15, 5);
    with[6] = 0xffff;
    with[10] = 7;
    out.push(Case::indexed(
        primitive,
        Form::Indexed(UInt16),
        with.clone(),
    ));
    let mut wide = with.clone();
    wide[6] = ones;
    out.push(Case::indexed(primitive, Form::Indexed(UInt32), wide));
    // After the encoder's call: 7 restarts a strip, and the all-ones index is a vertex.
    let mut k7 = Case::indexed(primitive, Form::Indexed(UInt16), with.clone());
    k7.k = Some(7);
    out.push(k7);
    let mut indirect = Case::indexed(primitive, Form::IndexedIndirect(UInt16), with);
    indirect.start = 1;
    indirect.count = 13;
    out.push(indirect);
    // Many positions across the rewrite pass's runs and groups.
    let long: Vec<u32> = (0..3001u32)
        .map(|i| match i % 41 {
            40 => ones,
            _ => ((i * 11 + 1) % SCATTERED) | 8,
        })
        .collect();
    out.push(Case::indexed(primitive, Form::Indexed(UInt32), long));
    out
}

/// Each attachment's texels, four words each.
type Texels = Vec<Vec<[u32; 4]>>;

/// The lane's device, its queue, the fixture's library and the positions.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
    positions: Buffer,
    /// The geometry stages the pipeline of the last draw carries (`None` where it recorded
    /// none).
    geometry_stages: Cell<Option<usize>>,
    /// Whether the pipeline of the last draw ran the pulled list form
    /// (`RuntimeFact::PrivateList`).
    pulled: Cell<bool>,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("rt6_provoking.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            positions: buffer_with(&device, &positions()).unwrap(),
            geometry_stages: Cell::new(None),
            pulled: Cell::new(false),
            device,
        }
    }

    /// The pipeline of `fragment` and the vertex function that writes what it reads.
    fn pipeline(&self, fragment: Fragment) -> RenderPipelineState {
        self.pipeline_of(fragment, false)
    }

    /// [`pipeline`](Self::pipeline)'s, that indirect command buffers may hold where
    /// `executable`.
    fn pipeline_of(&self, fragment: Fragment, executable: bool) -> RenderPipelineState {
        let function = |name| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let vertex = match fragment.flat {
            true => "pv_vertex",
            false => "pv_vertex_smooth",
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
        descriptor.fragment = Some(function(fragment.name));
        descriptor.support_indirect_command_buffers = executable;
        for (i, format) in FORMATS.iter().enumerate() {
            descriptor.color_attachments[i] = Some(ColorAttachmentDescriptor::new(*format));
        }
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// `case` drawn by `pipeline` at provoking vertex mode `mode` over cleared targets: every
    /// attachment's texels.
    fn draw(
        &self,
        pipeline: &RenderPipelineState,
        case: &Case,
        mode: u64,
    ) -> Result<Texels, Error> {
        let index_type = case.form.index_type().unwrap_or(IndexType::UInt32);
        let size = match index_type {
            IndexType::UInt16 => 2,
            _ => 4,
        };
        let index_bytes: Vec<u8> = bytes_of(index_type, &case.indices)
            .into_iter()
            .chain([0; 4])
            .collect();
        let index_buffer = buffer_with(&self.device, &index_bytes).unwrap();
        let words: Vec<u32> = match case.form {
            Form::Indirect => vec![case.count, 1, case.start, 0],
            _ => vec![case.count, 1, case.start, case.base_vertex as u32, 0],
        };
        let arguments: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let indirect = buffer_with(&self.device, &arguments).unwrap();
        self.render(case, |encoder| {
            encoder.set_render_pipeline_state(pipeline)?;
            encoder.set_vertex_buffer(Some(&self.positions), 0, 0)?;
            encoder.set_provoking_vertex_mode(mode)?;
            if let Some(k) = case.k {
                encoder.set_primitive_restart_enabled(true, k);
            }
            match case.form {
                Form::Direct => {
                    encoder.draw_primitives(case.primitive, case.start, case.count, 1, 0)
                }
                Form::Indirect => encoder.draw_primitives_indirect(case.primitive, &indirect, 0),
                Form::Indexed(t) => encoder.draw_indexed_primitives(
                    case.primitive,
                    case.count,
                    t,
                    &index_buffer,
                    u64::from(case.start) * size,
                    1,
                    case.base_vertex,
                    0,
                ),
                Form::IndexedIndirect(t) => encoder.draw_indexed_primitives_indirect(
                    case.primitive,
                    t,
                    &index_buffer,
                    0,
                    &indirect,
                    0,
                ),
            }
        })
    }

    /// What `encode` draws (`case` names it in failures) over cleared targets: every
    /// attachment's texels.
    fn render(
        &self,
        case: &Case,
        encode: impl FnOnce(&mut RenderCommandEncoder) -> Result<(), Error>,
    ) -> Result<Texels, Error> {
        let targets: Vec<_> = FORMATS
            .iter()
            .map(|format| {
                let mut descriptor = TextureDescriptor::texture_2d(*format, SIDE, SIDE, false);
                descriptor.usage = TextureUsage::RENDER_TARGET;
                descriptor.resource_options = ResourceOptions::shared();
                self.device.new_texture(&descriptor).unwrap()
            })
            .collect();
        let mut pass = RenderPassDescriptor::new();
        for (i, target) in targets.iter().enumerate() {
            let color = &mut pass.color_attachments[i];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = StoreAction::Store;
        }
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        let drawn = encode(&mut encoder);
        let ended = encoder.end_encoding();
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        drawn?;
        ended?;
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{case:?}: {:?}",
            command_buffer.error()
        );
        let stages = command_buffer
            .executions()
            .iter()
            .find_map(|e| match e.pipeline() {
                ExecutedPipeline::Render(executed) => Some(
                    executed
                        .stages
                        .iter()
                        .filter(|s| s.stage == vk::ShaderStage::GEOMETRY)
                        .count(),
                ),
                _ => None,
            });
        self.geometry_stages.set(stages);
        let pulled = command_buffer
            .executions()
            .iter()
            .any(|e| match e.pipeline() {
                ExecutedPipeline::Render(executed) => executed
                    .dependencies
                    .iter()
                    .any(|dependency| dependency.fact == RuntimeFact::PrivateList),
                _ => false,
            });
        self.pulled.set(pulled);
        Ok(targets
            .iter()
            .map(|target| {
                let mut out = vec![0u8; (SIDE * SIDE * 16) as usize];
                target
                    .get_bytes(
                        &mut out,
                        SIDE * 16,
                        SIDE * SIDE * 16,
                        Region::new_2d(0, 0, SIDE, SIDE),
                        0,
                        0,
                    )
                    .unwrap();
                out.as_chunks::<16>()
                    .0
                    .iter()
                    .map(|t| {
                        let (words, _) = t.as_chunks::<4>();
                        std::array::from_fn(|i| u32::from_le_bytes(words[i]))
                    })
                    .collect()
            })
            .collect())
    }
}

/// A fragment function of the fixture, and what it reads.
#[derive(Clone, Copy, Debug)]
struct Fragment {
    name: &'static str,
    flat: bool,
    weights: bool,
    primitive_id: bool,
}

const FRAGMENTS: [Fragment; 5] = [
    Fragment {
        name: "pv_flat",
        flat: true,
        weights: false,
        primitive_id: true,
    },
    Fragment {
        name: "pv_both",
        flat: true,
        weights: true,
        primitive_id: true,
    },
    Fragment {
        name: "pv_weights",
        flat: false,
        weights: true,
        primitive_id: true,
    },
    Fragment {
        name: "pv_smooth",
        flat: false,
        weights: false,
        primitive_id: true,
    },
    Fragment {
        name: "pv_bare",
        flat: true,
        weights: true,
        primitive_id: false,
    },
];

/// Whether a covered pixel's texels `(weights, smooth, ids)` are those of `primitive` (vertex IDs
/// in Metal's order): its flat vertex ID the primitive's last where `flat`, and its smooth value
/// the weights' sum over the vertices where `weights`.
fn pixel_of(
    (weights, smooth, ids): ([f32; 4], f32, [u32; 4]),
    primitive: &[u32],
    (flat, check_weights): (bool, bool),
) -> bool {
    let flat_ok = !flat || ids[0] == *primitive.last().unwrap();
    let sum: f32 = primitive
        .iter()
        .zip(weights)
        .map(|(&v, w)| v as f32 * w)
        .sum();
    // The interpolation's error scales with the values interpolated, not with their sum: a
    // fragment near a vertex of a small ID beside one of a large one (65537) interpolates the
    // large one's share to a few units of 1e-5 of it (lavapipe: 0.04 of 65537 * 0.01).
    let scale: f32 = primitive.iter().map(|&v| v as f32).sum();
    let weights_ok = !check_weights || (sum - smooth).abs() <= 0.02 + scale * 1e-5;
    flat_ok && weights_ok
}

/// The covered pixels of `texels` that fail Metal's mode-1 checks for `case` under `fragment`:
/// its flat vertex where `flat`, its weights' order where `weights`. With the counts, the first
/// failing pixel: its texels and its primitive's vertices.
fn failing(
    case: &Case,
    fragment: Fragment,
    texels: &Texels,
    checks: (bool, bool),
) -> (usize, usize, Option<String>) {
    let primitives = case.primitives(true);
    let (mut covered, mut failed, mut first) = (0, 0, None);
    let pixels = texels[0].iter().zip(&texels[1]).zip(&texels[2]);
    for ((weights, smooth), &ids) in pixels {
        if ids[3] != 1 {
            continue;
        }
        covered += 1;
        let w = weights.map(f32::from_bits);
        let smooth = f32::from_bits(smooth[0]);
        let got = (w, smooth, ids);
        let ok = match fragment.primitive_id {
            true => primitives
                .get(ids[1] as usize)
                .is_some_and(|p| pixel_of(got, p, checks)),
            false => primitives.iter().any(|p| pixel_of(got, p, checks)),
        };
        if !ok && first.is_none() {
            let primitive = primitives.get(ids[1] as usize);
            first = Some(format!("{got:?} of {primitive:?}"));
        }
        failed += usize::from(!ok);
    }
    (covered, failed, first)
}

/// What one run of every case found.
#[derive(Default, Debug)]
struct Found {
    cases: usize,
    covered: usize,
    /// Pixels of the mode-0 draws that fail the mode-1 checks.
    told_apart: usize,
    /// Pixels a listed line covers otherwise at mode 1 than at mode 0.
    lines_moved: usize,
    /// Cases of listed triangles whose weights the rotated-weights variant takes back.
    rotated_weights: usize,
    /// Cases of public draws from vertex starts past `i32::MAX` drawn in the pulled list form.
    pulled: usize,
}

/// Every case of every type under every fragment function, at modes 0 and 1, on `device`, whose
/// last provoking vertex is `native` and which reads the weights as `weights` says: the checks the
/// module docs state.
fn check_every_case(device: Device, native: bool, weights: Weights, messages: &Messages) -> Found {
    let name = device.name().to_owned();
    let lane = Lane::new(device);
    let mut found = Found::default();
    let before = lane.device.work_counts();
    for fragment in FRAGMENTS {
        let pipeline = lane.pipeline(fragment);
        for primitive in TYPES {
            // Metal's quads count one primitive ID for two triangles, which the runtime refuses
            // under a fragment function that reads it; `pv_bare` reads none.
            if quads(primitive) == fragment.primitive_id {
                continue;
            }
            for case in cases(primitive) {
                let at = |mode| {
                    lane.draw(&pipeline, &case, mode).unwrap_or_else(|e| {
                        panic!("{case:?} {fragment:?} at {mode} on {name}: {e}")
                    })
                };
                let first = at(0);
                let listed_before = lane.device.work_counts().provoking_listed_draws;
                let last = at(1);
                let listed_now = lane.device.work_counts().provoking_listed_draws > listed_before;
                // A public type's direct draw whose last vertex ID passes `i32::MAX`, listed, runs
                // the pulled list form.
                let far = case.form == Form::Direct
                    && matches!(case.primitive, DrawPrimitiveType::Public(_))
                    && u64::from(case.start) + u64::from(case.count) > 1 << 31;
                if far && listed_now {
                    assert!(
                        lane.pulled.get(),
                        "{case:?} {fragment:?} on {name}: not drawn in the pulled list form"
                    );
                    found.pulled += 1;
                }
                if weights == Weights::Withheld && fragment.weights {
                    assert!(
                        lane.geometry_stages.get().is_none_or(|stages| stages > 0),
                        "{case:?} {fragment:?} on {name}: no geometry stage with the weights \
                         withheld"
                    );
                }
                found.cases += 1;
                let reads = fragment.flat || fragment.weights;
                if !reads {
                    // Nothing reads the provoking vertex: the draw is the same at both modes.
                    assert!(
                        first == last,
                        "{case:?} {fragment:?} on {name}: differs at mode 1"
                    );
                    continue;
                }
                // The listed form of a device without the last vertex draws a line from its last
                // vertex, announced, its weights left unchecked; a triangle, of a pipeline that
                // reads flat inputs too, from its last vertex, its weights taken back to Metal's
                // order by the rotated-weights variant (but for rounding, announced).
                let listed_line = !native && fragment.flat && lines(primitive);
                let checks = (fragment.flat, fragment.weights && !listed_line);
                if !native && fragment.flat && fragment.weights && !listed_line {
                    found.rotated_weights += 1;
                }
                let (covered, failed, example) = failing(&case, fragment, &last, checks);
                assert_eq!(
                    failed, 0,
                    "{case:?} {fragment:?} on {name} (native {native}): {failed} of {covered} \
                     pixels are not Metal's at mode 1, the first {example:?}"
                );
                found.covered += covered;
                // The same pixels, under the same primitive IDs.
                let mut moved = 0;
                for (a, b) in first[2].iter().zip(&last[2]) {
                    if a[3] != b[3] || (fragment.primitive_id && a[3] == 1 && a[1] != b[1]) {
                        moved += 1;
                    }
                }
                match listed_line {
                    true => found.lines_moved += moved,
                    false => assert_eq!(
                        moved, 0,
                        "{case:?} {fragment:?} on {name}: {moved} pixels covered otherwise"
                    ),
                }
                let mode0 = failing(&case, fragment, &first, checks).1;
                found.told_apart += mode0;
            }
        }
    }
    let after = lane.device.work_counts();
    let natively = after.provoking_last_pipeline_draws - before.provoking_last_pipeline_draws;
    let listed = after.provoking_listed_draws - before.provoking_listed_draws;
    // The listed direct indexed draws take a run of the index gather pass (lists) or of the
    // index rewrite pass (strips), and so do the private types' at either mode (under a
    // last-vertex pipeline in Metal's mode-1 order).
    let gathered = after.gathered_draws - before.gathered_draws;
    match native {
        true => assert!(
            natively > 0 && listed == 0 && gathered > 0,
            "{name}: {natively} {listed} {gathered}"
        ),
        false => assert!(
            natively == 0 && listed > 0 && gathered > 0,
            "{name}: {natively} {listed} {gathered}"
        ),
    }
    let kinds: Vec<Degradation> = messages
        .take_notices()
        .iter()
        .filter_map(|m| m.degradation.as_ref())
        .map(|notice| notice.degradation)
        .collect();
    for kind in PROVOKING {
        assert_eq!(kinds.contains(&kind), !native, "{name}: {kinds:?}");
    }
    eprintln!(
        "{name} (native {native}, weights {weights:?}): {} cases, {} covered pixels at mode 1 \
         all Metal's; {} pixels of the mode-0 draws not; draws {natively} through last-vertex \
         pipelines, {listed} listed, {gathered} drawn from a run's lists; {} pixels of listed lines \
         covered otherwise; {} cases of rotated weights checked; {} far draws pulled",
        found.cases,
        found.covered,
        found.told_apart,
        found.lines_moved,
        found.rotated_weights,
        found.pulled
    );
    // The listed path checks rotated triangles' weights and pulls far public draws' lists; the
    // native one lists none.
    assert_eq!(found.rotated_weights > 0, !native, "{name}");
    assert_eq!(found.pulled > 0, !native, "{name}");
    found
}

/// How the lane's device reads the barycentric weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Weights {
    /// As the device is made (`BaryCoordKHR` where it has `fragmentShaderBarycentric`).
    Device,
    /// With `fragmentShaderBarycentric` withheld: varyings a generated geometry stage writes,
    /// whose primitive's provoking vertex is the stage's last vertex under a last-vertex
    /// pipeline (the Vulkan specification's Flat Shading: "the provoking vertex is determined by
    /// the primitive topology defined by the ... OutputTriangleStrip execution mode").
    Withheld,
}

/// Run [`check_every_case`] on the cfg's device, with its last provoking vertex withheld where
/// `withheld`, reading the weights as `weights` says. A device without `geometryShader` cannot
/// read them withheld, and the run is skipped, printed.
fn run(withheld: bool, weights: Weights) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            if withheld {
                options
                    .withheld_features
                    .insert(VkFeature::ProvokingVertexLast);
            }
            if weights == Weights::Withheld {
                options
                    .withheld_features
                    .insert(VkFeature::FragmentShaderBarycentric);
            }
        })
        .unwrap();
        let has = |feature| {
            device
                .compile_device()
                .profile()
                .enabled()
                .features
                .contains(&feature)
        };
        let native = !withheld && has(VkFeature::ProvokingVertexLast);
        if weights == Weights::Withheld && !has(VkFeature::GeometryShader) {
            eprintln!(
                "{}: no geometryShader, so no weights without BaryCoordKHR: skipped",
                device.name()
            );
            return;
        }
        let found = check_every_case(device, native, weights, &messages);
        assert!(found.covered > 10_000, "{found:?}");
        assert!(found.told_apart > 1_000, "{found:?}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// Every case draws Metal's mode 1 through the device's last provoking vertex where it has one
/// (else listed).
#[test]
fn draws_take_metal_s_last_vertex() {
    run(false, Weights::Device);
}

/// Every case draws Metal's mode 1 listed, the device's last provoking vertex withheld.
#[test]
fn draws_take_metal_s_last_vertex_listed() {
    run(true, Weights::Device);
}

/// Every case draws Metal's mode 1 with the weights a geometry stage writes, through the
/// device's last provoking vertex where it has one (else listed).
#[test]
fn draws_take_metal_s_last_vertex_through_a_geometry_stage() {
    run(false, Weights::Withheld);
}

/// Every case draws Metal's mode 1 listed with the weights a geometry stage writes.
#[test]
fn draws_take_metal_s_last_vertex_listed_through_a_geometry_stage() {
    run(true, Weights::Withheld);
}

/// The modes past 1 of [`the_third_way_is_drawn_as_mode_1_announced_where_metal_differs`], its
/// top bit set or not.
const THIRD: [u64; 6] = [
    2,
    0xffff_ffff,
    i64::MAX as u64,
    1 << 63,
    u64::MAX - 1,
    u64::MAX,
];

/// A mode past 1, which Metal draws a third way, is drawn as mode 1, bit for bit, on the cfg's
/// device, under `pv_flat` (flat inputs and the primitive ID). A triangle list and a fan, which
/// Metal's third way draws as mode 1 there, announce nothing; a triangle strip and a line strip,
/// which it draws otherwise, are announced at their draws (`provoking-vertex-unmatched`), each on
/// a device of its own, as a device announces a kind once.
#[test]
fn the_third_way_is_drawn_as_mode_1_announced_where_metal_differs() {
    use DrawPrimitiveType::{Public, TriangleFan};
    let silent = vec![
        Case::direct(Public(PrimitiveType::Triangle), Form::Direct, 12, 2),
        Case::direct(TriangleFan, Form::Direct, 9, 1),
    ];
    let announced = [
        Case::direct(Public(PrimitiveType::TriangleStrip), Form::Direct, 12, 2),
        Case::direct(Public(PrimitiveType::LineStrip), Form::Direct, 9, 1),
    ];
    let third = |messages: &Messages| {
        messages
            .take_notices()
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .filter(|notice| notice.degradation == Degradation::ProvokingVertexUnmatched)
            .filter(|notice| notice.object.contains("a mode past 1"))
            .count()
    };
    for (cases, announces) in [(silent, false)]
        .into_iter()
        .chain(announced.into_iter().map(|case| (vec![case], true)))
    {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device =
                validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
            let name = device.name().to_owned();
            let lane = Lane::new(device);
            let pipeline = lane.pipeline(FRAGMENTS[0]);
            let at = |case: &Case, mode| lane.draw(&pipeline, case, mode).unwrap();
            let last: Vec<Texels> = cases.iter().map(|case| at(case, 1)).collect();
            assert_eq!(third(&messages), 0, "{name}");
            for mode in THIRD {
                for (case, last) in cases.iter().zip(&last) {
                    assert!(at(case, mode) == *last, "{case:?} at {mode:#x} on {name}");
                }
            }
            assert_eq!(
                third(&messages) > 0,
                announces,
                "{cases:?} on {name}: announced"
            );
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// A draw that makes no primitive, too few vertices or indices for one (a line of one vertex,
/// quads of three), draws nothing at mode 1, as Metal's does, and announces no provoking kind
/// ([`PROVOKING`]): through the device's last vertex and with it withheld, of a pipeline that
/// reads flat inputs and weights, on the cfg's device.
#[test]
fn draws_of_no_primitive_announce_nothing() {
    use DrawPrimitiveType::{LineLoop, Public, QuadStrip, Quads, TriangleFan};
    for withheld in [false, true] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                if withheld {
                    options
                        .withheld_features
                        .insert(VkFeature::ProvokingVertexLast);
                }
            })
            .unwrap();
            let name = device.name().to_owned();
            let lane = Lane::new(device);
            // `pv_bare`: the flat vertex ID and the weights, no primitive ID.
            let pipeline = lane.pipeline(FRAGMENTS[4]);
            let short = [
                (Public(PrimitiveType::Line), 1),
                (Public(PrimitiveType::LineStrip), 1),
                (Public(PrimitiveType::Triangle), 2),
                (Public(PrimitiveType::TriangleStrip), 2),
                (LineLoop, 1),
                (TriangleFan, 2),
                (Quads, 3),
                (QuadStrip, 3),
            ];
            for (primitive, count) in short {
                let indices = (0..count).collect();
                for case in [
                    Case::direct(primitive, Form::Direct, count, 1),
                    Case::indexed(primitive, Form::Indexed(IndexType::UInt16), indices),
                ] {
                    let texels = lane.draw(&pipeline, &case, 1).unwrap();
                    assert!(
                        texels[2].iter().all(|t| t[3] != 1),
                        "{case:?} on {name}: drew"
                    );
                }
            }
            // The pipeline's stages' precision notes (a device's interpolation) are its own.
            let kinds: Vec<Degradation> = messages
                .take_notices()
                .iter()
                .filter_map(|m| m.degradation.as_ref())
                .map(|notice| notice.degradation)
                .filter(|kind| PROVOKING.contains(kind))
                .collect();
            assert!(kinds.is_empty(), "{name} (withheld {withheld}): {kinds:?}");
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// Draws of one encoder after the private setters, each kind in turn: under a restart index, a line
/// width of 2, triangles filled as lines and mode 1, draws of no primitive, direct then indexed,
/// of every public type and too few vertices or indices for one; and at mode 1, then at modes
/// past 1, then at 0, draws of three vertices or indices of every type, direct, indexed and
/// indirect. A draw after one of the same primitive type, of other arguments alone, is rebound
/// from it (`render/draw_again.rs`), and in a build with debug assertions every second one also
/// takes every step, which must work out what the rebound did. Every pipeline of the fixture, on
/// the cfg's device: the encoders end and their command buffers complete.
#[test]
fn draws_after_the_private_setters_rebind_what_every_step_binds() {
    use DrawPrimitiveType::{LineLoop, Public, TriangleFan};
    use PrimitiveType::{Line, LineStrip, Point, Triangle, TriangleStrip};
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let lane = Lane::new(device);
        // All zeros: indices of vertex 0, and indirect arguments of no vertex.
        let zeros = buffer_with(&lane.device, &[0; 64]).unwrap();
        let direct = |e: &mut RenderCommandEncoder, primitive: DrawPrimitiveType, count| {
            e.draw_primitives(primitive, 0, count, 1, 0)
        };
        let indexed = |e: &mut RenderCommandEncoder, primitive: DrawPrimitiveType, count| {
            e.draw_indexed_primitives(primitive, count, IndexType::UInt16, &zeros, 0, 1, 0, 0)
        };
        let three = |e: &mut RenderCommandEncoder, primitive: DrawPrimitiveType, at: usize| match at
        {
            0 => direct(e, primitive, 3),
            1 => indexed(e, primitive, 3),
            _ => e.draw_primitives_indirect(primitive, &zeros, 0),
        };
        let case = Case::direct(Public(Triangle), Form::Direct, 0, 0);
        for fragment in FRAGMENTS {
            let pipeline = lane.pipeline(fragment);
            lane.render(&case, |e| {
                e.set_render_pipeline_state(&pipeline)?;
                e.set_vertex_buffer(Some(&lane.positions), 0, 0)?;
                e.set_primitive_restart_enabled(true, 0xffff);
                e.set_line_width(2.0)?;
                e.set_triangle_fill_mode(TriangleFillMode::Lines)?;
                e.set_provoking_vertex_mode(1)?;
                direct(e, Public(Point), 0)?;
                indexed(e, Public(Point), 0)?;
                let least = [(Line, 2), (LineStrip, 2), (Triangle, 3), (TriangleStrip, 3)];
                for (primitive, least) in least {
                    for count in 0..least {
                        direct(e, Public(primitive), count)?;
                        indexed(e, Public(primitive), count)?;
                    }
                }
                Ok(())
            })
            .unwrap_or_else(|error| panic!("{}: empty draws: {error:?}", fragment.name));
            lane.render(&case, |e| {
                e.set_render_pipeline_state(&pipeline)?;
                e.set_vertex_buffer(Some(&lane.positions), 0, 0)?;
                e.set_provoking_vertex_mode(1)?;
                let at_one = [
                    (Public(Triangle), 0),
                    (Public(TriangleStrip), 0),
                    (Public(Triangle), 1),
                    (Public(TriangleStrip), 1),
                    (TriangleFan, 0),
                    (Public(Triangle), 2),
                    (Public(TriangleStrip), 2),
                    (Public(Line), 0),
                    (Public(LineStrip), 1),
                    (LineLoop, 0),
                ];
                for (primitive, at) in at_one {
                    three(e, primitive, at)?;
                }
                for mode in [2, 7, 1 << 32, 1 << 63, u64::MAX] {
                    e.set_provoking_vertex_mode(mode)?;
                    let past_one = [
                        (Public(Triangle), 0),
                        (Public(Triangle), 1),
                        (TriangleFan, 0),
                        (Public(TriangleStrip), 0),
                        (Public(TriangleStrip), 0),
                        (Public(Line), 0),
                    ];
                    for (primitive, at) in past_one {
                        three(e, primitive, at)?;
                    }
                }
                e.set_provoking_vertex_mode(0)?;
                for (primitive, at) in [(Public(Triangle), 1), (Public(Line), 0), (TriangleFan, 0)]
                {
                    three(e, primitive, at)?;
                }
                Ok(())
            })
            .unwrap_or_else(|error| panic!("{}: draws of three: {error:?}", fragment.name));
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// An execution of indirect render commands takes the first vertex at mode 1, as Metal's does: its
/// command draws, bit for bit, what the encoder's own draw does at mode 0, and not at mode 1; on
/// the cfg's device.
#[test]
fn executions_take_the_first_vertex() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let mut executed_draws = 0;
        // `pv_flat` and `pv_both`: the flat vertex ID, and with it the weights.
        for fragment in &FRAGMENTS[..2] {
            let pipeline = lane.pipeline_of(*fragment, true);
            for primitive in [
                PrimitiveType::Line,
                PrimitiveType::LineStrip,
                PrimitiveType::Triangle,
                PrimitiveType::TriangleStrip,
            ] {
                let case = Case::direct(DrawPrimitiveType::Public(primitive), Form::Direct, 12, 2);
                let at = |mode| lane.draw(&pipeline, &case, mode).unwrap();
                let (first, last) = (at(0), at(1));
                let mut descriptor = IndirectCommandBufferDescriptor::new();
                descriptor.command_types = IndirectCommandType::DRAW;
                descriptor.inherit_buffers = true;
                let icb = lane
                    .device
                    .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
                    .unwrap();
                let command = icb.indirect_render_command_at(0).unwrap();
                command.set_render_pipeline_state(&pipeline).unwrap();
                command
                    .draw_primitives(
                        primitive,
                        u64::from(case.start),
                        u64::from(case.count),
                        1,
                        0,
                    )
                    .unwrap();
                let executed = lane
                    .render(&case, |encoder| {
                        encoder.set_render_pipeline_state(&pipeline)?;
                        encoder.set_vertex_buffer(Some(&lane.positions), 0, 0)?;
                        encoder.set_provoking_vertex_mode(1)?;
                        encoder.execute_commands_in_buffer(&icb, 0..1)
                    })
                    .unwrap_or_else(|e| panic!("{case:?} {fragment:?} on {name}: {e}"));
                assert!(
                    executed == first,
                    "{case:?} {fragment:?} on {name}: the execution is not mode 0's draw"
                );
                assert!(
                    executed != last,
                    "{case:?} {fragment:?} on {name}: the modes draw alike"
                );
                executed_draws += 1;
            }
        }
        eprintln!("{name}: {executed_draws} executions at mode 1 drew mode 0's draws");
    }
    layer_reported_nothing_at_all(&messages);
}

/// The model's strip: an odd triangle's order at mode 1 is `(j + 1, j, j + 2)`, at mode 0
/// `(j, j + 2, j + 1)`, and an even one's alike at both.
#[test]
fn metal_s_strip_orders_differ() {
    let strip = DrawPrimitiveType::Public(PrimitiveType::TriangleStrip);
    let case = Case::direct(strip, Form::Direct, 6, 0);
    let (first, last) = (case.primitives(false), case.primitives(true));
    assert_eq!(first[1], [1, 3, 2]);
    assert_eq!(last[1], [2, 1, 3]);
    assert_eq!(first[0], last[0]);
}
