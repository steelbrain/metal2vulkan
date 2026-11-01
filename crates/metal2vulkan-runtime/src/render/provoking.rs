//! The provoking vertex: Apple's `setProvokingVertexMode:`.
//!
//! Measured on the M3 Max with macOS 27.0.1 (each primitive's flat inputs, `[[primitive_id]]`,
//! barycentric weights, slice and viewport, solved for its vertices):
//!
//! - Mode 0 takes a primitive's flat inputs, and the slice and viewport its vertex function
//!   selects, from its first vertex; mode 1 from its last. Every other value, its top bit set or
//!   not, draws as 2 does, a third way: a triangle list's, a fan's and Metal's private triangle
//!   types' primitives as mode 1 draws them, flat inputs, `[[primitive_id]]` and weights, but for
//!   a pipeline whose vertex function selects a slice or viewport, whose triangles take their
//!   middle vertex; a triangle strip's odd triangle `(i + 2, i + 1, i)` from `i`; every line's
//!   flat inputs and `[[primitive_id]]` 0; points whose vertex function writes their size another
//!   way than mode 1's; and patches as mode 0. No value is a validation error.
//!
//! The runtime draws the third way as mode 1, and patches as mode 0, announcing at each draw
//! whose shape Metal's third way draws otherwise than mode 1, as far as its pipeline state reads
//! the difference ([`RenderLowering::third_way`]): a strip of triangles whose pipeline reads flat
//! inputs, weights, or a slice or viewport; lines whose pipeline reads flat inputs, the primitive
//! ID, or a slice or viewport; sized points whose pipeline reads flat inputs or the primitive ID;
//! triangles of a pipeline that selects a slice or viewport; and mesh draws, unmeasured, whose
//! pipeline reads flat inputs, weights, or a slice or viewport
//! (`Degradation::ProvokingVertexUnmatched`). A list's, a fan's and a private type's triangles of
//! any other pipeline, and patches, draw as Metal's, silent.
//! - At mode 1 the vertex is a line list's `2j + 1`, a line strip's `j + 1`, a triangle list's
//!   `3j + 2`, a triangle strip's `j + 2`, and Metal's private types' list primitives' last
//!   (`DrawPrimitiveType`). The barycentric order is mode 0's but for a triangle strip's odd
//!   triangle, `(j + 1, j, j + 2)` where mode 0 gives `(j, j + 2, j + 1)`: the Vulkan
//!   specification's last-vertex strip order. Indexed draws (16- and 32-bit, a base vertex,
//!   restarts, each piece's parity its own), indirect draws, instances, tessellated patches and
//!   mesh draws follow the mode; an execution of indirect render commands does not, its
//!   primitives taking the first vertex.
//! - The mode is the encoder's: the last call wins, before or after its pipeline state is set,
//!   and a new encoder starts at the first vertex.
//!
//! A draw at the first vertex, one of a pipeline whose primitives' vertex order nothing reads
//! (its fragment function reads no flat input and no `[[barycentric_coord]]`, and its vertex
//! function selects no slice or viewport), and one that makes no primitive (too few vertices or
//! indices for one, which Metal draws as nothing), draws as it is ([`Provoking::AsIs`]),
//! announcing nothing. Otherwise:
//!
//! - **Native**: on a device that enables `VK_EXT_provoking_vertex`'s `provokingVertexLast` and
//!   whose pipelines of one render pass instance may differ in mode
//!   (`provokingVertexModePerPipeline`), the draw's `VkPipeline` takes the last vertex
//!   (`FixedState::provoking_last`), whose primitive topologies' provoking vertices and
//!   barycentric orders are Metal's mode 1 (the Vulkan specification's primitive topologies and
//!   barycentric interpolation). A list the runtime draws for a draw (a strip restarting at
//!   another index than Vulkan's, a private type's) is written in Metal's mode-1 order
//!   ([`RewriteOrder::Last`]). Announced (`Degradation::ProvokingVertexUnmatched`): a strip whose
//!   vertex function computes `[[primitive_id]]` into its varying (a device without
//!   `geometryShader` and `tessellationShader`), whose IDs then read its primitives' last
//!   vertices; and a slice or viewport the vertex function selects where the pipeline selects it
//!   in a synthesized geometry stage (a device without `shaderOutputLayer` or
//!   `shaderOutputViewportIndex`, an index that differs between a primitive's vertices, or
//!   weights the stage writes), from each primitive's first vertex. Tessellated patches take the
//!   vertex Vulkan leaves to the device ("the provoking vertex may be any of the vertices in each
//!   primitive", the Vulkan specification's Flat Shading), which
//!   `PrecisionNote::TessellationTriangleOrder` covers.
//! - **Listed**, on any other device: the draw is drawn as the list the index rewrite pass writes
//!   (the CPU for a direct draw without indices, `index_rewrite.rs`), each primitive in Metal's
//!   mode-1 order with its last vertex moved first ([`RewriteOrder::LastFirst`]), so the
//!   pipeline's first vertex is Metal's: exact for a triangle's flat inputs, slice and viewport
//!   (a rotated triangle covers the same pixels and interpolates its smooth inputs alike, bit for
//!   bit, over 200 perspective triangles on the M3 Max). A line drawn from its last vertex to its
//!   first covers other pixels and interpolates otherwise (206 pixels and 2922 interpolated
//!   values of 200 lines on the M3 Max), announced (`Degradation::ProvokingVertexLines`). A strip
//!   whose fragment function reads barycentrics and no flat input is listed in Metal's mode-1
//!   order instead ([`RewriteOrder::Last`]), exact; a list, a line or a private type of such a
//!   pipeline draws as it is, its order mode 0's. A triangle drawn with its last vertex first
//!   whose fragment function reads barycentrics runs the variant that takes Metal's weights back
//!   from the drawn triangle's (`BindState::rotated_weights`): Metal's first weight bit for bit,
//!   since Metal interpolates the rotated triangle alike, and its other two Metal's but for
//!   rounding (Metal probes on the M3 Max, 200 triangles: 9,150 of 67,821 pixels' first two
//!   weights bit for bit, the rest within `2^-15`). What the list cannot draw as Metal does is
//!   announced (`Degradation::ProvokingVertexUnmatched`): those two weights of a triangle whose
//!   fragment function reads flat inputs and barycentrics, and patches and mesh draws, whose
//!   primitives no list holds (their flat inputs from the first vertex).
//!
//! An execution's commands take the first vertex, at every mode: its snapshot of the encoder's
//! state does (`render/executions.rs`).

use super::{Assembly, RenderLowering};
use crate::degradation::Degradation;
use crate::pipeline::GraphicsState as DrawingState;
use crate::render_state::EncoderState;
use crate::Device;
use metal2vulkan::bind::{DrawCall, RewriteOrder};
use metal2vulkan::device::{VkFeature, VkProperty};
use metal2vulkan::metal::PrimitiveType;

/// How a draw takes the encoder's provoking vertex (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Provoking {
    /// As it is: at the first vertex, or nothing reads its primitives' vertex order.
    AsIs,
    /// Through a `VkPipeline` whose provoking vertex is the last.
    Native,
    /// As the list of its primitives in `RewriteOrder`, through a first-vertex pipeline.
    Listed(RewriteOrder),
    /// An emulated mesh draw of lines or triangles, each primitive drawn from its last index
    /// first through a first-vertex pipeline (`BindState::mesh_last_first`).
    LastCorners,
}

/// What of a draw's pipeline variant its provoking vertex asks for (`BindState`): weights taken
/// back from triangles drawn with their last vertex first, and an emulated mesh's primitives
/// drawn from their last index first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ProvokingForms {
    /// `BindState::rotated_weights`.
    pub(super) rotated_weights: bool,
    /// `BindState::mesh_last_first`.
    pub(super) mesh_last_first: bool,
}

impl Provoking {
    /// The order a list the runtime draws for the draw is written in: Metal's mode-1 order under
    /// a last-vertex pipeline, the listed order, else mode 0's.
    pub(super) fn order(self) -> RewriteOrder {
        match self {
            Provoking::AsIs | Provoking::LastCorners => RewriteOrder::First,
            Provoking::Native => RewriteOrder::Last,
            Provoking::Listed(order) => order,
        }
    }

    /// The variant a draw of `pipeline` as `assembly` taken so runs: where it draws triangles
    /// with Metal's last vertex first and its fragment function reads `[[barycentric_coord]]`,
    /// the one that takes Metal's weights back from the drawn triangle's; and an emulated mesh
    /// draw's whose primitives are drawn from their last index first.
    pub(super) fn forms(self, assembly: Assembly, pipeline: &DrawingState) -> ProvokingForms {
        let triangles = match Class::of(assembly) {
            Class::Triangles { .. } => self == Provoking::Listed(RewriteOrder::LastFirst),
            _ => self == Provoking::LastCorners && mesh_corners(pipeline) == Some(3),
        };
        ProvokingForms {
            rotated_weights: triangles && reads(pipeline).1,
            mesh_last_first: self == Provoking::LastCorners,
        }
    }
}

/// The indices of each primitive of `pipeline`'s emulated mesh function (`GraphicsState::
/// mesh_records`): 1, 2 or 3; `None` where its mesh function is not emulated or it has none.
fn mesh_corners(pipeline: &DrawingState) -> Option<u32> {
    let graphics = &pipeline.base().graphics;
    graphics.mesh_records.as_ref().map(|r| r.corners)
}

/// How Metal draws `setProvokingVertexMode:`'s `mode` (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// The first vertex: 0.
    First,
    /// The last vertex: 1.
    Last,
    /// The third way: every other value.
    Third,
}

impl Mode {
    fn of(mode: u64) -> Mode {
        match mode {
            0 => Mode::First,
            1 => Mode::Last,
            _ => Mode::Third,
        }
    }
}

/// What of a draw of `pipeline` reads its primitives' provoking vertex and vertex order: a flat
/// input, or a slice or viewport its vertex function selects; and the barycentric weights.
fn reads(pipeline: &DrawingState) -> (bool, bool) {
    let graphics = &pipeline.base().graphics;
    let flat = graphics.flat_fragment_inputs
        || pipeline.vertex_chooses_slice()
        || graphics.writes_viewport_index;
    (flat, graphics.reads_barycentric_coord)
}

/// Whether `device` draws the last provoking vertex natively: it enables
/// `VK_EXT_provoking_vertex`'s `provokingVertexLast`, and its pipelines of one render pass
/// instance may differ in mode (`provokingVertexModePerPipeline`; where they may not, "all
/// pipelines bound within a render pass instance must have the same provokingVertexMode", the
/// Vulkan specification's Flat Shading, and an execution's commands, which take the first vertex,
/// share the encoder's instances).
fn native(device: &Device) -> bool {
    let enabled = device.compile_device().profile().enabled();
    enabled.features.contains(&VkFeature::ProvokingVertexLast)
        && enabled
            .properties
            .contains(&VkProperty::ProvokingVertexModePerPipeline)
}

/// Whether a draw of `pipeline` as `assembly` under `state` runs a `VkPipeline` of the last
/// provoking vertex ([`Provoking::Native`]): patches at a mode past 1 take the first.
pub(super) fn natively_last(
    device: &Device,
    state: &EncoderState,
    (pipeline, assembly): (&DrawingState, Assembly),
) -> bool {
    let (flat, weights) = reads(pipeline);
    state.provoking_last()
        && !third_way_first(state, assembly)
        && (flat || weights)
        && native(device)
}

/// Whether a draw as `assembly` under `state` takes the first vertex at a mode past 1, as Metal's
/// third way draws it: patches.
fn third_way_first(state: &EncoderState, assembly: Assembly) -> bool {
    state.provoking_third() && matches!(assembly, Assembly::Patches)
}

/// Whether `draw` of `assembly` makes no primitive: a direct or indexed draw of too few vertices
/// or indices for one of its primitives (a line of one vertex, quads of three), which Metal draws
/// as nothing. An indirect draw's counts are the GPU's, and a draw of points, patches or meshes
/// that is not empty makes some (an empty draw records nothing before it asks).
fn makes_none(assembly: Assembly, draw: &DrawCall) -> bool {
    let count = match *draw {
        DrawCall::Primitives { vertex_count, .. } => vertex_count,
        DrawCall::Indexed { index_count, .. } => index_count,
        _ => return false,
    };
    assembly
        .program()
        .is_some_and(|program| program.capacity(u64::from(count)) == 0)
}

/// The classes of primitives a draw assembles, as the provoking vertex treats them.
enum Class {
    Points,
    Lines,
    Triangles {
        strip: bool,
    },
    /// Patches and meshes: primitives no list the runtime writes holds.
    Unlisted,
}

impl Class {
    /// What a draw of the class draws, in words.
    fn name(&self) -> &'static str {
        match self {
            Class::Points => "points",
            Class::Lines => "lines",
            Class::Triangles { strip: true } => "a triangle strip",
            Class::Triangles { strip: false } => "triangles",
            Class::Unlisted => "patches or meshes",
        }
    }

    fn of(assembly: Assembly) -> Class {
        use metal2vulkan::bind::RewriteProgram as R;
        match assembly {
            Assembly::Primitives(PrimitiveType::Point) => Class::Points,
            Assembly::Primitives(PrimitiveType::Line | PrimitiveType::LineStrip)
            | Assembly::Private(R::LineLoop) => Class::Lines,
            Assembly::Primitives(PrimitiveType::TriangleStrip) => Class::Triangles { strip: true },
            Assembly::Primitives(PrimitiveType::Triangle) | Assembly::Private(_) => {
                Class::Triangles { strip: false }
            }
            Assembly::Patches | Assembly::Meshes => Class::Unlisted,
        }
    }
}

impl RenderLowering {
    /// The lowering of `setProvokingVertexMode:`'s `mode`: the encoder's draws from here take
    /// the first vertex or the last, the third way drawn as the last but for patches, each draw
    /// it draws otherwise than Metal announced there (the module docs).
    pub(crate) fn set_provoking_vertex(&mut self, mode: u64) {
        let mode = Mode::of(mode);
        let (last, third) = (mode != Mode::First, mode == Mode::Third);
        // Only a change takes the state to a new version (`versioned.rs`), so the draws' memo
        // holds across a call that repeats the mode.
        if self.state.provoking_last() != last {
            self.state.set_provoking_last(last);
        }
        if self.state.provoking_third() != third {
            self.state.set_provoking_third(third);
        }
    }

    /// Announce `call`'s draw of `pipeline` as `assembly` at a mode past 1 where Metal's third
    /// way draws it otherwise than mode 1, as the runtime does, as far as the pipeline state
    /// reads the difference (the module docs). Patches, which take the first vertex as Metal's,
    /// and a list's, a fan's and a private type's triangles of a pipeline that selects no slice
    /// or viewport, which mode 1 draws as Metal, announce nothing.
    fn third_way(&self, call: &'static str, pipeline: &DrawingState, assembly: Assembly) {
        let graphics = &pipeline.base().graphics;
        let flat = graphics.flat_fragment_inputs;
        let selects = pipeline.vertex_chooses_slice() || graphics.writes_viewport_index;
        let (weights, ids) = (
            graphics.reads_barycentric_coord,
            graphics.reads_primitive_id,
        );
        let differs = match Class::of(assembly) {
            Class::Points => (graphics.writes_point_size && (flat || ids)).then_some(
                "a point whose vertex function writes its size is drawn otherwise than at mode 1",
            ),
            Class::Lines => (flat || ids || selects)
                .then_some("every line's flat inputs and [[primitive_id]] read 0"),
            Class::Triangles { strip: true } => (flat || weights || selects).then_some(
                "a strip's odd triangle takes its first vertex, its weights in the order (i + 2, \
                 i + 1, i)",
            ),
            Class::Triangles { strip: false } => selects.then_some(
                "a triangle of a pipeline that selects its slice or viewport \
                                   takes its middle vertex",
            ),
            Class::Unlisted if matches!(assembly, Assembly::Meshes) => {
                (flat || weights || selects).then_some("a mesh draw is unmeasured there")
            }
            Class::Unlisted => None,
        };
        let Some(how) = differs else {
            return;
        };
        self.core
            .device()
            .degrade(Degradation::ProvokingVertexUnmatched, || {
                (
                    format!(
                        "{call}: a draw of {} under setProvokingVertexMode: of a mode past 1, \
                         whose pipeline state reads what Metal draws a third way there",
                        Class::of(assembly).name()
                    ),
                    format!("Metal draws such a mode a third way: {how}; the runtime draws it as mode 1"),
                )
            });
    }

    /// How `call`'s `draw` of `pipeline` as `assembly` takes the encoder's provoking vertex (the
    /// module docs), announcing what a listed draw cannot draw as Metal does. A draw that makes
    /// no primitive draws nothing, as on Metal, and announces nothing ([`makes_none`]). A draw of
    /// lines as bands (`band`, `render/line_width.rs`), whose every corner takes a flat input
    /// from its line's first vertex, takes Metal's last vertex from the list led by each line's
    /// last vertex on every device, silently: the band covers the same pixels drawn either way.
    pub(super) fn provoking(
        &self,
        call: &'static str,
        pipeline: &DrawingState,
        (assembly, band): (Assembly, bool),
        draw: &DrawCall,
    ) -> Provoking {
        if !self.state.provoking_last() || makes_none(assembly, draw) {
            return Provoking::AsIs;
        }
        if self.state.provoking_third() {
            if third_way_first(&self.state, assembly) {
                return Provoking::AsIs;
            }
            self.third_way(call, pipeline, assembly);
        }
        let (flat, weights) = reads(pipeline);
        if !(flat || weights) {
            return Provoking::AsIs;
        }
        if band {
            return match flat {
                true => Provoking::Listed(RewriteOrder::LastFirst),
                false => Provoking::AsIs,
            };
        }
        let device = self.core.device();
        if native(device) {
            // A strip's primitive ID computed into a varying reads its last vertex's stream
            // position there, past Metal's count by the strip's vertices less one
            // (`LoweringInputs::primitive_id_varying`).
            let strip = matches!(
                assembly,
                Assembly::Primitives(PrimitiveType::LineStrip | PrimitiveType::TriangleStrip)
            );
            if strip && super::writes_primitive_ids(pipeline) {
                device.degrade(Degradation::ProvokingVertexUnmatched, || {
                    (
                        format!(
                            "{call}: a draw of a strip under setProvokingVertexMode: 1 whose \
                             vertex function computes [[primitive_id]] into its varying"
                        ),
                        "the varying reads each primitive's last vertex, so the IDs run one \
                         or two past Metal's"
                            .to_owned(),
                    )
                });
            }
            // A pipeline with synthesized geometry stages selects its slice and viewport there,
            // from each primitive's first vertex (`link::geometry`; `lower::stage_io`'s
            // `IndexStore::Selected`).
            let base = pipeline.base();
            let geometry = base
                .stages
                .iter()
                .any(|s| s.stage == metal2vulkan::vk::ShaderStage::GEOMETRY);
            let selects = pipeline.vertex_chooses_slice() || base.graphics.writes_viewport_index;
            if geometry && selects {
                device.degrade(Degradation::ProvokingVertexUnmatched, || {
                    (
                        format!(
                            "{call}: a draw under setProvokingVertexMode: 1 whose vertex function \
                             selects its primitives' slice or viewport"
                        ),
                        "the pipeline selects it in a geometry stage, from each primitive's \
                         first vertex"
                            .to_owned(),
                    )
                });
            }
            return Provoking::Native;
        }
        match (Class::of(assembly), flat) {
            (Class::Points, _) => Provoking::AsIs,
            (Class::Lines, true) => {
                device.degrade(Degradation::ProvokingVertexLines, || {
                    (
                        format!(
                            "{call}: a draw of lines under setProvokingVertexMode: 1 whose \
                             pipeline state reads their provoking vertex"
                        ),
                        "the device has no last provoking vertex \
                         (VK_EXT_provoking_vertex's provokingVertexLast, with \
                         provokingVertexModePerPipeline), so each line is drawn from its last \
                         vertex to its first"
                            .to_owned(),
                    )
                });
                Provoking::Listed(RewriteOrder::LastFirst)
            }
            // A line's weights, and a list's, are in mode 0's order at mode 1 too.
            (Class::Lines | Class::Triangles { strip: false }, false) => Provoking::AsIs,
            (Class::Triangles { strip: true }, false) => Provoking::Listed(RewriteOrder::Last),
            (Class::Triangles { .. }, true) => {
                if weights {
                    device.degrade(Degradation::ProvokingVertexUnmatched, || {
                        (
                            format!(
                                "{call}: a draw of triangles under setProvokingVertexMode: 1 \
                                 whose fragment function reads flat inputs and \
                                 [[barycentric_coord]]"
                            ),
                            "the device has no last provoking vertex, so each triangle is \
                             drawn from its last vertex, and its second and third weights are \
                             computed from that triangle's, Metal's but for rounding (within \
                             2^-15)"
                                .to_owned(),
                        )
                    });
                }
                Provoking::Listed(RewriteOrder::LastFirst)
            }
            // A mesh's weights are in mode 0's order at mode 1 too, as a list's are (a Metal
            // probe on the M3 Max), so one whose pipeline reads no flat input draws as it is.
            (Class::Unlisted, false) if matches!(assembly, Assembly::Meshes) => Provoking::AsIs,
            // An emulated mesh draws its primitives through a vertex stage of its own, which
            // reads each from its last index first: a triangle's flat inputs, slice and viewport
            // are Metal's, and its weights taken back to Metal's order but for rounding; a line
            // runs the other way, announced as a listed line is. One of points draws as it is.
            (Class::Unlisted, true)
                if matches!(assembly, Assembly::Meshes) && mesh_corners(pipeline).is_some() =>
            {
                match mesh_corners(pipeline) {
                    Some(1) => return Provoking::AsIs,
                    Some(2) => device.degrade(Degradation::ProvokingVertexLines, || {
                        (
                            format!(
                                "{call}: a mesh draw of lines under setProvokingVertexMode: 1 \
                                 whose pipeline state reads their provoking vertex"
                            ),
                            "the device has no last provoking vertex \
                             (VK_EXT_provoking_vertex's provokingVertexLast, with \
                             provokingVertexModePerPipeline), so each line is drawn from its \
                             last vertex to its first"
                                .to_owned(),
                        )
                    }),
                    _ if weights => device.degrade(Degradation::ProvokingVertexUnmatched, || {
                        (
                            format!(
                                "{call}: a mesh draw of triangles under setProvokingVertexMode: \
                                 1 whose fragment function reads flat inputs and \
                                 [[barycentric_coord]]"
                            ),
                            "the device has no last provoking vertex, so each triangle is drawn \
                             from its last vertex, and its second and third weights are \
                             computed from that triangle's, Metal's but for rounding (within \
                             2^-15)"
                                .to_owned(),
                        )
                    }),
                    _ => {}
                }
                Provoking::LastCorners
            }
            (Class::Unlisted, _) => {
                device.degrade(Degradation::ProvokingVertexUnmatched, || {
                    (
                        format!(
                            "{call}: a draw of patches or meshes under setProvokingVertexMode: 1 \
                             whose pipeline state reads their provoking vertex"
                        ),
                        "the device has no last provoking vertex, and no list holds the \
                         primitives, so they take the first vertex"
                            .to_owned(),
                    )
                });
                Provoking::AsIs
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Mode;

    /// The values Metal takes: 0 the first vertex, 1 the last, every other value the third way,
    /// its top bit set or not (Metal probes on the M3 Max, macOS 27.0.1).
    #[test]
    fn the_modes_are_metal_s_measured_ones() {
        assert_eq!(Mode::of(0), Mode::First);
        assert_eq!(Mode::of(1), Mode::Last);
        for mode in [
            2,
            3,
            5,
            0xff,
            0x8000_0000,
            0xffff_ffff,
            1 << 32,
            1 << 62,
            i64::MAX as u64,
            1 << 63,
            u64::MAX - 1,
            u64::MAX,
        ] {
            assert_eq!(Mode::of(mode), Mode::Third, "{mode:#x}");
        }
    }
}
