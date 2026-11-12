//! Device tests of patch index and control-point index buffers on the device the cfg selects, under
//! the validation layer. The owned fixture `rt6_patch_index`, a `[[patch(triangle, 3)]]`
//! post-tessellation vertex function drawn with rasterization disabled in a 1x1 pass without
//! attachments, stores `ptag << 24 | patch_id << 16 | tag0 << 8 | tag1 << 4
//! | tag2` (its per-patch attribute, its patch index, and its three control points' tags) at mark
//! `tag0 / 3` of buffer 2. Control point `e` of buffer 0 (per patch control point) has tag `e`, so
//! without a control-point index buffer patch `p`'s points are `3p` to `3p + 2` and it stores at
//! mark `p`; patch `p`'s attribute in buffer 1 (per patch) is `0xa0 + p`; every patch's factors
//! are 1 but patch 3's, which are 0, so it culls.
//!
//! Each case's marks are those Metal stored (a Metal probe on an M4 with macOS 27.0.1, with and
//! without API validation): patch `i` of a draw with a patch index buffer is the patch its `uint`
//! at `patchStart + i` names, for its control points, its attribute, and its `[[patch_id]]`, while
//! its factors stay those at `patchStart + i`; control point `k` of the patch whose index is `p`,
//! drawn by `drawIndexedPatches`, is the index at `3p + k` of the control-point index buffer from
//! its offset, `ushort` or `uint` as the pipeline's `tessellationControlPointIndexType` states.
//! What Metal's validation stops of a patch index buffer is `patches.rs`'s; of a control-point
//! index buffer, this file's (probes 14d and 14e).
//!
//! Each test runs on the device's tessellation stages and on the emulation (`patches::Path`).

use super::apart_writes::NotedWrites;
use super::patches::{on_path_noted, Path};
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{
    TessellationControlPointIndexType, TessellationFactorStepFunction, TessellationPartitionMode,
    VertexFormat, VertexStepFunction,
};
use metal2vulkan::pipeline::{
    RenderPipelineDescriptor, StageSource, TessellationDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DrawProblem, Error, Function, Invalid,
    PipelineOptions, Problem, RenderPassDescriptor, RenderPipelineState,
};

/// The patches whose factors the factor buffer holds; patch 3's cull.
const FACTOR_PATCHES: u16 = 16;

/// Patch `i` of four is patch `[1, 3, 0, 2][i]`.
const SHUFFLED: &[u32] = &[1, 3, 0, 2];

/// Patch `p`'s control points are those of patch `3 - p` reversed patch by patch.
const REVERSED: &[u32] = &[9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2];
const REVERSED16: &[u16] = &[9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2];

/// After 2 zero bytes, which a read from offset 2 rounded down to 0 takes with them, the 32-bit
/// indices 0, 9, 10, 11, 6, 7: patch 0's control points 0, 9, 10 and patch 1's 11, 6, 7
/// (`ROUNDED_DOWN`). A read from offset 2 itself would take `0x9_0000` and on, past the
/// control points.
const SPLIT32: &[u32] = &[0x9_0000, 0xa_0000, 0xb_0000, 0x6_0000, 0x7_0000, 0x8_0000];
/// After 1 zero byte, the 16-bit indices 0, 9, 10, 11, 6, 7 likewise from offset 0.
const SPLIT16: &[u16] = &[0x900, 0xa00, 0xb00, 0x600, 0x700, 0x800];
/// The marks of two patches through `SPLIT32` or `SPLIT16` read from offset 0.
const ROUNDED_DOWN: [Option<u32>; 4] = [Some(0xa000_009a), Some(0), Some(0), Some(0xa101_0b67)];

/// A control-point index buffer's indices, of its pipeline's index type.
#[derive(Clone, Copy)]
enum Points {
    U16(&'static [u16]),
    U32(&'static [u32]),
}

impl Points {
    /// The pipeline's `tessellationControlPointIndexType`.
    fn index_type(self) -> TessellationControlPointIndexType {
        match self {
            Points::U16(_) => TessellationControlPointIndexType::UInt16,
            Points::U32(_) => TessellationControlPointIndexType::UInt32,
        }
    }

    fn bytes(self) -> Vec<u8> {
        match self {
            Points::U16(indices) => indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
            Points::U32(indices) => indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
        }
    }
}

/// One draw: its patch index buffer's indices and the bytes before them, its offset (`None`: no
/// buffer), its control-point index buffer's likewise (`None`: `drawPatches`), its patch start
/// and count, and whether it takes them as indirect arguments.
#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    patch_indices: Option<(&'static [u32], u64)>,
    control_points: Option<(Points, u64)>,
    patch_start: u32,
    patch_count: u32,
    indirect: bool,
}

impl Case {
    /// Four patches from patch 0, without a patch index or control-point index buffer.
    const fn new(name: &'static str) -> Case {
        Case {
            name,
            patch_indices: None,
            control_points: None,
            patch_start: 0,
            patch_count: 4,
            indirect: false,
        }
    }

    /// Two patches from patch 0 through `points` from `offset`.
    const fn indexed(name: &'static str, points: Points, offset: u64) -> Case {
        Case {
            control_points: Some((points, offset)),
            patch_count: 2,
            ..Case::new(name)
        }
    }

    /// The pipeline's `tessellationControlPointIndexType`.
    fn index_type(&self) -> TessellationControlPointIndexType {
        self.control_points
            .map_or(TessellationControlPointIndexType::None, |(p, _)| {
                p.index_type()
            })
    }
}

/// Each case and the marks Metal stored (the module docs).
fn cases() -> Vec<(Case, [u32; 4])> {
    vec![
        (
            Case::new("plain"),
            [0xa000_0012, 0xa101_0345, 0xa202_0678, 0],
        ),
        (
            Case {
                patch_start: 1,
                patch_count: 2,
                ..Case::new("plain_start1")
            },
            [0, 0xa101_0345, 0xa202_0678, 0],
        ),
        (
            Case {
                patch_indices: Some((SHUFFLED, 0)),
                ..Case::new("pib")
            },
            [0xa000_0012, 0xa101_0345, 0, 0xa303_09ab],
        ),
        (
            Case {
                patch_indices: Some((SHUFFLED, 0)),
                patch_start: 1,
                patch_count: 2,
                ..Case::new("pib_start1")
            },
            [0xa000_0012, 0, 0, 0xa303_09ab],
        ),
        (
            Case {
                patch_indices: Some((SHUFFLED, 4)),
                patch_count: 3,
                ..Case::new("pib_offset4")
            },
            [0xa000_0012, 0xa101_0345, 0, 0xa303_09ab],
        ),
        // An offset off a multiple of 4, which Metal's validation stops, read from the offset
        // rounded down, as plain Metal reads it (an indirect command buffer's patch draw of the
        // words 9, 3, 1 at offset 2 drew patches 9 and 3; Metal probe
        // case `pib_offset2` on an M4 drew nothing through indices that read as 0x10000 and 0x30000
        // so): after the 2 zero bytes, the words 0x2_0000 and 0x1_0000 read as patches 0 and 2,
        // where the offset itself would read patches past the buffers.
        (
            Case {
                patch_indices: Some((&[0x2_0000, 0x1_0000], 2)),
                patch_count: 2,
                ..Case::new("pib_offset2")
            },
            [0xa000_0012, 0, 0xa202_0678, 0],
        ),
        (
            Case {
                patch_indices: Some((&[2, 2, 0, 0], 0)),
                patch_count: 2,
                ..Case::new("pib_repeat")
            },
            [0, 0, 0xa202_0678, 0],
        ),
        (
            Case {
                patch_indices: Some((&[3, 3, 3, 0], 0)),
                patch_start: 3,
                patch_count: 1,
                ..Case::new("pib_at3")
            },
            [0; 4],
        ),
        (
            Case {
                patch_indices: Some((SHUFFLED, 0)),
                patch_start: 1,
                patch_count: 2,
                indirect: true,
                ..Case::new("pib_indirect")
            },
            [0xa000_0012, 0, 0, 0xa303_09ab],
        ),
        (
            Case::indexed("cpi32", Points::U32(REVERSED), 0),
            [0, 0, 0xa101_0678, 0xa000_09ab],
        ),
        (
            Case {
                patch_start: 1,
                ..Case::indexed("cpi32_start1", Points::U32(REVERSED), 0)
            },
            [0, 0xa202_0345, 0xa101_0678, 0],
        ),
        (
            Case::indexed("cpi32_offset4", Points::U32(REVERSED), 4),
            [0, 0, 0xa101_0678, 0xa000_09ab],
        ),
        (
            Case::indexed("cpi32_exact", Points::U32(REVERSED.split_at(6).0), 0),
            [0, 0, 0xa101_0678, 0xa000_09ab],
        ),
        (
            Case::indexed("cpi16", Points::U16(REVERSED16), 0),
            [0, 0, 0xa101_0678, 0xa000_09ab],
        ),
        (
            Case::indexed("cpi16_offset2", Points::U16(REVERSED16), 2),
            [0, 0, 0xa101_0678, 0xa000_09ab],
        ),
        (
            Case::indexed("cpi32_mixed", Points::U32(&[0, 4, 8, 11, 7, 3]), 0),
            [0xa000_0048, 0, 0, 0xa101_0b73],
        ),
        (
            Case {
                patch_indices: Some((&[3, 1], 0)),
                ..Case::indexed("cpi32_pib", Points::U32(REVERSED), 0)
            },
            [0xa303_0012, 0, 0xa101_0678, 0],
        ),
        (
            Case {
                patch_indices: Some((&[2, 0, 1], 0)),
                patch_start: 1,
                ..Case::indexed("cpi32_pib_start1", Points::U32(REVERSED), 0)
            },
            [0, 0, 0xa101_0678, 0xa000_09ab],
        ),
        (
            Case {
                patch_start: 1,
                indirect: true,
                ..Case::indexed("cpi32_indirect", Points::U32(REVERSED), 0)
            },
            [0, 0xa202_0345, 0xa101_0678, 0],
        ),
        (
            Case {
                patch_indices: Some((&[2, 0, 1], 0)),
                patch_start: 1,
                indirect: true,
                ..Case::indexed("cpi32_pib_indirect", Points::U32(REVERSED), 0)
            },
            [0, 0, 0xa101_0678, 0xa000_09ab],
        ),
    ]
}

/// The lane's device, its queue, and the fixture's function.
pub(super) struct Lane {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    function: Function,
}

impl Lane {
    pub(super) fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("rt6_patch_index.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            function: library.new_function_with_name("rt6_patch_index").unwrap(),
            device,
        }
    }

    /// The fixture's pipeline as probe 14d made it, of control-point index type `index_type`:
    /// rasterization disabled, the control points' tags at buffer 0 and the patches' attributes
    /// at buffer 1, integer partitioning to 16, half factors per patch.
    fn pipeline(
        &self,
        index_type: TessellationControlPointIndexType,
    ) -> Result<RenderPipelineState, Error> {
        self.device
            .new_render_pipeline_state(&self.descriptor(index_type), &PipelineOptions::default())
    }

    /// The descriptor of [`pipeline`](Self::pipeline).
    pub(super) fn descriptor(
        &self,
        index_type: TessellationControlPointIndexType,
    ) -> RenderPipelineDescriptor {
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            self.function.function_descriptor(),
        ));
        descriptor.rasterization_enabled = false;
        let mut vertex = VertexDescriptor::new();
        for (index, step) in [
            (0, VertexStepFunction::PerPatchControlPoint),
            (1, VertexStepFunction::PerPatch),
        ] {
            let mut layout = VertexBufferLayoutDescriptor::new(Some(4));
            layout.step_function = step;
            vertex.layouts.insert(index, layout);
            vertex.attributes.insert(
                index,
                VertexAttributeDescriptor::new(VertexFormat::UInt, 0, index),
            );
        }
        descriptor.vertex_descriptor = Some(vertex);
        let mut tessellation = TessellationDescriptor::default();
        tessellation.max_tessellation_factor = 16;
        tessellation.partition_mode = TessellationPartitionMode::Integer;
        tessellation.factor_step_function = TessellationFactorStepFunction::PerPatch;
        tessellation.control_point_index_type = index_type;
        descriptor.tessellation = Some(tessellation);
        descriptor
    }

    /// A shared buffer of `pad` zero bytes, then `bytes`.
    fn padded(&self, bytes: &[u8], pad: u64) -> Buffer {
        let mut all = vec![0; pad as usize];
        all.extend_from_slice(bytes);
        buffer_with(&self.device, &all).unwrap()
    }

    /// A shared buffer of `pad` zero bytes, then `words`.
    pub(super) fn words(&self, words: &[u32], pad: u64) -> Buffer {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.padded(&bytes, pad)
    }

    /// Record `case` with `pipeline`, its buffers and factors bound, in a command buffer of its
    /// own noted in `writes`, and return what it left in the marks: `Err` for a refused draw,
    /// which ends the encoder unrecorded.
    fn marks(
        &self,
        pipeline: &RenderPipelineState,
        case: &Case,
        writes: &mut NotedWrites,
    ) -> Result<[u32; 4], Error> {
        let points = self.words(&(0..12).collect::<Vec<u32>>(), 0);
        let tags = self.words(&(0..16).map(|p| 0xa0 + p).collect::<Vec<u32>>(), 0);
        let marks = self.words(&[0; 4], 0);
        // Half factors (three edges, one inside) of 1 per patch, but patch 3's 0.
        let factors: Vec<u8> = (0..FACTOR_PATCHES)
            .flat_map(|p| [if p == 3 { 0u16 } else { 0x3c00 }; 4])
            .flat_map(u16::to_le_bytes)
            .collect();
        let factors = buffer_with(&self.device, &factors).unwrap();
        let patch_indices = case
            .patch_indices
            .map(|(indices, offset)| (self.words(indices, offset), offset));
        let control_points = case
            .control_points
            .map(|(indices, offset)| (self.padded(&indices.bytes(), offset), offset));
        let arguments = self.words(&[case.patch_count, 1, case.patch_start, 0], 0);
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(pipeline)?;
        encoder.set_vertex_buffer(Some(&points), 0, 0)?;
        encoder.set_vertex_buffer(Some(&tags), 0, 1)?;
        encoder.set_vertex_buffer(Some(&marks), 0, 2)?;
        encoder.set_tessellation_factor_buffer(Some(&factors), 0, 0)?;
        let (buffer, offset) = match &patch_indices {
            Some((buffer, offset)) => (Some(buffer), *offset),
            None => (None, 0),
        };
        let drawn = match (&control_points, case.indirect) {
            (None, true) => encoder.draw_patches_indirect(3, buffer, offset, &arguments, 0),
            (None, false) => {
                encoder.draw_patches(3, case.patch_start, case.patch_count, buffer, offset, 1, 0)
            }
            (Some((points, at)), true) => {
                encoder.draw_indexed_patches_indirect(3, buffer, offset, points, *at, &arguments, 0)
            }
            (Some((points, at)), false) => encoder.draw_indexed_patches(
                3,
                case.patch_start,
                case.patch_count,
                buffer,
                offset,
                points,
                *at,
                1,
                0,
            ),
        };
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        if drawn.is_ok() {
            writes.drew_patches(&marks, 0..marks.length());
        }
        writes.next_command_buffer();
        drawn?;
        let bytes = read_buffer(&marks, 0, 16).unwrap();
        let mut words = [0; 4];
        for (word, chunk) in words.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
            *word = u32::from_le_bytes(*chunk);
        }
        Ok(words)
    }
}

/// Every case stores the marks Metal stored: a patch index buffer names each drawn patch's
/// control points, attribute and patch index, from its offset and the draw's patch start, direct
/// and indirect, and the factors stay at the patch's position; a control-point index buffer
/// names each control point of the patch its patch index names, `ushort` and `uint`, from its
/// offset, direct and indirect. A device that cannot make the fixture's pipelines refuses them,
/// reported as its refusal; the emulation never does.
#[test]
fn patch_index_buffers_name_patches_as_metal() {
    name_patches_as_metal(Path::Native);
}

/// [`patch_index_buffers_name_patches_as_metal`], emulated.
#[test]
fn emulated_patch_index_buffers_name_patches_as_metal() {
    name_patches_as_metal(Path::Emulated);
}

fn name_patches_as_metal(path: Path) {
    on_path_noted(path, |device, name, writes| {
        let lane = Lane::new(device);
        for (case, metal) in cases() {
            let pipeline = match lane.pipeline(case.index_type()) {
                Ok(pipeline) => pipeline,
                Err(Error::Unsupported(u)) if path == Path::Native => {
                    eprintln!("patch index buffers on {name}: the pipeline is refused, {u:?}");
                    return;
                }
                Err(error) => panic!("the pipeline on {name}: {error}"),
            };
            let marks = lane
                .marks(&pipeline, &case, writes)
                .unwrap_or_else(|e| panic!("{} on {name}: {e}", case.name));
            assert_eq!(
                marks.map(|w| format!("{w:x}")),
                metal.map(|w| format!("{w:x}")),
                "{} on {name}: the marks",
                case.name
            );
        }
    });
}

/// What a draw gives on Metal: the marks it leaves where they are defined (`None` where the
/// patch read past a buffer), or the problem Metal's validation stops it with.
type Expected = Result<[Option<u32>; 4], DrawProblem>;

/// What Metal's validation stops of a control-point index buffer (Metal probes on an M4) is
/// refused, typed, but for an offset that is not a multiple of the index size, which plain Metal
/// reads from the offset rounded down and the runtime likewise: a direct draw's
/// `(patchStart + patchCount) · 3` indices past the buffer (a patch index buffer's
/// patch indices are not checked against it, nor an indirect draw's). A draw of the other kind than
/// the pipeline's, direct or indirect, draws as plain Metal draws it. A draw Metal's validation
/// lets through draws; the marks of patches whose control points lie past the buffer, which Metal
/// reads from memory past it (and the runtime as index 0), are not compared.
#[test]
fn control_point_index_buffers_are_refused_as_metal_refuses_them() {
    refused_as_metal_refuses_them(Path::Native);
}

/// [`control_point_index_buffers_are_refused_as_metal_refuses_them`], emulated.
#[test]
fn emulated_control_point_index_buffers_are_refused_as_metal_refuses_them() {
    refused_as_metal_refuses_them(Path::Emulated);
}

fn refused_as_metal_refuses_them(path: Path) {
    use TessellationControlPointIndexType as T;
    on_path_noted(path, |device, name, writes| {
        let lane = Lane::new(device);
        let short = REVERSED.split_at(5).0;
        let six = REVERSED.split_at(6).0;
        let range = |indices, length| DrawProblem::ControlPointIndexBufferRange {
            offset: 0,
            indices,
            size: 4,
            length,
        };
        // Each draw, its pipeline's index type, and Metal's refusal or the marks compared.
        let rows: [(Case, T, Expected); 12] = [
            // Offsets that are not a multiple of the index size, read from the offset rounded
            // down, direct and indirect.
            (
                Case::indexed("cpi32_offset2", Points::U32(SPLIT32), 2),
                T::UInt32,
                Ok(ROUNDED_DOWN),
            ),
            (
                Case {
                    indirect: true,
                    ..Case::indexed("cpi32_offset2_indirect", Points::U32(SPLIT32), 2)
                },
                T::UInt32,
                Ok(ROUNDED_DOWN),
            ),
            (
                Case::indexed("cpi16_offset1", Points::U16(SPLIT16), 1),
                T::UInt16,
                Ok(ROUNDED_DOWN),
            ),
            (
                Case::indexed("cpi32_short", Points::U32(short), 0),
                T::UInt32,
                Err(range(6, 20)),
            ),
            (
                Case {
                    patch_start: 1,
                    ..Case::indexed("cpi_start1_count2", Points::U32(six), 0)
                },
                T::UInt32,
                Err(range(9, 24)),
            ),
            (
                Case {
                    patch_start: 1,
                    patch_count: 1,
                    ..Case::indexed("cpi_start1_count1", Points::U32(six), 0)
                },
                T::UInt32,
                Ok([Some(0), Some(0), Some(0xa101_0678), Some(0)]),
            ),
            (
                Case {
                    patch_indices: Some((&[3], 0)),
                    patch_count: 1,
                    ..Case::indexed("cpi_pib_id3", Points::U32(six), 0)
                },
                T::UInt32,
                Ok([None; 4]),
            ),
            (
                Case {
                    indirect: true,
                    ..Case::indexed("cpi_indirect_short", Points::U32(short), 0)
                },
                T::UInt32,
                Ok([None, None, None, Some(0xa000_09ab)]),
            ),
            // The other kind of draw than the pipeline's, which Metal's validation stops: an
            // indexed draw on a pipeline without an index type draws the patches' own control
            // points, the buffer unread, and a plain one on a pipeline with a type reads every
            // index as 0 (a Metal probe on an M4, rasterized: what each draws; a probe, a
            // command's, the marks), and their indirect forms likewise (a Metal probe, the marks of
            // both forms alike).
            (
                Case::indexed("indexed_without_type", Points::U32(REVERSED), 0),
                T::None,
                Ok([Some(0xa000_0012), Some(0xa101_0345), Some(0), Some(0)]),
            ),
            (
                Case {
                    patch_count: 1,
                    ..Case::new("plain_with_type")
                },
                T::UInt32,
                Ok([Some(0xa000_0000), Some(0), Some(0), Some(0)]),
            ),
            (
                Case {
                    indirect: true,
                    ..Case::indexed("indexed_without_type_indirect", Points::U32(REVERSED), 0)
                },
                T::None,
                Ok([Some(0xa000_0012), Some(0xa101_0345), Some(0), Some(0)]),
            ),
            (
                Case {
                    indirect: true,
                    patch_count: 1,
                    ..Case::new("plain_with_type_indirect")
                },
                T::UInt32,
                Ok([Some(0xa000_0000), Some(0), Some(0), Some(0)]),
            ),
        ];
        for (case, index_type, expected) in rows {
            let pipeline = match lane.pipeline(index_type) {
                Ok(pipeline) => pipeline,
                Err(Error::Unsupported(u)) if path == Path::Native => {
                    eprintln!("control-point refusals on {name}: the pipeline is refused, {u:?}");
                    return;
                }
                Err(error) => panic!("the pipeline on {name}: {error}"),
            };
            let got = lane.marks(&pipeline, &case, writes);
            match expected {
                Ok(metal) => {
                    let marks = got.unwrap_or_else(|e| panic!("{} on {name}: {e}", case.name));
                    for (k, (mark, metal)) in marks.iter().zip(metal).enumerate() {
                        if let Some(metal) = metal {
                            assert_eq!(*mark, metal, "{} on {name}: mark {k}", case.name);
                        }
                    }
                }
                Err(problem) => {
                    let call = match (case.control_points, case.indirect) {
                        (Some(_), true) => "draw_indexed_patches_indirect",
                        (Some(_), false) => "draw_indexed_patches",
                        (None, true) => "draw_patches_indirect",
                        (None, false) => "draw_patches",
                    };
                    let refused = Error::Invalid(Invalid::new(call, Problem::Draw(problem)));
                    assert_eq!(got, Err(refused), "{} on {name}", case.name);
                }
            }
        }
    });
}
