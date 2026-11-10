//! Device tests of patch draws on the device the cfg selects, under the validation layer: the
//! render encoder's tessellation factor buffer, offset, instance stride and scale, direct and
//! indirect patch draws, and Metal's refusals of them.
//!
//! The draws, as a Metal probe on an M4 drew them: the owned fixture `k7_tess_triangle_domain`,
//! with rasterization disabled in a 1x1 pass without attachments, marks the 65x65 cell of every
//! domain point of the triangle patches it draws, its per-patch grid 64 by 64 (integer
//! partitioning, half factors, a maximum factor of 64). Each case sets the factors and draws; the
//! cells it marked are Metal's. A factor of 1 marks the patch's corners, and 2 and 3 more points,
//! so the marks say which factors each patch read. The refusals are those of another Metal probe on
//! an M4.
//!
//! Each test runs on the device as the cfg opens it, through its tessellation stages ([`Path`]),
//! and on the device made without `tessellationShader`, whose patch draws the runtime emulates
//! (`render/tessellation_scan.rs`): the same marks, and the emulation announced once
//! ([`EMULATED_TESSELLATION`]).

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{
    PrimitiveType, TessellationFactorStepFunction, TessellationPartitionMode, VertexFormat,
    VertexStepFunction,
};
use metal2vulkan::pipeline::{
    RenderPipelineDescriptor, StageSource, TessellationDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan::vk::DriverId;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Degradation, Device, DeviceOptions,
    DrawProblem, Error, Function, Invalid, PipelineOptions, Problem, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState,
};
use sha2::Digest;

/// The kind a device without `tessellationShader` takes for its patch draws, which the runtime
/// emulates (`render/tessellation_scan.rs`).
pub(super) const EMULATED_TESSELLATION: [Degradation; 1] = [Degradation::EmulatedTessellationDraws];

/// Which tessellation a lane test's device draws patches through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Path {
    /// The device as the cfg opens it: its tessellation stages where it has `tessellationShader`,
    /// else its refusal.
    Native,
    /// The device made without `tessellationShader` (and `multiviewTessellationShader`): the
    /// runtime's emulation.
    Emulated,
}

/// The device options of `path`: the emulated path's withholds `tessellationShader` and
/// `multiviewTessellationShader`.
pub(super) fn options_of(path: Path) -> impl FnOnce(&mut DeviceOptions) {
    move |options| {
        if path == Path::Emulated {
            options.withheld_features.extend([
                VkFeature::TessellationShader,
                VkFeature::MultiviewTessellationShader,
            ]);
        }
    }
}

/// `check` on the lane device of `path` and its name; then the validation layer was silent, on
/// the emulated path the device announced the emulation once, and the device's triangle order
/// (`tessellation-triangle-order`) was announced at most once, only on the native path of a
/// device whose tessellator is not Apple's (MoltenVK on an Apple GPU hands the patches to Metal's):
/// the emulation draws Metal's order.
pub(super) fn on_path(path: Path, check: impl FnOnce(Device, &str)) {
    on_path_with(|_| {}, path, check)
}

/// [`on_path`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
pub(super) fn on_path_with(
    adjust: fn(&mut DeviceOptions),
    path: Path,
    check: impl FnOnce(Device, &str),
) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let apple_tessellator;
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options_of(path)(options);
            adjust(options)
        })
        .unwrap();
        let description = &device.properties().description;
        apple_tessellator =
            description.driver_id == Some(DriverId::MOLTENVK) && description.vendor_id == 0x106B;
        let name = device.name().to_owned();
        check(device, &name);
    }
    let notices = messages.take_notices();
    let announced = |kind: Degradation| {
        notices
            .iter()
            .filter(|m| {
                m.degradation
                    .as_ref()
                    .is_some_and(|d| d.degradation == kind)
            })
            .count()
    };
    if path == Path::Emulated {
        let emulated: usize = EMULATED_TESSELLATION.iter().map(|&k| announced(k)).sum();
        assert_eq!(emulated, 1, "the emulation's notice");
    }
    let order = announced(Degradation::TessellationTriangleOrder);
    if path == Path::Emulated || apple_tessellator {
        assert_eq!(order, 0, "the triangle order on {path:?}");
    } else {
        assert!(order <= 1, "the triangle order announced {order} times");
    }
    layer_reported_nothing_at_all(&messages);
}

/// `error`, a refusal of the fixture's pipeline, which only the native path may answer (on a
/// device that cannot make it); the emulation on a lane device never does.
fn refusal_allowed(path: Path, name: &str, error: Error) -> Error {
    assert_eq!(
        path,
        Path::Native,
        "on {name} the emulation refused: {error}"
    );
    error
}

/// The fixture's marks buffer: a word per cell of the 65x65 grid, then three per cell.
pub(super) const MARK_WORDS: usize = 4225 * 4;

/// The vertex buffer index of the per-patch grid; the marks are at buffer 1.
const GRID_INDEX: u32 = 2;

/// Half-float 1, 2 and 3.
pub(super) const ONE: u16 = 0x3c00;
pub(super) const TWO: u16 = 0x4000;
const THREE: u16 = 0x4200;

/// One patch draw: the pipeline's step function and scale, the encoder's factors (each value an
/// 8-byte triangle patch's edges and inside, but for an `inside` that differs), offset, instance
/// stride and scale, and the draw, direct or by indirect arguments (`patchCount`,
/// `instanceCount`, `patchStart`, `baseInstance`).
#[derive(Clone)]
struct Case {
    name: &'static str,
    step: TessellationFactorStepFunction,
    scale_enabled: bool,
    scale: Option<f32>,
    factors: &'static [u16],
    inside: Option<u16>,
    offset: u64,
    stride: u64,
    patch_start: u32,
    patch_count: u32,
    instance_count: u32,
    base_instance: u32,
    indirect: Option<[u32; 4]>,
    /// The control points the draw names (the function states 1).
    points: u32,
    /// The factor buffer's `half`s, then 16 bytes of zeros, in place of `factors` and `inside`.
    halves: Option<&'static [u16]>,
}

impl Case {
    /// One patch of factors 1, read per patch.
    const fn new(name: &'static str) -> Case {
        Case {
            name,
            step: TessellationFactorStepFunction::PerPatch,
            scale_enabled: false,
            scale: None,
            factors: &[ONE],
            inside: None,
            offset: 0,
            stride: 0,
            patch_start: 0,
            patch_count: 1,
            instance_count: 1,
            base_instance: 0,
            indirect: None,
            points: 1,
            halves: None,
        }
    }
}

/// The lane's device, its queue, the fixture's function, and a plain `void` vertex function
/// (`icb_render_replay_template`, for the draw-kind refusals).
pub(super) struct Lane {
    device: Device,
    pub(super) queue: CommandQueue,
    domain: Function,
    plain: Function,
}

impl Lane {
    pub(super) fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("k7_tess_triangle_domain.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            domain: library
                .new_function_with_name("k7_tess_triangle_domain")
                .unwrap(),
            // A `void` vertex function: Metal refuses a state with rasterization disabled whose
            // vertex function returns a value.
            plain: device
                .new_library_with_air(
                    &llvm_assemble(&engine_dir().join("icb_render_replay_template.ll")).unwrap(),
                )
                .unwrap()
                .new_function_with_name("icb_render_replay_template")
                .unwrap(),
            device,
        }
    }

    /// The fixture's pipeline, rasterization disabled, its factors read by `step`, scaled where
    /// `scale_enabled`.
    fn pipeline(
        &self,
        step: TessellationFactorStepFunction,
        scale_enabled: bool,
    ) -> Result<RenderPipelineState, Error> {
        self.device.new_render_pipeline_state(
            &self.descriptor(step, scale_enabled),
            &PipelineOptions::default(),
        )
    }

    /// The descriptor of [`Lane::pipeline`].
    pub(super) fn descriptor(
        &self,
        step: TessellationFactorStepFunction,
        scale_enabled: bool,
    ) -> RenderPipelineDescriptor {
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(self.domain.function_descriptor()));
        descriptor.rasterization_enabled = false;
        let mut vertex = VertexDescriptor::new();
        let mut layout = VertexBufferLayoutDescriptor::new(Some(16));
        layout.step_function = VertexStepFunction::PerPatch;
        vertex.layouts.insert(GRID_INDEX, layout);
        vertex.attributes.insert(
            0,
            VertexAttributeDescriptor::new(VertexFormat::Float4, 0, GRID_INDEX),
        );
        descriptor.vertex_descriptor = Some(vertex);
        let mut tessellation = TessellationDescriptor::default();
        tessellation.max_tessellation_factor = 64;
        tessellation.partition_mode = TessellationPartitionMode::Integer;
        tessellation.factor_step_function = step;
        tessellation.factor_scale_enabled = scale_enabled;
        descriptor.tessellation = Some(tessellation);
        descriptor
    }

    /// A pipeline of the plain vertex function, rasterization disabled.
    fn plain_pipeline(&self) -> RenderPipelineState {
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(self.plain.function_descriptor()));
        descriptor.rasterization_enabled = false;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// A shared buffer of `bytes`.
    pub(super) fn buffer(&self, bytes: &[u8]) -> Buffer {
        buffer_with(&self.device, bytes).unwrap()
    }

    /// The bytes of triangle factors `values`, each value every edge and the inside, then 16
    /// bytes of zeros.
    pub(super) fn factors(&self, values: &[u16]) -> Buffer {
        self.factors_inside(values, None)
    }

    /// As [`Lane::factors`], but each patch's inside factor `inside` where given.
    fn factors_inside(&self, values: &[u16], inside: Option<u16>) -> Buffer {
        let mut bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| [v, v, v, inside.unwrap_or(v)])
            .flat_map(u16::to_le_bytes)
            .collect();
        bytes.extend([0; 16]);
        self.buffer(&bytes)
    }

    /// Open a 1x1 pass without attachments on a new command buffer, set `pipeline`, bind
    /// `marks` and four patches' grids, and run `encode`; then end, commit and wait. The command
    /// buffer comes back completed.
    pub(super) fn pass(
        &self,
        pipeline: &RenderPipelineState,
        marks: &Buffer,
        encode: impl FnOnce(&mut RenderCommandEncoder) -> Result<(), Error>,
    ) -> Result<CommandBuffer, Error> {
        let grid: Vec<u8> = [64.0f32, 64.0, 0.0, 0.0]
            .repeat(4)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let grid = self.buffer(&grid);
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(pipeline)?;
        encoder.set_vertex_buffer(Some(marks), 0, 1)?;
        encoder.set_vertex_buffer(Some(&grid), 0, GRID_INDEX)?;
        encode(&mut encoder)?;
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(command_buffer)
    }

    /// The cells `case` marked.
    fn marked(&self, case: &Case) -> Result<Vec<u32>, Error> {
        let pipeline = self.pipeline(case.step, case.scale_enabled)?;
        let marks = self.buffer(&vec![0; MARK_WORDS * 4]);
        let factors = match case.halves {
            Some(halves) => {
                let mut bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
                bytes.extend([0; 16]);
                self.buffer(&bytes)
            }
            None => self.factors_inside(case.factors, case.inside),
        };
        self.pass(&pipeline, &marks, |encoder| {
            encoder.set_tessellation_factor_buffer(Some(&factors), case.offset, case.stride)?;
            if let Some(scale) = case.scale {
                encoder.set_tessellation_factor_scale(scale);
            }
            match case.indirect {
                Some(words) => {
                    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let arguments = self.buffer(&bytes);
                    encoder.draw_patches_indirect(case.points, None, 0, &arguments, 0)
                }
                None => encoder.draw_patches(
                    case.points,
                    case.patch_start,
                    case.patch_count,
                    None,
                    0,
                    case.instance_count,
                    case.base_instance,
                ),
            }
        })?;
        Ok(cells(&marks))
    }
}

/// The cells `marks` holds a mark in.
pub(super) fn cells(marks: &Buffer) -> Vec<u32> {
    read_buffer(marks, 0, 4225 * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .filter(|(_, word)| **word != [0; 4])
        .map(|(cell, _)| cell as u32)
        .collect()
}

/// The cells of a patch of factors 1: its three corners (a Metal probe on an M4).
pub(super) const CORNERS: &[u32] = &[0, 64, 4160];

/// The cells of a patch of factors 2: the corners, the edges' midpoints and the centre.
pub(super) const TWOS: &[u32] = &[0, 32, 64, 1386, 2080, 2112, 4160];

/// The cells of a patch of factors 3.
const THREES: &[u32] = &[0, 21, 43, 64, 924, 946, 1365, 1408, 2354, 2795, 2816, 4160];

/// The cells of a patch of the maximum factor, 64, as Metal marked it: each cell's index,
/// little-endian, ascending (their probe's grid is the fixture's, its digests of factors 1, 2 and
/// 3 those of `CORNERS`, `TWOS` and `THREES`).
const MAXIMUM_CELLS: &[u8] = include_bytes!("../../fixtures/public/patches/maximum_cells.bin");

/// The SHA-256 of [`MAXIMUM_CELLS`], and the count and FNV-1a of them the probe printed.
const MAXIMUM_SHA: &str = "bc3b3cd507303a0b45a8e671e18fdbd886bee90d4a393fd65380cad08d863fa8";
const MAXIMUM_PROBE: (usize, u32) = (1815, 0xb267_fce1);

/// The cells of a patch whose inside factor is `+∞` beside edge factors of 1, as Metal marked it,
/// laid out as [`MAXIMUM_CELLS`].
const INSIDE_CELLS: &[u8] = include_bytes!("../../fixtures/public/patches/inside_cells.bin");

/// The SHA-256 of [`INSIDE_CELLS`], and the count and FNV-1a of them the probe printed.
const INSIDE_SHA: &str = "f93df24eb5f0bdaea72fc0173bcb4f14c50bad7de98ad1bc536fd5c3f221204d";
const INSIDE_PROBE: (usize, u32) = (1689, 0x540d_bf35);

/// The cells `bytes` hold.
fn cells_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|cell| u32::from_le_bytes(*cell))
        .collect()
}

/// The cells Metal marked at the maximum factor, 64 ([`MAXIMUM_CELLS`]).
pub(super) fn maximum() -> Vec<u32> {
    cells_of(MAXIMUM_CELLS)
}

/// How `cells` differ from the `metal` a probe recorded, or `None` where they are alike: the
/// first cell they part at, with both counts.
pub(super) fn unlike(cells: &[u32], metal: &[u32]) -> Option<String> {
    if cells == metal {
        return None;
    }
    let counts = format!("{} cells marked, Metal's {}", cells.len(), metal.len());
    Some(
        match cells.iter().zip(metal).position(|(got, want)| got != want) {
            Some(i) => format!(
                "{counts}: cell {i} of them is {}, Metal's {}",
                cells[i], metal[i]
            ),
            None => format!("{counts}, the shorter their first"),
        },
    )
}

/// The count of `cells` and the FNV-1a hash of their indices' little-endian bytes, as our probes
/// printed them.
fn probe_digest(cells: &[u32]) -> (usize, u32) {
    let hash = cells
        .iter()
        .flat_map(|cell| cell.to_le_bytes())
        .fold(0x811c_9dc5u32, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
        });
    (cells.len(), hash)
}

/// The recorded cells are the probes': each list's SHA-256, and the count and FNV-1a hash the
/// probe printed of its own marks.
#[test]
fn the_recorded_cells_are_the_probes() {
    for (what, bytes, sha, probe) in [
        (
            "the maximum factor",
            MAXIMUM_CELLS,
            MAXIMUM_SHA,
            MAXIMUM_PROBE,
        ),
        ("an infinite inside", INSIDE_CELLS, INSIDE_SHA, INSIDE_PROBE),
    ] {
        let cells = cells_of(bytes);
        assert_eq!(
            crate::hash::hex_encode(sha2::Sha256::digest(bytes)),
            sha,
            "{what}"
        );
        assert_eq!(probe_digest(&cells), probe, "{what}");
    }
}

/// Each case and the cells Metal marked (a Metal probe on an M4 with macOS 27.0.1, with and without
/// Metal's API validation). A pipeline that enables the scale culls every patch until a scale is
/// set; one that does not ignores the scale.
fn cases() -> Vec<(Case, &'static [u32])> {
    use TessellationFactorStepFunction as S;
    let two_values: &'static [u16] = &[ONE, TWO];
    vec![
        (Case::new("one"), CORNERS),
        (
            Case {
                factors: &[TWO],
                ..Case::new("two")
            },
            TWOS,
        ),
        (
            Case {
                factors: &[THREE],
                ..Case::new("three")
            },
            THREES,
        ),
        (
            Case {
                factors: two_values,
                offset: 8,
                ..Case::new("offset_8")
            },
            TWOS,
        ),
        (
            Case {
                factors: two_values,
                patch_start: 1,
                ..Case::new("patch_start_1")
            },
            TWOS,
        ),
        (
            Case {
                factors: two_values,
                patch_count: 2,
                ..Case::new("two_patches")
            },
            TWOS,
        ),
        (
            Case {
                step: S::Constant,
                factors: &[TWO, ONE],
                patch_count: 2,
                ..Case::new("constant_two_patches")
            },
            TWOS,
        ),
        (
            Case {
                step: S::PerInstance,
                factors: two_values,
                stride: 8,
                base_instance: 1,
                ..Case::new("instance_stride_8_base_1")
            },
            TWOS,
        ),
        (
            Case {
                step: S::PerInstance,
                factors: two_values,
                stride: 8,
                instance_count: 2,
                ..Case::new("instance_stride_8_two")
            },
            TWOS,
        ),
        (
            Case {
                step: S::PerPatchAndPerInstance,
                factors: &[ONE, ONE, THREE],
                stride: 16,
                base_instance: 1,
                ..Case::new("patch_and_instance")
            },
            THREES,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(2.0),
                ..Case::new("scale_2")
            },
            TWOS,
        ),
        (
            Case {
                scale_enabled: true,
                ..Case::new("scale_default")
            },
            &[],
        ),
        (
            Case {
                scale: Some(2.0),
                ..Case::new("scale_ignored")
            },
            CORNERS,
        ),
        // The scale rounded to `half`: 1.3 and 2.0005 are 2-ish `half`s, 10^-6 and 2^-24 subnormal
        // ones, and 2^-30 is 0.
        (
            Case {
                scale_enabled: true,
                scale: Some(1.3),
                ..Case::new("scale_1.3")
            },
            TWOS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(2.0005),
                ..Case::new("scale_2.0005")
            },
            TWOS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(1.0e-6),
                ..Case::new("scale_1e-6")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(2f32.powi(-24)),
                ..Case::new("scale_2^-24")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(2f32.powi(-30)),
                ..Case::new("scale_2^-30")
            },
            &[],
        ),
        // Scales Metal's validation stops at the setter, which plain Metal takes (a probe `ts_*`,
        // an M4 with macOS 27.0.1): NaN, `-∞` and a subnormal `float` of either sign (0 as a
        // `half`) cull every patch of a pipeline that enables the scale; each draws as 1 does on
        // one that does not.
        (
            Case {
                scale_enabled: true,
                scale: Some(f32::NAN),
                ..Case::new("scale_nan")
            },
            &[],
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(f32::NEG_INFINITY),
                ..Case::new("scale_negative_infinity")
            },
            &[],
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(f32::from_bits(0x0000_0400)),
                ..Case::new("scale_subnormal")
            },
            &[],
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(f32::from_bits(0x8000_0400)),
                ..Case::new("scale_negative_subnormal")
            },
            &[],
        ),
        (
            Case {
                scale: Some(f32::NAN),
                ..Case::new("scale_nan_ignored")
            },
            CORNERS,
        ),
        (
            Case {
                scale: Some(f32::INFINITY),
                ..Case::new("scale_infinity_ignored")
            },
            CORNERS,
        ),
        // Factors at the edges of Metal's rules: a subnormal edge factor draws, unscaled or scaled;
        // a product below `half`'s least subnormal does not cull; an inside factor that is NaN or 0
        // takes the lower end, which beside edges of 2 Vulkan's and Metal's tessellators both draw
        // as 2.
        (
            Case {
                scale_enabled: true,
                scale: Some(1.0),
                factors: &[0x0001],
                ..Case::new("subnormal_edges")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(2.0),
                factors: &[0x0001],
                ..Case::new("subnormal_edges_by_2")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(65504.0),
                factors: &[0x0001],
                ..Case::new("subnormal_edges_by_65504")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(2f32.powi(-24)),
                factors: &[0x3800],
                ..Case::new("half_by_2^-24")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(2f32.powi(-14)),
                factors: &[0x0400],
                ..Case::new("2^-14_by_2^-14")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(1.0),
                inside: Some(0x7e00),
                ..Case::new("inside_nan")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(1.0),
                inside: Some(0),
                ..Case::new("inside_zero")
            },
            CORNERS,
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(2.0),
                inside: Some(0x7e00),
                ..Case::new("inside_nan_by_2")
            },
            TWOS,
        ),
        // Factors 0x47f3 by the `half` 0x3007: 1.000445 in `float`, two segments, where the
        // product rounded to `half` would be 1.0, one: Metal does not
        // round it.
        (
            Case {
                scale_enabled: true,
                scale: Some(f32::from_bits(0x3e00_e000)),
                factors: &[0x47f3],
                ..Case::new("scale_product_unrounded")
            },
            TWOS,
        ),
        (
            Case {
                factors: &[TWO],
                indirect: Some([1, 1, 0, 0]),
                ..Case::new("indirect")
            },
            TWOS,
        ),
        (
            Case {
                factors: two_values,
                indirect: Some([1, 1, 1, 0]),
                ..Case::new("indirect_patch_start_1")
            },
            TWOS,
        ),
        (
            Case {
                step: S::PerInstance,
                factors: two_values,
                stride: 8,
                indirect: Some([1, 1, 0, 1]),
                ..Case::new("indirect_base_instance_1")
            },
            TWOS,
        ),
        (
            Case {
                factors: &[TWO],
                indirect: Some([0, 1, 0, 0]),
                ..Case::new("indirect_zero")
            },
            &[],
        ),
        // What Metal's validation stops and plain Metal draws (a Metal probe on an M4): other
        // control points than the function's draw with its own, direct and indirect; no instances
        // or patches draw nothing; a scale of 0 or -1 culls where the pipeline enables it and is
        // ignored where it does not.
        (
            Case {
                factors: &[TWO],
                points: 0,
                ..Case::new("control_points_0")
            },
            TWOS,
        ),
        (
            Case {
                factors: &[TWO],
                points: 4,
                ..Case::new("control_points_4")
            },
            TWOS,
        ),
        (
            Case {
                factors: &[TWO],
                points: 3,
                indirect: Some([1, 1, 0, 0]),
                ..Case::new("indirect_control_points_3")
            },
            TWOS,
        ),
        (
            Case {
                instance_count: 0,
                ..Case::new("zero_instances")
            },
            &[],
        ),
        (
            Case {
                patch_count: 0,
                ..Case::new("zero_patches")
            },
            &[],
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(0.0),
                ..Case::new("scale_zero")
            },
            &[],
        ),
        (
            Case {
                scale_enabled: true,
                scale: Some(-1.0),
                ..Case::new("scale_negative")
            },
            &[],
        ),
        (
            Case {
                scale: Some(0.0),
                ..Case::new("scale_zero_without_enable")
            },
            CORNERS,
        ),
        // A factor offset of 2, which Metal's validation stops, read from that byte as plain Metal
        // reads it (Metal probe case `factor_offset_2` on an M4: the factors 2 bytes past a 2-byte
        // pad drew): factors of 2 after a `half` of 0, which a read from the offset rounded down to
        // 0 would take as an edge, culling.
        (
            Case {
                halves: Some(&[0, TWO, TWO, TWO, TWO]),
                offset: 2,
                ..Case::new("factor_offset_2")
            },
            TWOS,
        ),
        // Instance strides 2 past a multiple of 4, which Metal's validation stops, stepped by the
        // multiple below as plain Metal steps them (Metal probes, two instances, the encoder's and
        // a command's alike: 2 and 14 drew nothing, 6 and 10 drew instance 1). Instance 0 culls on
        // its edge of 0; instance 1 reads at the rounded stride, where a read at the stride itself
        // would draw for 2 and 14 and cull on the zeros past the `half`s for 6 and 10.
        (
            Case {
                step: S::PerInstance,
                halves: Some(&[0, TWO, TWO, TWO, TWO]),
                stride: 2,
                instance_count: 2,
                ..Case::new("factor_stride_2")
            },
            &[],
        ),
        (
            Case {
                step: S::PerInstance,
                halves: Some(&[0, TWO, TWO, TWO, TWO, TWO]),
                stride: 6,
                instance_count: 2,
                ..Case::new("factor_stride_6")
            },
            TWOS,
        ),
        (
            Case {
                step: S::PerInstance,
                halves: Some(&[0, TWO, TWO, TWO, TWO, TWO, TWO, TWO]),
                stride: 10,
                instance_count: 2,
                ..Case::new("factor_stride_10")
            },
            TWOS,
        ),
        (
            Case {
                step: S::PerInstance,
                halves: Some(&[0, TWO, TWO, TWO, TWO, TWO, 0, TWO, TWO, TWO, TWO]),
                stride: 14,
                instance_count: 2,
                ..Case::new("factor_stride_14")
            },
            &[],
        ),
        // Odd factor offsets, read from the byte below as plain Metal reads them (the encoder's and
        // a command's alike: 1 culled on the `half` of 0 at byte 0, 3 drew from byte 2), and odd
        // strides, stepped by the multiple of 4 below (13 and 15 culled where 12 does, 17 and 19
        // where 16 does): a read at 3 itself would straddle two `half`s; one at stride 7 itself, or
        // at 6, would reach the zeros past the `half`s, and one at 15 or 14 the `half`s past the 0.
        (
            Case {
                halves: Some(&[0, TWO, TWO, TWO, TWO]),
                offset: 1,
                ..Case::new("factor_offset_1")
            },
            &[],
        ),
        (
            Case {
                halves: Some(&[0, TWO, TWO, TWO, TWO]),
                offset: 3,
                ..Case::new("factor_offset_3")
            },
            TWOS,
        ),
        (
            Case {
                step: S::PerInstance,
                halves: Some(&[0, TWO, TWO, TWO, TWO, TWO]),
                stride: 7,
                instance_count: 2,
                ..Case::new("factor_stride_7")
            },
            TWOS,
        ),
        (
            Case {
                step: S::PerInstance,
                halves: Some(&[0, TWO, TWO, TWO, TWO, TWO, 0, TWO, TWO, TWO, TWO]),
                stride: 15,
                instance_count: 2,
                ..Case::new("factor_stride_15")
            },
            &[],
        ),
    ]
}

/// Every case marks the cells Metal marked: the factors each patch read, at the encoder's offset,
/// by the pipeline's step function from the draw's patch and instance (the base instance
/// included), scaled where the pipeline enables it, and none for indirect arguments of no
/// patches. A device that cannot make the fixture's pipeline refuses it, reported as its refusal.
#[test]
fn patch_draws_read_their_factors_as_metal() {
    read_their_factors_as_metal(Path::Native);
}

/// [`patch_draws_read_their_factors_as_metal`], emulated: no case refused.
#[test]
fn emulated_patch_draws_read_their_factors_as_metal() {
    read_their_factors_as_metal(Path::Emulated);
}

fn read_their_factors_as_metal(path: Path) {
    on_path(path, |device, name| {
        let lane = Lane::new(device);
        let mut answers = Vec::new();
        for (case, metal) in cases() {
            match lane.marked(&case) {
                Ok(cells) => {
                    assert_eq!(cells, metal, "{} on {name}: the marked cells", case.name);
                    answers.push(format!("{} Metal's", case.name));
                }
                Err(Error::Unsupported(u)) => {
                    let u = refusal_allowed(path, name, Error::Unsupported(u));
                    answers.push(format!("{} refused, {u:?}", case.name));
                }
                Err(error) => panic!("{} on {name}: {error}", case.name),
            }
        }
        eprintln!("patch draws on {name} ({path:?}): {}", answers.join("; "));
    });
}

/// A scale that is infinite as a `half` (70000, as the largest `half`, 65504, and 64 do) takes
/// the factors to the maximum, 64, as Metal's did ([`MAXIMUM_CELLS`]), and so does the `float`
/// `+∞`, which Metal's validation stops at the setter and plain Metal takes (a probe
/// `ts_inf_enabled`: the maximum factor's vertices). A device that cannot make the fixture's
/// pipeline refuses it, reported as its refusal.
#[test]
fn a_scale_infinite_as_a_half_draws_the_maximum_factor() {
    scale_infinite_draws_the_maximum(Path::Native);
}

/// [`a_scale_infinite_as_a_half_draws_the_maximum_factor`], emulated.
#[test]
fn emulated_a_scale_infinite_as_a_half_draws_the_maximum_factor() {
    scale_infinite_draws_the_maximum(Path::Emulated);
}

fn scale_infinite_draws_the_maximum(path: Path) {
    on_path(path, |device, name| {
        let lane = Lane::new(device);
        for scale in [64.0, 65504.0, 70000.0, f32::INFINITY] {
            let case = Case {
                scale_enabled: true,
                scale: Some(scale),
                ..Case::new("scale_maximum")
            };
            match lane.marked(&case) {
                Ok(cells) => {
                    if let Some(unlike) = unlike(&cells, &maximum()) {
                        panic!("scale {scale} on {name}: {unlike}");
                    }
                }
                Err(Error::Unsupported(u)) => {
                    let u = refusal_allowed(path, name, Error::Unsupported(u));
                    eprintln!("the maximum scale on {name}: refused, {u:?}");
                    return;
                }
                Err(error) => panic!("scale {scale} on {name}: {error}"),
            }
        }
    });
}

/// An inside factor of `+∞` beside edge factors of 1 takes the maximum, 64, as Metal's did
/// ([`INSIDE_CELLS`]). A device that cannot make the fixture's pipeline refuses
/// it, reported as its refusal.
#[test]
fn an_infinite_inside_factor_draws_the_maximum_inside() {
    infinite_inside_draws_the_maximum(Path::Native);
}

/// [`an_infinite_inside_factor_draws_the_maximum_inside`], emulated.
#[test]
fn emulated_an_infinite_inside_factor_draws_the_maximum_inside() {
    infinite_inside_draws_the_maximum(Path::Emulated);
}

fn infinite_inside_draws_the_maximum(path: Path) {
    on_path(path, |device, name| {
        let lane = Lane::new(device);
        let case = Case {
            scale_enabled: true,
            scale: Some(1.0),
            inside: Some(0x7c00),
            ..Case::new("inside_infinite")
        };
        match lane.marked(&case) {
            Ok(cells) => {
                if let Some(unlike) = unlike(&cells, &cells_of(INSIDE_CELLS)) {
                    panic!("on {name}: {unlike}");
                }
            }
            Err(Error::Unsupported(u)) => {
                let u = refusal_allowed(path, name, Error::Unsupported(u));
                eprintln!("the infinite inside on {name}: refused, {u:?}")
            }
            Err(error) => panic!("on {name}: {error}"),
        }
    });
}

/// `Invalid` as Metal's validation stops the call: `problem` of `call`.
fn invalid(call: &'static str, problem: DrawProblem) -> Result<(), Error> {
    Err(Error::Invalid(Invalid::new(call, Problem::Draw(problem))))
}

/// What Metal's validation stops and plain Metal runs draws as plain Metal does (the cases' table
/// pins what each draws); the rest is refused, typed, and a refused call changes nothing (a Metal
/// probe on an M4 with macOS 27.0.1). The setters take any factor offset and instance stride
/// (`patch_draws_read_their_factors_as_metal` draws them) and every scale, which this pipeline, not
/// enabling it, ignores; a patch draw is refused without a factor buffer, with a plain pipeline,
/// with indirect arguments past the buffer, and a direct draw's patch indices past the buffer
/// (Metal probes on an M4); a draw with other control points than the function's draws with its
/// own, one of no patches or instances draws nothing, as does a draw of primitives with the
/// tessellation pipeline, and indirect arguments and a patch index buffer at an offset that is not
/// a multiple of 4 are read from the offset rounded down. Each pass draws only patch 0 of factors
/// 2, so it marks what that patch marks.
#[test]
fn patch_draws_answer_as_plain_metal_and_refuse_the_rest() {
    answer_as_plain_metal(Path::Native);
}

/// [`patch_draws_answer_as_plain_metal_and_refuse_the_rest`], emulated.
#[test]
fn emulated_patch_draws_answer_as_plain_metal_and_refuse_the_rest() {
    answer_as_plain_metal(Path::Emulated);
}

fn answer_as_plain_metal(path: Path) {
    on_path(path, |device, name| {
        let lane = Lane::new(device);
        let pipeline = match lane.pipeline(TessellationFactorStepFunction::PerPatch, false) {
            Ok(pipeline) => pipeline,
            Err(Error::Unsupported(u)) => {
                let u = refusal_allowed(path, name, Error::Unsupported(u));
                eprintln!("patch draw refusals on {name}: the pipeline is refused, {u:?}");
                return;
            }
            Err(error) => panic!("the pipeline on {name}: {error}"),
        };
        let plain = lane.plain_pipeline();
        let factors = lane.factors(&[TWO]);
        let arguments = lane.buffer(&[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let index = lane.buffer(&[0; 8]);
        let draw =
            |e: &mut RenderCommandEncoder, points| e.draw_patches(points, 0, 1, None, 0, 1, 0);
        let marks = lane.buffer(&vec![0; MARK_WORDS * 4]);
        lane.pass(&pipeline, &marks, |e| {
            let patches = "draw_patches";
            let indirect = "draw_patches_indirect";
            assert_eq!(
                draw(e, 1),
                invalid(patches, DrawProblem::NoTessellationFactorBuffer)
            );
            assert_eq!(
                e.draw_patches_indirect(1, None, 0, &arguments, 0),
                invalid(indirect, DrawProblem::NoTessellationFactorBuffer)
            );
            e.set_tessellation_factor_buffer(Some(&factors), 0, 0)?;
            // Taken, and ignored by this pipeline, which does not enable the scale.
            for taken in [
                0.0,
                -0.0,
                -1.0,
                f32::MIN_POSITIVE / 2.0,
                f32::INFINITY,
                f32::NAN,
            ] {
                e.set_tessellation_factor_scale(taken);
            }
            for points in [0, 3] {
                assert_eq!(draw(e, points), Ok(()));
                assert_eq!(
                    e.draw_patches_indirect(points, None, 0, &arguments, 0),
                    Ok(())
                );
            }
            assert_eq!(e.draw_patches(1, 0, 1, None, 0, 0, 0), Ok(()));
            assert_eq!(e.draw_patches(1, 0, 0, None, 0, 1, 0), Ok(()));
            assert_eq!(e.draw_patches_indirect(1, None, 0, &arguments, 2), Ok(()));
            assert_eq!(
                e.draw_patches_indirect(1, None, 0, &arguments, 4),
                invalid(indirect, DrawProblem::IndirectBufferRange)
            );
            assert_eq!(e.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0), Ok(()));
            // A patch index buffer's offset off a multiple of 4 (read from 0, patch 0), and a
            // direct draw's indices past the buffer (Metal probes on an M4).
            assert_eq!(e.draw_patches(1, 0, 1, Some(&index), 2, 1, 0), Ok(()));
            assert_eq!(
                e.draw_patches_indirect(1, Some(&index), 2, &arguments, 0),
                Ok(())
            );
            assert_eq!(
                e.draw_patches(1, 1, 2, Some(&index), 0, 1, 0),
                invalid(
                    patches,
                    DrawProblem::PatchIndexBufferRange {
                        offset: 0,
                        indices: 3,
                        length: 8
                    }
                )
            );
            // The factors stayed 2 at offset 0; patch 0 drawn through its index.
            e.draw_patches(1, 0, 1, Some(&index), 4, 1, 0)?;
            draw(e, 1)
        })
        .unwrap();
        assert_eq!(cells(&marks), TWOS, "after the refusals on {name}");
        let marks = lane.buffer(&vec![0; MARK_WORDS * 4]);
        lane.pass(&plain, &marks, |e| {
            e.set_tessellation_factor_buffer(Some(&factors), 0, 0)?;
            assert_eq!(draw(e, 1), invalid("draw_patches", DrawProblem::DrawKind));
            Ok(())
        })
        .unwrap();
    });
}
