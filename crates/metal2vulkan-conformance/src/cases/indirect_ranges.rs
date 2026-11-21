//! What an execution of an indirect command buffer over a range **the GPU wrote** draws
//! (`executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`).
//!
//! The range is read when the execution runs, not when it is encoded, so each row writes it on the
//! GPU in the execution's own command buffer, over a stale range the host wrote first that would
//! draw every command: a side that read the range before the GPU wrote it draws all eight. The
//! commands are the one shape both sides build ([`EncodedRange`]): indexed triangle draws that
//! inherit the pipeline state and bind their own buffers, encoded by the owned fixture's
//! `icb_indexed_16` kernel through the buffer's `gpuResourceID`. Command `i` draws rows `2i` and
//! `2i + 1` of an 8 by 16 `RGBA32Uint` target, each texel its mark, the provoking vertex's
//! `vid`, its `iid` and its tint, so the target names exactly which commands drew.
//!
//! | Row | The range, and what writes it |
//! |---|---|
//! | [`AT_ZERO`] | commands 0 to 2, a blit's copy before the pass |
//! | [`PAST_ZERO`] | commands 5 and 6, a blit's copy before the pass |
//! | [`DRAWN`] | commands 2 to 5, an earlier draw of the same pass, behind a memory barrier |
//!
//! Every row is bounded by construction: eight commands of two triangles each, one target of 8
//! by 16, one kernel thread a command, and no function holds a loop.

use super::indirect_commands::{fixture_inputs, shaders_of};
use crate::case::{call, made, Abort, Case, EncodedRange, Failure, Input, MetalSource, RangeWrite};
use crate::case::{Side, TARGET_TEXEL};
use crate::fixture::Fixture;
use crate::recording::ShaderRef;
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites};
use metal2vulkan::metal::PixelFormat;

/// The owned fixture whose kernel encodes the commands and whose functions they draw with.
const FIXTURE: &str = "icb_merged_indexed";

/// The kernel that encodes command `tid`, and the state's two functions.
const KERNEL: &str = "icb_indexed_16";
const VERTEX: &str = "icb_indexed_vertex";
const FRAGMENT: &str = "icb_indexed_fragment";

/// The owned fixture whose rasterization-disabled vertex function writes
/// `params[0] * (vid + 1) + params[1]` to `out[vid]` (instance 0): the earlier draw that writes
/// [`DRAWN`]'s range.
const WRITER: &str = "icb_render_replay_template";

/// The commands the buffer holds.
const COMMANDS: u32 = 8;

/// The target's extent: each command's two rows.
const SIZE: [u32; 2] = [8, 2 * COMMANDS];

/// The offset of the range in its buffer: past a first range-sized word pair, so a side that
/// read the range at offset 0 reads the filler.
const OFFSET: u64 = 8;

/// The words the range buffer holds before the GPU writes the range: a filler pair, then a stale
/// range of every command.
const STALE: [u32; 4] = [0xeeee_eeee, 0xeeee_eeee, 0, COMMANDS];

/// Each command's mark and tint: `MARK + i` and `TINT + i`.
const MARK: u32 = 0x100;
const TINT: u32 = 0x200;

/// The rows.
pub const AT_ZERO: &str = "indirect-range-gpu-written-at-zero";
pub const PAST_ZERO: &str = "indirect-range-gpu-written-past-zero";
pub const DRAWN: &str = "indirect-range-written-by-an-earlier-draw";

/// One row: its name, the range the GPU writes (location, length), and what writes it.
pub struct GpuRange {
    name: &'static str,
    range: [u32; 2],
    drawn: bool,
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The kernel's buffers 1 to 6, by name: each command's draw (first index, index count,
/// instances, base vertex, base instance), the marks, the frame (the target's extent and no
/// instance step), the pixel corners of each command's quad, the tints, and two copies of a
/// quad's six 16-bit indices, command `i` drawing from copy `i % 2`.
fn kernel_buffers() -> [(&'static str, Vec<u8>); 6] {
    let draws: Vec<u32> = (0..COMMANDS)
        .flat_map(|i| [6 * (i % 2), 6, 1, 4 * i, i])
        .collect();
    let xy: Vec<f32> = (0..COMMANDS)
        .flat_map(|i| {
            let (top, bottom) = ((2 * i) as f32, (2 * i + 2) as f32);
            [0.0, top, 8.0, top, 0.0, bottom, 8.0, bottom]
        })
        .collect();
    let marks: Vec<u32> = (0..COMMANDS).map(|i| MARK + i).collect();
    let tints: Vec<u32> = (0..COMMANDS).map(|i| TINT + i).collect();
    let indices: Vec<u8> = [0u16, 1, 2, 2, 1, 3]
        .repeat(2)
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    [
        ("buffer:draws", words(&draws)),
        ("buffer:marks", words(&marks)),
        (
            "buffer:frame",
            floats(&[SIZE[0] as f32, SIZE[1] as f32, 0.0, 0.0]),
        ),
        ("buffer:xy", floats(&xy)),
        ("buffer:tints", words(&tints)),
        ("buffer:indices", indices),
    ]
}

impl GpuRange {
    /// The bytes of the buffer the range is written from: a blit's source holds the range itself;
    /// the writer's `params` the pair whose two vertices write the location and the length,
    /// `params[0] * (vid + 1) + params[1]` in wrapping 32-bit arithmetic.
    fn source(&self) -> Vec<u8> {
        let [location, length] = self.range;
        match self.drawn {
            false => words(&[location, length]),
            true => words(&[
                length.wrapping_sub(location),
                (2 * location).wrapping_sub(length),
            ]),
        }
    }

    fn source_name(&self) -> &'static str {
        match self.drawn {
            false => "buffer:copied-range",
            true => "buffer:writer-params",
        }
    }
}

/// The layout the target is read under: every texel of level 0, four 32-bit words each, so a
/// diff names the row (the command) and the word.
fn target_layout() -> ByteLayout {
    let row_pitch = SIZE[0] * TARGET_TEXEL;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIZE[0], SIZE[1], 1],
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: row_pitch / 4,
        row_pitch,
        rows: SIZE[1],
        image_pitch: row_pitch * SIZE[1],
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The library of `fixture`, made on the side.
fn library(
    side: &mut dyn Side,
    sites: &mut Sites,
    fixture: &'static str,
    at: &str,
) -> Result<crate::case::Handle, Abort> {
    let held = Fixture::new(fixture);
    let loaded = held.loaded().map_err(|why| Abort {
        at: at.to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    made(sites, at, side.library(&loaded.source()))
}

impl Case for GpuRange {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in [KERNEL, VERTEX, FRAGMENT] {
            shaders.extend(shaders_of(FIXTURE, entry)?);
        }
        if self.drawn {
            shaders.extend(shaders_of(WRITER, WRITER)?);
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(FIXTURE)?;
        if self.drawn {
            for mut input in fixture_inputs(WRITER)? {
                input.name = format!("writer-{}", input.name);
                inputs.push(input);
            }
        }
        for (name, bytes) in kernel_buffers() {
            inputs.push(Input::bytes(name, &bytes));
        }
        inputs.push(Input::bytes("buffer:range", &words(&STALE)));
        inputs.push(Input::bytes(self.source_name(), &self.source()));
        inputs.push(Input::bytes(
            "execution",
            format!(
                "{COMMANDS} commands of DrawIndexed, inherited state, 3 vertex and 1 fragment \
                 buffers; the range at offset {OFFSET}; {}; target RGBA32Uint {}x{} cleared to 0",
                match self.drawn {
                    false => "copied by a blit before the pass",
                    true => "written by two points of the writer, then a barrier of buffers",
                },
                SIZE[0],
                SIZE[1]
            )
            .as_bytes(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library_handle = library(side, sites, FIXTURE, "library")?;
        let writer = match self.drawn {
            true => Some(library(side, sites, WRITER, "writer-library")?),
            false => None,
        };
        let mut buffers = Vec::new();
        for (name, bytes) in kernel_buffers() {
            buffers.push(made(sites, name, side.buffer(&bytes))?);
        }
        let range = made(sites, "buffer:range", side.buffer(&words(&STALE)))?;
        let source = made(sites, self.source_name(), side.buffer(&self.source()))?;
        let written = match writer {
            Some(library) => RangeWrite::Drawn {
                library,
                function: WRITER,
                params: source,
            },
            None => RangeWrite::Copied { from: source },
        };
        // The commands read the marks, the frame, the corners, the tints and the indices through
        // their bindings; the draws only the kernel reads.
        let run = EncodedRange {
            library: library_handle,
            kernel: KERNEL,
            buffers: &buffers,
            commands: COMMANDS,
            bind_counts: [3, 1],
            vertex: VERTEX,
            fragment: FRAGMENT,
            format: PixelFormat::RGBA32Uint,
            size: SIZE,
            row_pitch: SIZE[0] * TARGET_TEXEL,
            used: &buffers[1..],
            range,
            offset: OFFSET,
            written,
        };
        let filled = call(sites, "execution", side.encoded_range(&run), |filled| {
            filled.status.clone()
        })?;
        sites.push(Node::leaf(
            "target-texels",
            Observed::bytes(filled.bytes, target_layout()),
        ));
        Ok(())
    }
}

/// The family's rows, in the order the report lists them.
pub fn boxed() -> Vec<Box<dyn Case>> {
    [
        (AT_ZERO, [0, 3], false),
        (PAST_ZERO, [5, 2], false),
        (DRAWN, [2, 4], true),
    ]
    .into_iter()
    .map(|(name, range, drawn)| Box::new(GpuRange { name, range, drawn }) as Box<dyn Case>)
    .collect()
}
