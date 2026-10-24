//! The stage-in region a kernel's `[[stage_in]]` fetch clamps to.
//!
//! One case over the whole family: the owned fixture `stage_in_regions` run once per row, each row
//! a step function, a step rate, a region, a grid, a threadgroup and the way the region is set.
//! Every row's words are a site of their own, laid out one row of the buffer per row of the grid,
//! so a diff names the thread whose fetch moved rather than a table.
//!
//! One site a row holds the whole output buffer, under a layout whose rows are the grid's rows and
//! whose image is the buffer: the grid's words compare element by element, so a diff names the
//! thread whose fetch moved, and the bytes past the last row compare as the layout's gaps, so
//! "the dispatch wrote nothing past its grid" is a fact the comparison checks rather than one
//! nobody looked at. Every byte of the buffer starts at the same fill on both sides.
//!
//! The fixture's `stage_in_region_word` writes its attribute plus `0x100` and
//! `stage_in_region_pair` writes `(a << 16) | b`, so the recording holds what Metal wrote through
//! the committed fixture, not an attribute a probe kernel reported. Buffer 1 holds `0x10 + e` at
//! element `e`, buffer 2 `0x200 + e`, and buffer 3 the index `20 + 2i` at `i`, each a `UInt` of
//! stride 4; the output buffer is bound at 0.
//!
//! Nothing here can fault a GPU: one compute pass a row, four buffers, no texture, no loop, and
//! every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, Side, StageInAttribute,
    StageInLayout, StageInRegion, StageInput,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{AttributeFormat, StepFunction};
use metal2vulkan_runtime::Degradation;

/// The fixture both entry points live in.
const FIXTURE: &str = "stage_in_regions";

/// The entry point one attribute runs.
const WORD: &str = "stage_in_region_word";

/// The entry point two attributes run.
const PAIR: &str = "stage_in_region_pair";

/// The byte every output buffer holds before a row runs, so the words no thread wrote are a
/// recorded value rather than whatever the allocator left.
const FILL: u8 = 0xee;

/// The output buffer's length, the same for every row: longer than any row's grid, so every row
/// also records that the dispatch wrote nothing past it.
const OUTPUT_BYTES: usize = 1024;

/// The elements each source buffer holds.
const ELEMENTS: u32 = 256;

/// The Metal buffer index the indexed steps read their indices from.
const INDEX_BUFFER: u32 = 3;

/// How a row sets the stage-in region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum How {
    /// `setStageInRegion:`.
    Direct,
    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`, from a buffer holding the
    /// region's six words.
    Indirect,
    /// `setStageInRegion:` of 8 x 8 x 1, then of the row's own region: the later call replaces the
    /// earlier.
    Twice,
}

/// One row: the layouts of buffers 1 (and 2) as `(step, rate)`, the region, the grid and the
/// threadgroup in x and y, and how the region is set. One layout runs [`WORD`], two [`PAIR`].
struct Row {
    name: &'static str,
    layouts: &'static [(StepFunction, u32)],
    origin: [u32; 3],
    size: [u32; 3],
    grid: [u32; 2],
    group: [u32; 2],
    how: How,
}

impl Row {
    /// The entry point the row's layouts choose.
    fn entry(&self) -> &'static str {
        match self.layouts.len() {
            1 => WORD,
            _ => PAIR,
        }
    }

    /// The row's stage-input descriptor: attribute `i` a `UInt` at offset 0 of buffer `i + 1`,
    /// whose layout has stride 4 and the row's step and rate; 32-bit indices at buffer 3.
    fn stage_input(&self) -> StageInput {
        let attributes = (0..)
            .zip(self.layouts)
            .map(|(attribute, _)| StageInAttribute {
                index: attribute,
                format: AttributeFormat::UInt,
                offset: 0,
                buffer_index: attribute + 1,
            })
            .collect();
        let layouts = (0..)
            .zip(self.layouts)
            .map(|(attribute, &(step_function, step_rate))| StageInLayout {
                buffer_index: attribute + 1,
                stride: Some(4),
                step_function,
                step_rate,
            })
            .collect();
        StageInput::new(attributes, layouts, INDEX_BUFFER)
    }

    /// The region's six words, as the indirect form reads them: the origin, then the size.
    fn region_words(&self) -> Vec<u8> {
        self.origin
            .iter()
            .chain(&self.size)
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    /// The stage-in region calls the row makes, in order. `indirect` is the buffer holding
    /// [`Self::region_words`].
    fn stage_in(&self, indirect: Handle) -> Vec<StageInRegion> {
        let own = StageInRegion::Direct {
            origin: self.origin,
            size: self.size,
        };
        match self.how {
            How::Direct => vec![own],
            How::Indirect => vec![StageInRegion::Indirect {
                buffer: indirect,
                offset: 0,
            }],
            How::Twice => vec![
                StageInRegion::Direct {
                    origin: [0, 0, 0],
                    size: [8, 8, 1],
                },
                own,
            ],
        }
    }

    /// Everything the row sets that is not a buffer's bytes: its layouts, region, grid,
    /// threadgroup and the calls that set the region. Its digest is one of the case's inputs, so a
    /// row whose parameters move is `Unrecorded` at that row rather than a quiet pass.
    fn parameters(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.entry().as_bytes());
        for &(step, rate) in self.layouts {
            bytes.extend(step.as_raw().to_le_bytes());
            bytes.extend(rate.to_le_bytes());
            bytes.extend(AttributeFormat::UInt.as_raw().to_le_bytes());
            // The attribute's offset and its layout's stride.
            bytes.extend(0u32.to_le_bytes());
            bytes.extend(4u32.to_le_bytes());
        }
        bytes.extend(self.region_words());
        for axis in self.grid.iter().chain(&self.group) {
            bytes.extend(axis.to_le_bytes());
        }
        bytes.extend(u32::from(self.how as u8).to_le_bytes());
        bytes
    }

    /// The layout the row's output buffer is read under: one row of the layout per row of the
    /// grid, so a diff names the thread by its position, and one image of the whole buffer, so
    /// every byte the grid did not reach is one of the layout's gaps and compares as well.
    fn layout(&self) -> ByteLayout {
        let pitch = self.grid[0] * 4;
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: OUTPUT_BYTES as u64,
            },
            element: Element::U32,
            row_offset: 0,
            row_elements: self.grid[0],
            row_pitch: pitch,
            rows: self.grid[1],
            image_pitch: OUTPUT_BYTES as u32,
            images: 1,
            gaps: Gaps::Compared,
        }
    }
}

/// Every row the lane measured: its steps, region, grid, threadgroup and how the region is set.
///
/// The family is exhaustive in the directions that matter and small in every other: a step on x
/// and on y, a step rate of 1, 2 and 3, an origin inside and past the buffer's elements, a size of
/// 0, a size of 1, a size below the grid and a size equal to it, both axes of a 2D region, the
/// indexed steps, a constant step, a region set indirectly and a region set twice.
const ROWS: &[Row] = &[
    Row {
        name: "x_r8_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [8, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "x_r6_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "x_r6_g8_tg4",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [4, 1],
        how: How::Direct,
    },
    Row {
        name: "x_o2_r4_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [2, 0, 0],
        size: [4, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "x_r0_g4",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
    },
    Row {
        name: "x_o3_r0_g4",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [3, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
    },
    Row {
        name: "x_rate2_r2_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 2)],
        origin: [0, 0, 0],
        size: [2, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "x_rate2_r5_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 2)],
        origin: [0, 0, 0],
        size: [5, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "x_ind_r6_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Indirect,
    },
    Row {
        name: "x_twice_r6_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Twice,
    },
    Row {
        name: "y_r3_g2x6",
        layouts: &[(StepFunction::ThreadPositionInGridY, 1)],
        origin: [0, 0, 0],
        size: [1, 3, 1],
        grid: [2, 6],
        group: [2, 6],
        how: How::Direct,
    },
    Row {
        name: "y_o1_r3_g2x6",
        layouts: &[(StepFunction::ThreadPositionInGridY, 1)],
        origin: [0, 1, 0],
        size: [1, 3, 1],
        grid: [2, 6],
        group: [2, 6],
        how: How::Direct,
    },
    Row {
        name: "xy_r3x2_g4x4",
        layouts: &[
            (StepFunction::ThreadPositionInGridX, 1),
            (StepFunction::ThreadPositionInGridY, 1),
        ],
        origin: [0, 0, 0],
        size: [3, 2, 1],
        grid: [4, 4],
        group: [4, 4],
        how: How::Direct,
    },
    Row {
        name: "xy_o1x1_r2x2_g4x4",
        layouts: &[
            (StepFunction::ThreadPositionInGridX, 1),
            (StepFunction::ThreadPositionInGridY, 1),
        ],
        origin: [1, 1, 0],
        size: [2, 2, 1],
        grid: [4, 4],
        group: [4, 4],
        how: How::Direct,
    },
    Row {
        name: "xy_r0x2_g4x4",
        layouts: &[
            (StepFunction::ThreadPositionInGridX, 1),
            (StepFunction::ThreadPositionInGridY, 1),
        ],
        origin: [0, 0, 0],
        size: [0, 2, 1],
        grid: [4, 4],
        group: [4, 4],
        how: How::Direct,
    },
    Row {
        name: "xi_r6_g8",
        layouts: &[(StepFunction::ThreadPositionInGridXIndexed, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "xi_o2_r4_g8",
        layouts: &[(StepFunction::ThreadPositionInGridXIndexed, 1)],
        origin: [2, 0, 0],
        size: [4, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "xi_r0_g4",
        layouts: &[(StepFunction::ThreadPositionInGridXIndexed, 1)],
        origin: [0, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
    },
    Row {
        name: "yi_r3_g2x6",
        layouts: &[(StepFunction::ThreadPositionInGridYIndexed, 1)],
        origin: [0, 0, 0],
        size: [1, 3, 1],
        grid: [2, 6],
        group: [2, 6],
        how: How::Direct,
    },
    Row {
        name: "yi_o1_r3_g2x6",
        layouts: &[(StepFunction::ThreadPositionInGridYIndexed, 1)],
        origin: [0, 1, 0],
        size: [1, 3, 1],
        grid: [2, 6],
        group: [2, 6],
        how: How::Direct,
    },
    Row {
        name: "c_r6_g8",
        layouts: &[(StepFunction::Constant, 0)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "c_r0_g4",
        layouts: &[(StepFunction::Constant, 0)],
        origin: [0, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
    },
    Row {
        name: "x_r1_g4",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [1, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
    },
    Row {
        name: "x_rate2_o1_r2_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 2)],
        origin: [1, 0, 0],
        size: [2, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "x_rate2_o3_r0_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 2)],
        origin: [3, 0, 0],
        size: [0, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
    },
    Row {
        name: "x_rate3_o2_r3_g12",
        layouts: &[(StepFunction::ThreadPositionInGridX, 3)],
        origin: [2, 0, 0],
        size: [3, 1, 1],
        grid: [12, 1],
        group: [12, 1],
        how: How::Direct,
    },
    Row {
        name: "y_rate2_o1_r2_g2x8",
        layouts: &[(StepFunction::ThreadPositionInGridY, 2)],
        origin: [0, 1, 0],
        size: [1, 2, 1],
        grid: [2, 8],
        group: [2, 8],
        how: How::Direct,
    },
    Row {
        name: "xi_o2_r0_g4",
        layouts: &[(StepFunction::ThreadPositionInGridXIndexed, 1)],
        origin: [2, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
    },
];

/// Source buffer `index`'s bytes: buffer 1 holds `0x10 + e`, buffer 2 `0x200 + e`, and buffer 3
/// the index `20 + 2e`, each a `UInt` at element `e`.
fn source(index: u32) -> Vec<u8> {
    (0..ELEMENTS)
        .flat_map(|element| {
            let word = match index {
                1 => 0x10 + element,
                2 => 0x200 + element,
                _ => 20 + 2 * element,
            };
            word.to_le_bytes()
        })
        .collect()
}

/// The case.
pub struct StageInRegions;

/// The case the suite holds.
pub fn stage_in_region_clamps() -> StageInRegions {
    StageInRegions
}

impl StageInRegions {
    /// The fixture, under the entry point `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }
}

impl Case for StageInRegions {
    fn name(&self) -> &'static str {
        "stage-in-region-clamps"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The one row that sets its region indirectly reads that region on the host, because it is
        // GPU memory the work before the dispatch may write, and the runtime announces that. The
        // notice is a performance one: the words the row writes still compare exactly, and the
        // only site loosened is the set of notices, which Metal holds empty. The allowance fails
        // the case if the notice did not fire, so a row that stops reading the region on the host
        // has to drop this line rather than carry a stale excuse.
        // The second is the venue's own: the buffers this dispatch reads its arguments and writes
        // its answer through are `Shared`, which a device with no unified memory keeps in system
        // memory because that is the only memory both sides can reach.
        [
            Degradation::HostReadDispatchArguments,
            Degradation::SharedInSystemMemory,
        ]
        .iter()
        .map(|kind| crate::replay::notice_allowance(kind.code()))
        .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in [WORD, PAIR] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Self::fixture(entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture(WORD);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The output buffer every row starts from, so the words no thread wrote are part of
            // the answer rather than whatever was there.
            Input::bytes("buffer:0", &vec![FILL; OUTPUT_BYTES]),
        ];
        for index in 1..=3 {
            inputs.push(Input::bytes(format!("buffer:{index}"), &source(index)));
        }
        for row in ROWS {
            inputs.push(Input::bytes(format!("row:{}", row.name), &row.parameters()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = Self::fixture(WORD)
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let sources: Vec<Handle> = (1..=3)
            .map(|index| {
                made(
                    sites,
                    &format!("buffer/{index}"),
                    side.buffer(&source(index)),
                )
            })
            .collect::<Result<Vec<Handle>, Abort>>()?;

        for row in ROWS {
            let at = |site: &str| format!("{}/{site}", row.name);
            let pipeline = made(
                sites,
                &at("pipeline"),
                side.compute_pipeline_with_stage_input(library, row.entry(), &row.stage_input()),
            )?;
            let output = made(
                sites,
                &at("buffer/0"),
                side.buffer(&vec![FILL; OUTPUT_BYTES]),
            )?;
            let indirect = made(
                sites,
                &at("buffer/region"),
                side.buffer(&row.region_words()),
            )?;
            let mut bound = vec![output];
            bound.extend(&sources);
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::buffers(&bound),
                    &row.stage_in(indirect),
                    [row.grid[0], row.grid[1], 1],
                    [row.group[0], row.group[1], 1],
                ),
            )?;
            call(
                sites,
                &at("status"),
                side.wait(command_buffer),
                Clone::clone,
            )?;
            let bytes = call(
                sites,
                &at("read-back"),
                side.read_buffer(output, 0, OUTPUT_BYTES),
                |_| Value::Absent,
            )?;
            sites.push(
                Node::branch(row.name, Vec::new())
                    .with(Node::leaf("output", Observed::bytes(bytes, row.layout()))),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The bytes a row's grid writes: one word a thread.
    fn written(row: &Row) -> usize {
        row.grid[0] as usize * row.grid[1] as usize * 4
    }

    /// Every row's grid fits the output buffer, which the row's layout covers whole: a row whose
    /// grid outgrew [`OUTPUT_BYTES`] would read past the buffer rather than compare.
    #[test]
    fn every_rows_grid_fits_its_output_buffer() {
        for row in ROWS {
            assert!(
                written(row) <= OUTPUT_BYTES,
                "{}: its grid writes {} bytes of {OUTPUT_BYTES}",
                row.name,
                written(row)
            );
            assert!(
                row.layout().extent() <= OUTPUT_BYTES,
                "{}: its layout reaches {} bytes of {OUTPUT_BYTES}",
                row.name,
                row.layout().extent()
            );
        }
    }

    /// The rows' names are distinct, since each names a site of the transcript and the comparison
    /// pairs the sides' sites by name.
    #[test]
    fn the_rows_name_distinct_sites() {
        let named: BTreeSet<&str> = ROWS.iter().map(|row| row.name).collect();
        assert_eq!(named.len(), ROWS.len());
    }

    /// Each row's parameters digest differently, so the input fingerprint tells the rows apart:
    /// two rows that fed one digest would let a recording of one stand for the other.
    #[test]
    fn the_rows_parameters_are_distinct() {
        let digests: BTreeSet<String> = ROWS
            .iter()
            .map(|row| crate::sha256_hex(&row.parameters()))
            .collect();
        assert_eq!(digests.len(), ROWS.len());
    }
}
