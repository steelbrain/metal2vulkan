//! Mesh draws behind a memory barrier: whether a reader draw sees the writer draw's words.
//!
//! Ported from the runtime lane `runtime_lane/mesh_barriers.rs`, whose expectation is a transcript
//! of a Metal probe's printed lines (`PX`, `RESULT`, `CTL`). Metal's answer here is **run**, not
//! copied: the lane's own documentation says its table is what an **M4** on macOS 27.0.1 printed,
//! while the suite's oracle is this M3 Max, so porting the lane and recording it is also the first
//! measurement of this probe on the chip the recording is keyed to.
//!
//! Each row is one pass of one `RGBA32Uint` attachment, 8x8, cleared to zero and stored, and a
//! control buffer of four words, `(1, 1, 1, 0)` from the host. A writer draw covers texel (7, 7)
//! and stores its control words from the mesh stage or from the fragment stage; a barrier over
//! buffers, or over the control buffer, from the writer's stage to the reader's follows; then a
//! reader draw reads the words and draws with them.
//!
//! What a row observes is typed, not printed: the attachment's whole 8x8 as bytes under a layout
//! that reads them as 32-bit words, the texels drawn as a count, the control buffer's four words
//! after the pass as bytes of their own, every call of the pass by the Metal selector it makes,
//! and the command buffer's status and error. The lane folded all of that into one list of
//! formatted strings.
//!
//! The row whose reader is an indirect draw of an object function is a case of its own, so that a
//! refusal at its draw — a refused call stops a run — cannot cost the other six rows their sites.
//! **It is not refused on any device we have:** the claim that an emulating device refuses it
//! (`NotImplemented(MeshShading)`) was true until `bind::mesh_object_arguments_module` landed, and
//! the lane that still said so has been corrected. On MoltenVK 1.4.2 it draws Metal's words.
//!
//! Nothing here can fault a GPU: every grid is one or two threadgroups.

use crate::case::{
    Abort, Case, Failure, Handle, Input, MeshFunctions, MeshStep, MetalSource, Side, SCOPE_BUFFERS,
    STAGE_FRAGMENT, STAGE_MESH, STAGE_OBJECT, TARGET_TEXEL,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};

/// The owned fixture every row runs.
const FIXTURE: &str = "mesh_barrier";

/// The pass's side in texels, as the probe drew it.
const SIDE: u32 = 8;

/// The control words the host writes before the pass.
const CONTROL: [u32; 4] = [1, 1, 1, 0];

/// The fragment function every pipeline of the fixture writes its attachment through.
const FRAGMENT: &str = "mesh_barrier_fragment";

/// The writer draw's stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Writer {
    /// `mesh_barrier_writer` stores the control words from its mesh function.
    Mesh,
    /// `mesh_barrier_fragment_writer` stores them from its fragment function, under
    /// `mesh_barrier_cover`.
    Fragment,
}

impl Writer {
    /// The pipeline's mesh and fragment functions.
    fn functions(self) -> (&'static str, &'static str) {
        match self {
            Writer::Mesh => ("mesh_barrier_writer", FRAGMENT),
            Writer::Fragment => ("mesh_barrier_cover", "mesh_barrier_fragment_writer"),
        }
    }
}

/// The reader draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reader {
    /// The object function's, one object threadgroup.
    Object,
    /// Without an object function, a 4 by 3 grid of mesh threadgroups.
    Grid,
    /// Without an object function, its grid the control words.
    GridIndirect,
    /// The object function's, its object threadgroups the control words.
    ObjectIndirect,
}

impl Reader {
    /// The pipeline's object and mesh functions.
    fn functions(self) -> (Option<&'static str>, &'static str) {
        match self {
            Reader::Object | Reader::ObjectIndirect => {
                (Some("mesh_barrier_object"), "mesh_barrier_mesh")
            }
            Reader::Grid | Reader::GridIndirect => (None, "mesh_barrier_grid"),
        }
    }
}

/// The barrier between the two draws: over buffers, or over the control buffer, after and before
/// these raw `MTLRenderStages` bits.
#[derive(Clone, Copy, Debug)]
enum Barrier {
    Scope(u64, u64),
    Resources(u64, u64),
}

/// One row: the probe's name for it, the writer, the barrier and the reader.
#[derive(Clone, Copy, Debug)]
struct Row {
    name: &'static str,
    writer: Writer,
    barrier: Barrier,
    reader: Reader,
}

/// The rows both sides run to completion (the lane's `CASES` but for its last).
const ROWS: &[Row] = &[
    Row {
        name: "mesh-to-object",
        writer: Writer::Mesh,
        barrier: Barrier::Scope(STAGE_MESH, STAGE_OBJECT),
        reader: Reader::Object,
    },
    Row {
        name: "fragment-to-object",
        writer: Writer::Fragment,
        barrier: Barrier::Scope(STAGE_FRAGMENT, STAGE_OBJECT),
        reader: Reader::Object,
    },
    Row {
        name: "fragment-to-object-resources",
        writer: Writer::Fragment,
        barrier: Barrier::Resources(STAGE_FRAGMENT, STAGE_OBJECT),
        reader: Reader::Object,
    },
    Row {
        name: "fragment-to-mesh",
        writer: Writer::Fragment,
        barrier: Barrier::Scope(STAGE_FRAGMENT, STAGE_MESH),
        reader: Reader::Grid,
    },
    Row {
        name: "mesh-to-mesh",
        writer: Writer::Mesh,
        barrier: Barrier::Scope(STAGE_MESH, STAGE_MESH),
        reader: Reader::Grid,
    },
    Row {
        name: "fragment-to-indirect",
        writer: Writer::Fragment,
        barrier: Barrier::Scope(STAGE_FRAGMENT, STAGE_OBJECT | STAGE_MESH),
        reader: Reader::GridIndirect,
    },
];

/// The row the runtime refuses where a device emulates mesh draws: an indirect draw of an object
/// function (the lane's `fragment-to-indirect-object`).
const INDIRECT_OBJECT_ROW: &[Row] = &[Row {
    name: "fragment-to-indirect-object",
    writer: Writer::Fragment,
    barrier: Barrier::Scope(STAGE_FRAGMENT, STAGE_OBJECT | STAGE_MESH),
    reader: Reader::ObjectIndirect,
}];

/// A case of the family.
pub struct MeshBarriers {
    name: &'static str,
    rows: &'static [Row],
}

/// Every case of the family.
pub fn all() -> Vec<MeshBarriers> {
    vec![
        MeshBarriers {
            name: "mesh-barrier-readers",
            rows: ROWS,
        },
        MeshBarriers {
            name: "mesh-barrier-indirect-object-reader",
            rows: INDIRECT_OBJECT_ROW,
        },
    ]
}

impl MeshBarriers {
    /// The fixture both sides read.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// Run one row: its own attachment and control buffer, two pipelines, one pass, and the sites
    /// it observed. A step the side refused stops the run at that call, so a refusal is its own
    /// verdict rather than a pile of divergences over bytes a refused draw never wrote.
    fn row(
        &self,
        side: &mut dyn Side,
        library: Handle,
        row: &Row,
        sites: &mut Sites,
    ) -> Result<(), Abort> {
        let at = row.name;
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let mut node = Node::branch(at, Vec::new());

        let (writer_mesh, writer_fragment) = row.writer.functions();
        let writes = side
            .mesh_pipeline(
                library,
                &MeshFunctions {
                    object: None,
                    mesh: writer_mesh,
                    fragment: writer_fragment,
                    attachments: 1,
                },
            )
            .map_err(|failure| stop(format!("{at}/writer-pipeline"), failure))?;
        let (reader_object, reader_mesh) = row.reader.functions();
        let reads = side
            .mesh_pipeline(
                library,
                &MeshFunctions {
                    object: reader_object,
                    mesh: reader_mesh,
                    fragment: FRAGMENT,
                    attachments: 1,
                },
            )
            .map_err(|failure| stop(format!("{at}/reader-pipeline"), failure))?;

        let words: Vec<u8> = CONTROL.iter().flat_map(|word| word.to_le_bytes()).collect();
        let control = side
            .buffer(&words)
            .map_err(|failure| stop(format!("{at}/control"), failure))?;
        let target = side
            .render_target(SIDE, SIDE)
            .map_err(|failure| stop(format!("{at}/render_target"), failure))?;

        let mut steps = vec![
            MeshStep::Pipeline(writes),
            match row.writer {
                Writer::Mesh => MeshStep::MeshBuffer {
                    buffer: control,
                    offset: 0,
                    index: 0,
                },
                Writer::Fragment => MeshStep::FragmentBuffer {
                    buffer: control,
                    offset: 0,
                    index: 0,
                },
            },
            MeshStep::DrawThreadgroups {
                groups: ONE,
                object: ONE,
                mesh: ONE,
            },
            match row.barrier {
                Barrier::Scope(after, before) => MeshStep::BarrierScope {
                    scope: SCOPE_BUFFERS,
                    after,
                    before,
                },
                Barrier::Resources(after, before) => MeshStep::BarrierResources {
                    buffers: vec![control],
                    after,
                    before,
                },
            },
            MeshStep::Pipeline(reads),
        ];
        match row.reader {
            Reader::Object => {
                steps.push(MeshStep::ObjectBuffer {
                    buffer: control,
                    offset: 0,
                    index: 0,
                });
                steps.push(MeshStep::DrawThreadgroups {
                    groups: ONE,
                    object: ONE,
                    mesh: ONE,
                });
            }
            Reader::Grid => {
                steps.push(MeshStep::MeshBuffer {
                    buffer: control,
                    offset: 0,
                    index: 0,
                });
                steps.push(MeshStep::DrawThreadgroups {
                    groups: [4, 3, 1],
                    object: ONE,
                    mesh: ONE,
                });
            }
            Reader::GridIndirect => {
                steps.push(MeshStep::MeshBuffer {
                    buffer: control,
                    offset: 0,
                    index: 0,
                });
                steps.push(MeshStep::DrawIndirect {
                    buffer: control,
                    offset: 0,
                    object: ONE,
                    mesh: ONE,
                });
            }
            Reader::ObjectIndirect => {
                steps.push(MeshStep::ObjectBuffer {
                    buffer: control,
                    offset: 0,
                    index: 0,
                });
                steps.push(MeshStep::DrawIndirect {
                    buffer: control,
                    offset: 0,
                    object: ONE,
                    mesh: ONE,
                });
            }
        }

        let outcome = side
            .mesh_pass(&[target], &steps)
            .map_err(|failure| stop(format!("{at}/pass"), failure))?;
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
        node = node.with(calls);
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

        let texels = side
            .read_texture(target, [SIDE, SIDE], SIDE * TARGET_TEXEL)
            .map_err(|failure| stop(format!("{at}/attachment"), failure))?;
        node = node.with(Node::leaf(
            "attachment",
            Observed::bytes(texels.clone(), attachment_layout()),
        ));
        node = node.with(Node::leaf(
            "drawn",
            Observed::value(Value::Uint(drawn(&texels))),
        ));
        let after = side
            .read_buffer(control, 0, words.len())
            .map_err(|failure| stop(format!("{at}/control-after"), failure))?;
        node = node.with(Node::leaf(
            "control-after",
            Observed::bytes(
                after,
                ByteLayout::buffer(0, words.len() as u64, Element::U32),
            ),
        ));
        sites.push(node);
        Ok(())
    }
}

/// One threadgroup, one thread.
const ONE: [u32; 3] = [1, 1, 1];

/// The layout the attachment's texels are read under: the whole 8x8 of level 0 of slice 0, four
/// 32-bit words a texel, packed, so a diff names the texel's row and the word inside it.
fn attachment_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: SIDE * TARGET_TEXEL / 4,
        row_pitch: SIDE * TARGET_TEXEL,
        rows: SIDE,
        image_pitch: SIDE * SIDE * TARGET_TEXEL,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The texels the attachment holds a non-zero word in: the probe's `drawn=`.
fn drawn(texels: &[u8]) -> u64 {
    let texel = TARGET_TEXEL as usize;
    texels
        .chunks(texel)
        .filter(|words| words.iter().any(|byte| *byte != 0))
        .count() as u64
}

impl Case for MeshBarriers {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: FRAGMENT.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let mut rows = Vec::new();
        for row in self.rows {
            rows.extend(row.name.as_bytes());
            rows.extend(
                format!("={:?},{:?},{:?}\n", row.writer, row.barrier, row.reader).as_bytes(),
            );
        }
        let control: Vec<u8> = CONTROL.iter().flat_map(|word| word.to_le_bytes()).collect();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:0", &control),
            Input::bytes(
                "render-targets",
                format!("1x RGBA32Uint {SIDE}x{SIDE} cleared to zero").as_bytes(),
            ),
            Input::bytes("rows", &rows),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        for row in self.rows {
            self.row(side, library, row, sites)?;
        }
        Ok(())
    }
}
