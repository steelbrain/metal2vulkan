//! Object and mesh functions that wait at threadgroup barriers: what each thread wrote, in draws
//! whose object threadgroups are partial and in whole ones.
//!
//! The owned fixture `mesh_partial_barrier` holds three object functions, whose barriers stand in
//! uniform control flow and in a loop, after a divergent early return, and under a
//! thread-dependent condition, and a mesh function that returns before its barrier. Every thread
//! counts itself in word 0 of a buffer and writes two words at its grid position: what it read
//! across the barriers, and its threadgroup's threads with its own index. A `drawMeshThreads` whose
//! threads leave a partial object threadgroup runs its last threadgroup along an axis with the
//! threads left, so the count, the words and the threadgroup shapes say whether a partial
//! threadgroup ran only Metal's threads, whether they synchronized among themselves, and that
//! nothing else wrote: the words no thread owns keep the case's poison.
//!
//! What a row observes is typed: the buffer's whole bytes after the pass, read as 32-bit words,
//! every call of the pass by the Metal selector it makes, and the command buffer's status and
//! error. A refused call stops the run at its site.
//!
//! Nothing here can fault a GPU: every grid is a few threadgroups, and every object threadgroup
//! launches no mesh threadgroup.

use crate::allowance::Allowance;
use crate::case::{
    Abort, Case, Failure, Handle, Input, MeshFunctions, MeshStep, MetalSource, Side,
};
use crate::fixture::{Fixture, Loaded, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};

/// The owned fixture every case of the family runs.
const FIXTURE: &str = "mesh_partial_barrier";

/// The pass's side in texels.
const SIDE: u32 = 4;

/// The fragment function of every pipeline of the fixture.
const FRAGMENT: &str = "mesh_partial_barrier_fragment";

/// The control words: the uniform function's loop trip count, and the threads that stay past
/// the divergent function's early return or take its barrier.
const CONTROL: [u32; 4] = [3, 3, 0, 0];

/// The word every word of the buffer but the count starts as.
const POISON: u32 = 0xABAB_ABAB;

/// One draw of a row.
#[derive(Clone, Copy, Debug)]
enum Draw {
    /// `drawMeshThreads` of these threads in object threadgroups of these.
    Threads([u32; 3], [u32; 3]),
    /// `drawMeshThreadgroups` of these threadgroups, of these threads per object threadgroup
    /// and of these per mesh threadgroup.
    Groups([u32; 3], [u32; 3], [u32; 3]),
}

impl Draw {
    /// The threads of the grid, whose positions own their words.
    fn threads(self) -> [u32; 3] {
        match self {
            Draw::Threads(threads, _) => threads,
            Draw::Groups(groups, object, mesh) => {
                let per = if object == [1, 1, 1] { mesh } else { object };
                [0, 1, 2].map(|i| groups[i] * per[i])
            }
        }
    }

    /// The words the draw's parameters fingerprint as.
    fn words(self) -> Vec<u8> {
        let mut out = Vec::new();
        let axes = |out: &mut Vec<u8>, axes: [u32; 3]| {
            for axis in axes {
                out.extend(axis.to_le_bytes());
            }
        };
        match self {
            Draw::Threads(threads, object) => {
                out.extend(b"threads");
                axes(&mut out, threads);
                axes(&mut out, object);
            }
            Draw::Groups(groups, object, mesh) => {
                out.extend(b"groups");
                axes(&mut out, groups);
                axes(&mut out, object);
                axes(&mut out, mesh);
            }
        }
        out
    }
}

/// A row: its name and its draw.
#[derive(Clone, Copy, Debug)]
struct Row {
    name: &'static str,
    draw: Draw,
}

/// The object functions' draws: partial threadgroups along x, along both axes, past a simdgroup
/// and of fewer threads than a threadgroup, then whole ones by both draw calls.
const OBJECT_ROWS: &[Row] = &[
    Row {
        name: "threads-6-by-4",
        draw: Draw::Threads([6, 1, 1], [4, 1, 1]),
    },
    Row {
        name: "threads-5x3-by-4x2",
        draw: Draw::Threads([5, 3, 1], [4, 2, 1]),
    },
    Row {
        name: "threads-37-by-32",
        draw: Draw::Threads([37, 1, 1], [32, 1, 1]),
    },
    Row {
        name: "threads-2-by-4",
        draw: Draw::Threads([2, 1, 1], [4, 1, 1]),
    },
    Row {
        name: "threads-8x2-by-4x2",
        draw: Draw::Threads([8, 2, 1], [4, 2, 1]),
    },
    Row {
        name: "groups-2-by-4",
        draw: Draw::Groups([2, 1, 1], [4, 1, 1], [1, 1, 1]),
    },
];

/// The mesh function's draws, without an object function.
const MESH_ROWS: &[Row] = &[
    Row {
        name: "groups-1-by-8",
        draw: Draw::Groups([1, 1, 1], [1, 1, 1], [8, 1, 1]),
    },
    Row {
        name: "groups-1-by-2",
        draw: Draw::Groups([1, 1, 1], [1, 1, 1], [2, 1, 1]),
    },
];

/// A case of the family: the rows over one pipeline of the fixture.
pub struct MeshPartialBarriers {
    name: &'static str,
    /// The object function, where the case's pipeline has one: then it binds the buffer and the
    /// control words to the object stage, else to the mesh stage.
    object: Option<&'static str>,
    mesh: &'static str,
    rows: &'static [Row],
}

/// Every case of the family.
pub fn all() -> Vec<MeshPartialBarriers> {
    let object = |name, function| MeshPartialBarriers {
        name,
        object: Some(function),
        mesh: "mesh_partial_barrier_mesh",
        rows: OBJECT_ROWS,
    };
    vec![
        object(
            "mesh-partial-barrier-uniform",
            "mesh_partial_barrier_uniform",
        ),
        object(
            "mesh-partial-barrier-early-return",
            "mesh_partial_barrier_early_return",
        ),
        object("mesh-partial-barrier-if", "mesh_partial_barrier_if"),
        MeshPartialBarriers {
            name: "mesh-partial-barrier-mesh-early-return",
            object: None,
            mesh: "mesh_partial_barrier_mesh_early_return",
            rows: MESH_ROWS,
        },
    ]
}

/// The buffer's words for `draw`: the count, three spare words and two words per grid position.
fn buffer_words(draw: Draw) -> usize {
    let [x, y, z] = draw.threads();
    4 + 2 * (x * y * z) as usize
}

/// The layout the buffer is read under: its words, one row.
fn buffer_layout(words: usize) -> ByteLayout {
    let words = u32::try_from(words).unwrap_or(u32::MAX);
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(words) * 4,
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: words,
        row_pitch: words * 4,
        rows: 1,
        image_pitch: words * 4,
        images: 1,
        gaps: Gaps::Compared,
    }
}

impl MeshPartialBarriers {
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    fn functions(&self) -> MeshFunctions<'_> {
        MeshFunctions {
            object: self.object,
            mesh: self.mesh,
            fragment: FRAGMENT,
            attachments: 1,
        }
    }

    /// Run one row: a fresh pass and buffer of its own, emitting the sites it observed.
    fn row(
        &self,
        side: &mut dyn Side,
        pipeline: Handle,
        row: &Row,
        sites: &mut Sites,
    ) -> Result<(), Abort> {
        let at = row.name;
        let stop = |at: &str, failure: Failure| Abort {
            at: at.to_owned(),
            failure,
        };
        let target = side
            .render_target(SIDE, SIDE)
            .map_err(|failure| stop(&format!("{at}/render_target"), failure))?;
        let words = buffer_words(row.draw);
        let mut initial = vec![POISON; words];
        initial[0] = 0;
        let bytes: Vec<u8> = initial.iter().flat_map(|w| w.to_le_bytes()).collect();
        let buffer = side
            .buffer(&bytes)
            .map_err(|failure| stop(&format!("{at}/buffer"), failure))?;
        let control: Vec<u8> = CONTROL.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut steps = vec![MeshStep::Pipeline(pipeline)];
        if self.object.is_some() {
            steps.push(MeshStep::ObjectBuffer {
                buffer,
                offset: 0,
                index: 0,
            });
            steps.push(MeshStep::ObjectBytes {
                bytes: control,
                index: 1,
            });
        } else {
            let control = side
                .buffer(&control)
                .map_err(|failure| stop(&format!("{at}/control"), failure))?;
            steps.push(MeshStep::MeshBuffer {
                buffer,
                offset: 0,
                index: 0,
            });
            steps.push(MeshStep::MeshBuffer {
                buffer: control,
                offset: 0,
                index: 1,
            });
        }
        steps.push(match row.draw {
            Draw::Threads(threads, object) => MeshStep::DrawThreads {
                threads,
                object,
                mesh: [1, 1, 1],
            },
            Draw::Groups(groups, object, mesh) => MeshStep::DrawThreadgroups {
                groups,
                object,
                mesh,
            },
        });
        let outcome = side
            .mesh_pass(&[target], &steps)
            .map_err(|failure| stop(&format!("{at}/pass"), failure))?;
        let mut node = Node::branch(at, Vec::new());
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
        if let Some((path, failure)) =
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
            return Err(Abort { at: path, failure });
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
        let read = side
            .read_buffer(buffer, 0, words * 4)
            .map_err(|failure| stop(&format!("{at}/words"), failure))?;
        node = node.with(Node::leaf(
            "words",
            Observed::bytes(read, buffer_layout(words)),
        ));
        sites.push(node);
        Ok(())
    }
}

impl Case for MeshPartialBarriers {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        super::mesh_draws::mesh_notice_allowances()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: self.object.unwrap_or(self.mesh).to_owned(),
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
            rows.push(b'=');
            rows.extend(row.draw.words());
            rows.push(b'\n');
        }
        let mut functions = Vec::new();
        functions.extend(self.object.unwrap_or("-").as_bytes());
        functions.push(b',');
        functions.extend(self.mesh.as_bytes());
        functions.push(b',');
        functions.extend(FRAGMENT.as_bytes());
        let control: Vec<u8> = CONTROL.iter().flat_map(|w| w.to_le_bytes()).collect();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("functions", &functions),
            Input::bytes("control", &control),
            Input::bytes(
                "buffer",
                format!("word 0 zero, the rest {POISON:#x}; one RGBA32Uint {SIDE}x{SIDE} target")
                    .as_bytes(),
            ),
            Input::bytes("rows", &rows),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded: Loaded = self
            .fixture()
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = crate::case::made(
            sites,
            "pipeline",
            side.mesh_pipeline(library, &self.functions()),
        )?;
        for row in self.rows {
            self.row(side, pipeline, row, sites)?;
        }
        Ok(())
    }
}
