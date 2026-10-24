//! Mesh draws: the words each mesh threadgroup's primitive drew, and the grids Metal draws
//! nothing for.
//!
//! Ported from the runtime lane `runtime_lane/mesh_draws.rs`, whose expectation is a transcript of
//! a Metal probe's printed lines (`PX <case> <x> <y> a=<words> b=<words>`, then `RESULT <case>
//! drawn=<n>`). The lane holds those lines as text and compares its own printed lines with them.
//! Here the same measurement is **run on Metal** instead: the case is one body, record mode drives
//! it through `objc2-metal` on this Mac, and replay drives it through the runtime on any Vulkan
//! device.
//!
//! What a row observes is typed, not printed:
//!
//! - each attachment's whole 8x8 of `RGBA32Uint` texels as bytes under a [`ByteLayout`] that names
//!   the level, the slice and the extent, read as 32-bit words, so a difference names the texel's
//!   row, the word inside it and both values — where the lane's line diff named a whole line;
//! - the texels any attachment wrote a non-zero word into, as a count, which is the lane's
//!   `drawn=` at the altitude a reader wants it;
//! - every call of the pass, in order, by the Metal selector it makes, so a draw we refuse where
//!   Metal drew is a [`Refused`](crate::verdict::Verdict::Refused) verdict naming the call rather
//!   than a line printed to stderr;
//! - the command buffer's status and its error.
//!
//! The lane's line form drops every all-zero texel, so the two forms detect the same differences;
//! what the typed form adds is where the failure points and that the refusals are counted.
//!
//! Three families are separate cases because their verdicts mean different things: the grid draws
//! and the object draws, which match Metal; the grids past Metal's limits, which draw nothing; and
//! the object draws that leave a partial object threadgroup, which the runtime refuses
//! (`NotImplemented(MeshShading)`) where Metal draws. Keeping the last apart is the point of the
//! `Refused` verdict: the lane printed that gap and passed.
//!
//! Nothing here can fault a GPU: every grid is either small or one Metal and the runtime both
//! answer by drawing nothing. The lane's other two tests are **not** ported here. Its object-grid
//! counter tables (`OBJECT_GRID_CASES`) are a raw-value table rather than probe lines, so they
//! belong to a later batch, and one of their rows is a grid Metal's GPU discards — a recovery
//! victim, which this round must not run.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{
    Abort, Case, Failure, Handle, Input, MeshFunctions, MeshStep, MetalSource, Side, TARGET_TEXEL,
};
use crate::fixture::{Fixture, Loaded, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan_runtime::Degradation;

/// The owned fixture every case of the family runs.
const FIXTURE: &str = "mesh_draws";

/// The pass's side in texels, as the probe drew it.
const SIDE: u32 = 8;

/// The color attachments every case's pass holds.
const ATTACHMENTS: u32 = 2;

/// The bytes of the indirect buffer a probe's indirect draw reads its counts from: the probe's own
/// 64, zeroed but for the counts.
const INDIRECT_BYTES: usize = 64;

/// One mesh draw of the probe, with its threads per object threadgroup and per mesh threadgroup.
#[derive(Clone, Copy, Debug)]
enum Draw {
    /// `drawMeshThreadgroups` of these threadgroups.
    Groups([u32; 3], [u32; 3], [u32; 3]),
    /// `drawMeshThreads` of these threads.
    Threads([u32; 3], [u32; 3], [u32; 3]),
    /// `drawMeshThreadgroupsWithIndirectBuffer` of these counts at this offset of a zeroed
    /// [`INDIRECT_BYTES`]-byte buffer.
    Indirect([u32; 3], u64, [u32; 3], [u32; 3]),
}

impl Draw {
    /// The words the draw's parameters fingerprint as.
    fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let axes = |out: &mut Vec<u8>, axes: &[u32; 3]| {
            for axis in axes {
                out.extend(axis.to_le_bytes());
            }
        };
        match self {
            Draw::Groups(grid, object, mesh) => {
                out.extend(b"groups");
                axes(&mut out, grid);
                axes(&mut out, object);
                axes(&mut out, mesh);
            }
            Draw::Threads(grid, object, mesh) => {
                out.extend(b"threads");
                axes(&mut out, grid);
                axes(&mut out, object);
                axes(&mut out, mesh);
            }
            Draw::Indirect(counts, offset, object, mesh) => {
                out.extend(b"indirect");
                axes(&mut out, counts);
                out.extend(offset.to_le_bytes());
                axes(&mut out, object);
                axes(&mut out, mesh);
            }
        }
        out
    }
}

/// How a case binds the object function's control words: in a buffer, or as inline bytes, which
/// the probe read alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Binding {
    Buffer,
    Inline,
}

impl Binding {
    /// The word the row's name carries.
    fn word(self) -> &'static str {
        match self {
            Binding::Buffer => "buffer",
            Binding::Inline => "inline",
        }
    }
}

/// One row of a case: the probe's name for it, the object function's control words where the case
/// has an object function, and the draw.
#[derive(Clone, Copy, Debug)]
struct Row {
    name: &'static str,
    control: Option<[u32; 4]>,
    binding: Binding,
    draw: Draw,
}

/// The probe's cases of `mesh_draws_grid`, its threads per object threadgroup 1 (the lane's
/// `GRID_CASES`).
const GRID_ROWS: &[Row] = &[
    grid(
        "grid-groups-3x2-by-4",
        Draw::Groups([3, 2, 1], ONE, [4, 1, 1]),
    ),
    grid(
        "grid-groups-2x2-by-2x3",
        Draw::Groups([2, 2, 1], ONE, [2, 3, 1]),
    ),
    grid(
        "grid-threads-12x4-by-4x2",
        Draw::Threads([12, 4, 1], ONE, [4, 2, 1]),
    ),
    grid(
        "grid-threads-10x3-by-4x2",
        Draw::Threads([10, 3, 1], ONE, [4, 2, 1]),
    ),
    grid(
        "grid-threads-3x1-by-4",
        Draw::Threads([3, 1, 1], ONE, [4, 1, 1]),
    ),
    grid(
        "grid-indirect-3x1-by-8",
        Draw::Indirect([3, 1, 1], 8, ONE, [8, 1, 1]),
    ),
    grid("grid-groups-0", Draw::Groups([0, 2, 1], ONE, [4, 1, 1])),
];

/// Grids of `mesh_draws_grid` Metal draws nothing for (the lane's `PAST_CASES`): an axis past
/// 65,535, direct and indirect, and 2^20 threadgroups.
const GRID_PAST_ROWS: &[Row] = &[
    grid("grid-groups-65536", Draw::Groups([65_536, 1, 1], ONE, ONE)),
    grid(
        "grid-indirect-1x1x65536",
        Draw::Indirect([1, 1, 65_536], 0, ONE, ONE),
    ),
    grid(
        "grid-indirect-1024x1024",
        Draw::Indirect([1024, 1024, 1], 0, ONE, ONE),
    ),
    grid(
        "grid-groups-1024x1024",
        Draw::Groups([1024, 1024, 1], ONE, ONE),
    ),
];

/// The probe's object cases whose draw is of threadgroups or indirect (the lane's `OBJECT_CASES`
/// but for its two `drawMeshThreads` rows), each with the object function's control words, in both
/// binding forms.
const OBJECT_ROWS: &[Row] = &[
    object(
        "object-groups-3-by-4-mesh-2",
        [2, 1, 0, 0],
        Binding::Buffer,
        Draw::Groups([3, 1, 1], [4, 1, 1], [2, 1, 1]),
    ),
    object(
        "object-groups-3-by-4-mesh-2",
        [2, 1, 0, 0],
        Binding::Inline,
        Draw::Groups([3, 1, 1], [4, 1, 1], [2, 1, 1]),
    ),
    object(
        "object-groups-2-mesh-2x2",
        [1, 2, 0, 0],
        Binding::Buffer,
        Draw::Groups([2, 1, 1], [2, 1, 1], [2, 2, 1]),
    ),
    object(
        "object-groups-2-mesh-2x2",
        [1, 2, 0, 0],
        Binding::Inline,
        Draw::Groups([2, 1, 1], [2, 1, 1], [2, 2, 1]),
    ),
    object(
        "object-grid-0-then-1",
        [0, 1, 0, 0],
        Binding::Buffer,
        Draw::Groups([2, 1, 1], ONE, [2, 1, 1]),
    ),
    object(
        "object-grid-0-then-1",
        [0, 1, 0, 0],
        Binding::Inline,
        Draw::Groups([2, 1, 1], ONE, [2, 1, 1]),
    ),
    object(
        "object-indirect-2",
        [1, 1, 0, 0],
        Binding::Buffer,
        Draw::Indirect([2, 1, 1], 4, [4, 1, 1], [2, 1, 1]),
    ),
    object(
        "object-indirect-2",
        [1, 1, 0, 0],
        Binding::Inline,
        Draw::Indirect([2, 1, 1], 4, [4, 1, 1], [2, 1, 1]),
    ),
];

/// The probe's object cases whose `drawMeshThreads` leaves a partial object threadgroup (the
/// lane's `OBJECT_CASES`' two `Threads` rows). Metal draws them, running the threads left; the
/// runtime refuses them, which is these cases' whole subject.
///
/// Each is a case of its own, holding one row. A refused call stops a run, and the run's later
/// rows are then neither compared nor reported, so two refusing rows in one case would report one
/// refusal and lose the other.
const OBJECT_THREAD_ROWS: [&[Row]; 2] = [
    &[object(
        "object-threads-6-by-4",
        [2, 1, 0, 0],
        Binding::Buffer,
        Draw::Threads([6, 1, 1], [4, 1, 1], [2, 1, 1]),
    )],
    &[object(
        "object-threads-5x3-by-4x2",
        [1, 1, 0, 0],
        Binding::Buffer,
        Draw::Threads([5, 3, 1], [4, 2, 1], [2, 1, 1]),
    )],
];

/// Object draws whose object threadgroups each set a mesh grid Metal launches nothing for (the
/// lane's `OBJECT_PAST_CASES`).
const OBJECT_PAST_ROWS: &[Row] = &[
    object(
        "object-grid-1024x1024",
        [1024, 1024, 0, 0],
        Binding::Inline,
        Draw::Groups([2, 1, 1], ONE, ONE),
    ),
    object(
        "object-grid-65536",
        [65_536, 1, 0, 0],
        Binding::Inline,
        Draw::Groups([2, 1, 1], ONE, ONE),
    ),
];

/// One threadgroup, one thread.
const ONE: [u32; 3] = [1, 1, 1];

/// A row of the grid pipeline, which has no object function and binds no control words.
const fn grid(name: &'static str, draw: Draw) -> Row {
    Row {
        name,
        control: None,
        binding: Binding::Buffer,
        draw,
    }
}

/// A row of the object pipeline.
const fn object(name: &'static str, control: [u32; 4], binding: Binding, draw: Draw) -> Row {
    Row {
        name,
        control: Some(control),
        binding,
        draw,
    }
}

/// A case of the family: the probe's rows over one of the fixture's two mesh pipelines.
pub struct MeshDraws {
    name: &'static str,
    /// The object function, where the case's pipeline has one.
    object: Option<&'static str>,
    mesh: &'static str,
    rows: &'static [Row],
}

/// Every case of the family.
pub fn all() -> Vec<MeshDraws> {
    vec![
        MeshDraws {
            name: "mesh-grid-draws",
            object: None,
            mesh: "mesh_draws_grid",
            rows: GRID_ROWS,
        },
        MeshDraws {
            name: "mesh-grid-draws-past-limits",
            object: None,
            mesh: "mesh_draws_grid",
            rows: GRID_PAST_ROWS,
        },
        MeshDraws {
            name: "mesh-object-draws",
            object: Some("mesh_draws_object"),
            mesh: "mesh_draws_mesh",
            rows: OBJECT_ROWS,
        },
        MeshDraws {
            name: "mesh-object-draws-partial-threadgroup-6-by-4",
            object: Some("mesh_draws_object"),
            mesh: "mesh_draws_mesh",
            rows: OBJECT_THREAD_ROWS[0],
        },
        MeshDraws {
            name: "mesh-object-draws-partial-threadgroup-5x3-by-4x2",
            object: Some("mesh_draws_object"),
            mesh: "mesh_draws_mesh",
            rows: OBJECT_THREAD_ROWS[1],
        },
        MeshDraws {
            name: PAST_LIMITS,
            object: Some("mesh_draws_object"),
            mesh: "mesh_draws_mesh",
            rows: OBJECT_PAST_ROWS,
        },
    ]
}

/// The one row whose object grid is past the device's limits, so its pass allocates no `Shared`
/// buffer and announces nothing for one.
const PAST_LIMITS: &str = "mesh-object-draws-past-limits";

/// The fragment function every case's pipeline writes its attachments through.
const FRAGMENT: &str = "mesh_draws_fragment";

/// The degrade notices a mesh render pass may announce, each of which a mesh case loosens its
/// notices site by, because Metal announces none and a notice our side announced is the one field
/// that may then differ.
///
/// It is a list and not a device query, because a case never branches on the device: a case asks
/// by running and reads what came back. A venue that announces a kind not here fails the case
/// until its kind is added, which is the model working: an announced difference nobody named is
/// not an announced difference.
pub const MESH_PASS_NOTICES: &[Degradation] = &[
    // The pass drew through the mesh-draw emulation, on a device without `VK_EXT_mesh_shader`.
    Degradation::EmulatedMeshDraws,
    // A stage's compiler rewrote its fast-math arithmetic as it made the pipeline.
    Degradation::FastMathRewrites,
    // The notes the barrier and draw path makes: a fragment stage ordered before a later draw's
    // earlier stages, a pass split at a barrier, and the bytes of draws the pass does not order.
    Degradation::FragmentsBeforeGeometry,
    Degradation::RenderPassSplit,
    Degradation::UnorderedDrawBytes,
];

/// The allowances a mesh case keeps: one over its notices site per kind of
/// [`MESH_PASS_NOTICES`], so a device that announced one may hold it in its set and a device that
/// announced none compares the set exactly against Metal's empty one.
pub fn mesh_notice_allowances() -> Vec<Allowance> {
    let mut allowances = vec![
        // These passes read their payloads, draw arguments and attachments back through `Shared`
        // buffers ([`crate::replay::shared_memory_notice`]). Every row of the three mesh families
        // but `mesh-object-draws-past-limits` was measured announcing it on the RTX 5080; that row
        // stops before it allocates one, so it declares nothing
        // ([`MeshDraws::allowances`]).
        crate::replay::shared_memory_notice(),
    ];
    allowances.extend(plain_mesh_notice_allowances());
    allowances
}

/// The same list without the notice a device's own memory model provokes, for the one case that
/// never reaches a `Shared` buffer.
fn plain_mesh_notice_allowances() -> Vec<Allowance> {
    MESH_PASS_NOTICES
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

impl MeshDraws {
    /// The fixture both sides read.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The pipeline's functions.
    fn functions(&self) -> MeshFunctions<'_> {
        MeshFunctions {
            object: self.object,
            mesh: self.mesh,
            fragment: FRAGMENT,
            attachments: ATTACHMENTS,
        }
    }

    /// The row's site name: the probe's own, with the binding form where the case has two of one
    /// row.
    fn row_site(&self, row: &Row) -> String {
        match row.control {
            Some(_) => format!("{}/{}", row.name, row.binding.word()),
            None => row.name.to_owned(),
        }
    }

    /// Run one row: a fresh pass of its own targets, emitting the sites it observed.
    ///
    /// A step the side refused stops the run at that call, after the call's site is emitted. That
    /// is what makes a refusal its own verdict: were the run to carry on and read the attachments
    /// back, the bytes a refused draw did not write would be divergences of their own and the
    /// refusal would be buried under them.
    fn row(
        &self,
        side: &mut dyn Side,
        pipeline: Handle,
        row: &Row,
        sites: &mut Sites,
    ) -> Result<(), Abort> {
        let at = self.row_site(row);
        let mut node = Node::branch(at.clone(), Vec::new());
        let stop = |at: &str, failure: Failure| Abort {
            at: at.to_owned(),
            failure,
        };
        let mut targets = Vec::new();
        for _ in 0..ATTACHMENTS {
            targets.push(
                side.render_target(SIDE, SIDE)
                    .map_err(|failure| stop(&format!("{at}/render_target"), failure))?,
            );
        }
        let mut steps = vec![MeshStep::Pipeline(pipeline)];
        if let Some(control) = row.control {
            let bytes: Vec<u8> = control.iter().flat_map(|word| word.to_le_bytes()).collect();
            steps.push(match row.binding {
                Binding::Inline => MeshStep::ObjectBytes { bytes, index: 0 },
                Binding::Buffer => MeshStep::ObjectBuffer {
                    buffer: side
                        .buffer(&bytes)
                        .map_err(|failure| stop(&format!("{at}/control"), failure))?,
                    offset: 0,
                    index: 0,
                },
            });
        }
        steps.push(match row.draw {
            Draw::Groups(groups, object, mesh) => MeshStep::DrawThreadgroups {
                groups,
                object,
                mesh,
            },
            Draw::Threads(threads, object, mesh) => MeshStep::DrawThreads {
                threads,
                object,
                mesh,
            },
            Draw::Indirect(counts, offset, object, mesh) => {
                let mut bytes = vec![0u8; INDIRECT_BYTES];
                for (axis, count) in counts.iter().enumerate() {
                    let first = offset as usize + 4 * axis;
                    bytes[first..first + 4].copy_from_slice(&count.to_le_bytes());
                }
                MeshStep::DrawIndirect {
                    buffer: side
                        .buffer(&bytes)
                        .map_err(|failure| stop(&format!("{at}/arguments"), failure))?,
                    offset,
                    object,
                    mesh,
                }
            }
        });

        let outcome = side
            .mesh_pass(&targets, &steps)
            .map_err(|failure| stop(&format!("{at}/pass"), failure))?;
        // Every call of the pass, by the selector it makes, so a refusal of ours compares against
        // Metal's answer at the call's own site.
        let mut calls = Node::branch("calls", Vec::new());
        for (at, (step, refused)) in steps.iter().zip(&outcome.refused).enumerate() {
            let name = format!("{at}-{}", step.call());
            calls = calls.with(Node::leaf(
                name,
                Observed::value(match refused {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        node = node.with(calls);
        if let Some((at, failure)) =
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
            return Err(Abort { at, failure });
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

        let mut read = Vec::new();
        for (at, target) in targets.iter().enumerate() {
            read.push(
                side.read_texture(*target, [SIDE, SIDE], SIDE * TARGET_TEXEL)
                    .map_err(|failure| {
                        stop(&format!("{}/attachment/{at}", self.row_site(row)), failure)
                    })?,
            );
        }
        let mut attachments = Node::branch("attachments", Vec::new());
        for (at, bytes) in read.iter().enumerate() {
            attachments = attachments.with(Node::leaf(
                at.to_string(),
                Observed::bytes(bytes.clone(), attachment_layout()),
            ));
        }
        node = node.with(attachments);
        node = node.with(Node::leaf(
            "drawn",
            Observed::value(Value::Uint(drawn(&read))),
        ));
        sites.push(node);
        Ok(())
    }
}

/// The layout one attachment's texels are read under: the whole 8x8 of level 0 of slice 0, four
/// 32-bit words a texel, packed, so a diff names the texel's row and the word inside it. There are
/// no bytes outside the texels, so comparing the gaps costs nothing and says so.
fn attachment_layout() -> ByteLayout {
    let words_per_row = SIDE * TARGET_TEXEL / 4;
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
        row_elements: words_per_row,
        row_pitch: SIDE * TARGET_TEXEL,
        rows: SIDE,
        image_pitch: SIDE * SIDE * TARGET_TEXEL,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The texels any attachment wrote a non-zero word into: the probe's `drawn=`, which is what a
/// reader wants before descending into the words.
fn drawn(attachments: &[Vec<u8>]) -> u64 {
    let texel = TARGET_TEXEL as usize;
    let texels = (SIDE * SIDE) as usize;
    (0..texels)
        .filter(|at| {
            attachments.iter().any(|bytes| {
                bytes
                    .get(at * texel..(at + 1) * texel)
                    .is_some_and(|words| words.iter().any(|byte| *byte != 0))
            })
        })
        .count() as u64
}

impl Case for MeshDraws {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row of this family but one declares the notice a device's own memory model
        // provokes for the `Shared` buffers a mesh pass carries. `mesh-object-draws-past-limits`
        // asks for an object grid past the device's limits, so the pass stops before it allocates
        // one: the RTX 5080 announced the notice for every other row of the three mesh families
        // and not for this one, and a declaration no run justifies is one nobody is checking.
        match self.name == PAST_LIMITS {
            true => plain_mesh_notice_allowances(),
            false => mesh_notice_allowances(),
        }
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: self.mesh.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        // What the case feeds: both committed forms of the shader, the pipeline's functions, and
        // every row's parameters and control words in order.
        let mut rows = Vec::new();
        for row in self.rows {
            rows.extend(self.row_site(row).into_bytes());
            rows.push(b'=');
            rows.extend(row.draw.words());
            if let Some(control) = row.control {
                for word in control {
                    rows.extend(word.to_le_bytes());
                }
            }
            rows.push(b'\n');
        }
        let mut functions = Vec::new();
        functions.extend(self.object.unwrap_or("-").as_bytes());
        functions.push(b',');
        functions.extend(self.mesh.as_bytes());
        functions.push(b',');
        functions.extend(FRAGMENT.as_bytes());
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
            Input::bytes(
                "render-targets",
                format!("{ATTACHMENTS}x RGBA32Uint {SIDE}x{SIDE} cleared to zero").as_bytes(),
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
