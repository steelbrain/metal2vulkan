//! Mesh draws in an indirect command buffer: what commands with pipeline states of their own draw.
//!
//! Record mode measures the row on the chip its recording is keyed to.
//!
//! The row is an indirect command buffer of two commands the host encodes, over a pass of two
//! `RGBA32Uint` attachments, 8x8, cleared to zero and stored: command 0 sets `mesh_draws`' object
//! pipeline, binds the control words `(2, 1, 0, 0)` at object buffer 0 and draws 3 threadgroups of
//! 4 object threads and 2 mesh threads; command 1 sets its grid pipeline and draws 12 by 4 threads
//! in mesh threadgroups of 4 by 2. Both states are made with `supportIndirectCommandBuffers`, the
//! buffer inherits neither state nor buffers, and the control buffer is made resident for the
//! object, mesh and fragment stages before the execution.
//!
//! What the row observes is typed: each attachment's whole 8x8 as bytes under a layout that reads
//! them as 32-bit words, the drawn texel count, the calls of the pass, and the command buffer's
//! status and error.
//!
//! The second case here is [`unordered_draws`], the racy row: two commands with no barrier
//! between them, whose Metal answer is a **set** of outcomes with the runs behind each.
//!
//! ## What is **not** a case, and why
//!
//! - **Rows that end Metal's process**: the CPU's mesh draw on a buffer of draws of primitives,
//!   and a pipeline state set on a command of a buffer whose command types hold no draw of its
//!   kind. Metal never returns from the call, so the only Metal fact is "the process ended in this
//!   call", the form [`MetalSource::did_not_return`] records.
//! - **Rows where Metal's GPU faults**: a kernel's mesh draw on a buffer of draws of primitives,
//!   which Metal answers with `status=5`, nothing drawn and no mark written. Every one of them is
//!   encoded by a kernel, so they wait on the vocabulary below.
//! - **Rows a kernel encodes** (`kernel_own`, `enc_own`, `kernel_<kind>_on_<kind>`, `enc_barrier`,
//!   `kernel_reset`): a kernel writes the commands through the buffer's and each pipeline state's
//!   **GPU resource ID**, passed in an argument buffer, with the buffer made resident for writing.
//!   Three calls the [`Side`] vocabulary does not have — the resource ID of an indirect command
//!   buffer, the resource ID of a mesh pipeline state, and a dispatch that makes a resource
//!   resident — each needing an implementation on both sides and a forwarding arm in
//!   `validated::Announcing`. (A resource ID is a device-local 64-bit value and must never enter a
//!   transcript; it is an input to the kernel and nothing else.)

use crate::case::{
    Abort, Case, CommandDraw, Failure, Input, MeshCommand, MeshFunctions, MetalSource, Side,
    TARGET_TEXEL,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};

/// The case's name.
pub const CPU_OWN_COMMANDS: &str = "mesh-commands-own-state";

/// The owned fixture the case runs.
const FIXTURE: &str = "mesh_draws";

/// The pass's side in texels.
const SIDE: u32 = 8;

/// The color attachments the pass holds.
const ATTACHMENTS: u32 = 2;

/// The fragment function both pipelines write their attachments through.
const FRAGMENT: &str = "mesh_draws_fragment";

/// The object function's control words, as the probe wrote them.
const CONTROL: [u32; 4] = [2, 1, 0, 0];

/// Command 0's draw: 3 threadgroups of 4 object threads and 2 mesh threads.
const OBJECT_DRAW: CommandDraw = CommandDraw::Threadgroups {
    groups: [3, 1, 1],
    object: [4, 1, 1],
    mesh: [2, 1, 1],
};

/// Command 1's draw: 12 by 4 threads in mesh threadgroups of 4 by 2.
const GRID_DRAW: CommandDraw = CommandDraw::Threads {
    threads: [12, 4, 1],
    object: [1, 1, 1],
    mesh: [4, 2, 1],
};

/// The row's site name, as the probe named the case.
const ROW: &str = "cpu_own";

/// A case that compares what an indirect command buffer's mesh draws drew with Metal's.
pub struct MeshCommands {
    name: &'static str,
}

/// The case.
pub fn own_state_commands() -> MeshCommands {
    MeshCommands {
        name: CPU_OWN_COMMANDS,
    }
}

impl MeshCommands {
    /// The fixture both sides read.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The object pipeline's functions, and the grid pipeline's.
    fn functions() -> [MeshFunctions<'static>; 2] {
        [
            MeshFunctions {
                object: Some("mesh_draws_object"),
                mesh: "mesh_draws_mesh",
                fragment: FRAGMENT,
                attachments: ATTACHMENTS,
            },
            MeshFunctions {
                object: None,
                mesh: "mesh_draws_grid",
                fragment: FRAGMENT,
                attachments: ATTACHMENTS,
            },
        ]
    }
}

/// The layout one attachment's texels are read under: the whole 8x8 of level 0 of slice 0, four
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

/// The texels any attachment holds a non-zero word in: the probe's `drawn=`.
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

impl Case for MeshCommands {
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
                    entry: "mesh_draws_mesh".to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let control: Vec<u8> = CONTROL.iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut commands = Vec::new();
        for (functions, draw) in Self::functions().iter().zip([OBJECT_DRAW, GRID_DRAW]) {
            commands.extend(functions.object.unwrap_or("-").as_bytes());
            commands.push(b',');
            commands.extend(functions.mesh.as_bytes());
            commands.extend(format!(",{draw:?}\n").into_bytes());
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:object:0", &control),
            Input::bytes(
                "render-targets",
                format!("{ATTACHMENTS}x RGBA32Uint {SIDE}x{SIDE} cleared to zero").as_bytes(),
            ),
            Input::bytes("commands", &commands),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let [object, grid] = Self::functions();
        let object = crate::case::made(
            sites,
            "object-pipeline",
            side.mesh_pipeline_for_commands(library, &object),
        )?;
        let grid = crate::case::made(
            sites,
            "grid-pipeline",
            side.mesh_pipeline_for_commands(library, &grid),
        )?;
        let words: Vec<u8> = CONTROL.iter().flat_map(|word| word.to_le_bytes()).collect();
        let control = crate::case::made(sites, "control", side.buffer(&words))?;
        let commands = crate::case::made(
            sites,
            "command-buffer",
            side.mesh_command_buffer(&[
                MeshCommand {
                    pipeline: object,
                    object_buffers: vec![(control, 0)],
                    mesh_buffers: Vec::new(),
                    draw: OBJECT_DRAW,
                },
                MeshCommand {
                    pipeline: grid,
                    object_buffers: Vec::new(),
                    mesh_buffers: Vec::new(),
                    draw: GRID_DRAW,
                },
            ]),
        )?;
        let mut targets = Vec::new();
        for at in 0..ATTACHMENTS {
            targets.push(
                side.render_target(SIDE, SIDE)
                    .map_err(|failure| stop(format!("{ROW}/render_target/{at}"), failure))?,
            );
        }

        let outcome = side
            .execute_mesh_commands(&targets, commands, &[control])
            .map_err(|failure| stop(format!("{ROW}/pass"), failure))?;
        let mut node = Node::branch(ROW, Vec::new());
        // The pass's calls in the order the side makes them: one `useResource` per resident
        // resource, the execution, and the commit.
        let calls: Vec<&str> = ["useResource", "executeCommandsInBuffer", "commit"].to_vec();
        let mut emitted = Node::branch("calls", Vec::new());
        for (index, refused) in outcome.refused.iter().enumerate() {
            let name = calls.get(index).copied().unwrap_or("call");
            emitted = emitted.with(Node::leaf(
                format!("{index}-{name}"),
                Observed::value(match refused {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        node = node.with(emitted);
        if let Some((index, failure)) = outcome
            .refused
            .iter()
            .enumerate()
            .find_map(|(index, refused)| refused.as_ref().map(|failure| (index, failure.clone())))
        {
            let name = calls.get(index).copied().unwrap_or("call");
            sites.push(node);
            return Err(Abort {
                at: format!("{ROW}/calls/{index}-{name}"),
                failure,
            });
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
                    .map_err(|failure| stop(format!("{ROW}/attachments/{at}"), failure))?,
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

/// The racy row: two mesh commands with no barrier between them.
///
/// Command 0's mesh function stores the control words `(3, 2, 1, 7)` and its fragment function
/// covers texel (7, 7) with `0xAAAA`; command 1's object function reads the control words and
/// launches `x` by `y` mesh threadgroups, each covering its own texel. No barrier stands between
/// them, so **nothing in Metal orders the write before the read**.
///
/// ## What is compared, and why it is not the reader's texels
///
/// Measured here, which is the point of the row: over **30 runs** on this M3 Max (macOS 27.0.1),
/// plain Metal produced **one** outcome, every run — the reader read the words the host wrote, so
/// it drew its single threadgroup at (0, 0) with `(0, 0, 1, 0)` and `drawn` was 2. That matches
/// an Apple M4's answer for the row. The runtime builds no mesh pipeline (its README, the known
/// gaps of render pipeline states), so no run of it reaches this case's rows.
///
/// A recording can only sample an order Metal's own contract leaves open, and comparing against
/// one sample would make this chip's scheduling of the moment into a requirement; the outcome set
/// the plan gives racy rows cannot close that, because it can hold only the orders Metal happened
/// to produce. What the contract *does* determine: the writer's own texel at (7, 7), the control
/// words the pass leaves behind, the calls of the pass, and the command buffer's status and error.
///
/// The whole attachment is a site of the transcript, so the recording holds **Metal's own
/// texels** for every one of the 30 runs, and the next macOS is read against them instead of
/// against a memory of them. The case loosens nothing: `docs/API.md` section 10 holds no accepted
/// difference for the order, as the runtime draws no mesh.
pub const UNORDERED_DRAWS: &str = "mesh-commands-unordered-draws";

/// The owned fixture the racy row runs: `mesh_barrier`'s writer and object reader, the same
/// functions `mesh_barriers` uses.
const BARRIER_FIXTURE: &str = "mesh_barrier";

/// The fragment function both of the row's pipelines write their attachment through.
const BARRIER_FRAGMENT: &str = "mesh_barrier_fragment";

/// The control words the host writes before the pass: the reader's own
/// grid if it reads them, `(3, 2, 1, 7)` if it reads the writer's.
const BARRIER_CONTROL: [u32; 4] = [1, 1, 1, 0];

/// One threadgroup of one thread.
const ONE: [u32; 3] = [1, 1, 1];

/// Both commands' draw.
const ONE_DRAW: CommandDraw = CommandDraw::Threadgroups {
    groups: ONE,
    object: ONE,
    mesh: ONE,
};

/// The row's site name.
const UNORDERED_ROW: &str = "cpu_nobarrier";

/// The texel the writer's fragment function covers, whichever order the two draws run in.
const COVER: [u32; 2] = [7, 7];

/// The site the whole attachment is recorded at, under the row.
const ATTACHMENT_SITE: &str = "attachment";

/// The layout the writer's own texel is read under: the one texel at [`COVER`] of level 0 of
/// slice 0, as four 32-bit words, so a diff names the word inside the texel the writer wrote.
fn cover_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [COVER[0], COVER[1], 0],
            size: [1, 1, 1],
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: TARGET_TEXEL / 4,
        row_pitch: TARGET_TEXEL,
        rows: 1,
        image_pitch: TARGET_TEXEL,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The layout the whole attachment is read under: level 0 of slice 0, 8 by 8 texels of four
/// 32-bit words each, rows adjacent, so a diff names the texel and the word inside it.
fn whole_layout() -> ByteLayout {
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

/// The runs the racy row's recording stands on.
///
/// Two draws of one pass with no barrier between them are ordered by nothing, so one run says
/// nothing about what Metal does: the recording has to hold the **set** of orders Metal produced
/// and how often each. Thirty runs is what this case asks record mode for, and the recording
/// states the frequencies it actually saw — never a guess at what a longer run would find.
const UNORDERED_RUNS: u32 = 30;

/// A case that compares what two unordered mesh commands drew with the orders Metal produced.
pub struct UnorderedDraws {
    name: &'static str,
}

/// The case.
pub fn unordered_draws() -> UnorderedDraws {
    UnorderedDraws {
        name: UNORDERED_DRAWS,
    }
}

impl UnorderedDraws {
    /// The fixture both sides read.
    fn fixture(&self) -> Fixture {
        Fixture::new(BARRIER_FIXTURE)
    }

    /// The writer's pipeline functions, and the object reader's.
    fn functions() -> [MeshFunctions<'static>; 2] {
        [
            MeshFunctions {
                object: None,
                mesh: "mesh_barrier_writer",
                fragment: BARRIER_FRAGMENT,
                attachments: 1,
            },
            MeshFunctions {
                object: Some("mesh_barrier_object"),
                mesh: "mesh_barrier_mesh",
                fragment: BARRIER_FRAGMENT,
                attachments: 1,
            },
        ]
    }
}

impl Case for UnorderedDraws {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        UNORDERED_RUNS
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{BARRIER_FIXTURE}{extension}"),
                    entry: BARRIER_FRAGMENT.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let control: Vec<u8> = BARRIER_CONTROL
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let mut commands = Vec::new();
        for functions in Self::functions() {
            commands.extend(functions.object.unwrap_or("-").as_bytes());
            commands.push(b',');
            commands.extend(functions.mesh.as_bytes());
            commands.extend(format!(",{ONE_DRAW:?},no-barrier\n").into_bytes());
        }
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
            Input::bytes("commands", &commands),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let [writes, reads] = Self::functions();
        let writes = crate::case::made(
            sites,
            "writer-pipeline",
            side.mesh_pipeline_for_commands(library, &writes),
        )?;
        let reads = crate::case::made(
            sites,
            "reader-pipeline",
            side.mesh_pipeline_for_commands(library, &reads),
        )?;
        let words: Vec<u8> = BARRIER_CONTROL
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let control = crate::case::made(sites, "control", side.buffer(&words))?;
        // Command 0 stores the control words from its mesh function; command 1 reads them in its
        // object function and launches that many mesh threadgroups. Nothing orders the two.
        let commands = crate::case::made(
            sites,
            "command-buffer",
            side.mesh_command_buffer(&[
                MeshCommand {
                    pipeline: writes,
                    object_buffers: Vec::new(),
                    mesh_buffers: vec![(control, 0)],
                    draw: ONE_DRAW,
                },
                MeshCommand {
                    pipeline: reads,
                    object_buffers: vec![(control, 0)],
                    mesh_buffers: Vec::new(),
                    draw: ONE_DRAW,
                },
            ]),
        )?;
        let target = side
            .render_target(SIDE, SIDE)
            .map_err(|failure| stop(format!("{UNORDERED_ROW}/render_target"), failure))?;

        let outcome = side
            .execute_mesh_commands(&[target], commands, &[control])
            .map_err(|failure| stop(format!("{UNORDERED_ROW}/pass"), failure))?;
        let mut node = Node::branch(UNORDERED_ROW, Vec::new());
        let calls: Vec<&str> = ["useResource", "executeCommandsInBuffer", "commit"].to_vec();
        let mut emitted = Node::branch("calls", Vec::new());
        for (index, refused) in outcome.refused.iter().enumerate() {
            let name = calls.get(index).copied().unwrap_or("call");
            emitted = emitted.with(Node::leaf(
                format!("{index}-{name}"),
                Observed::value(match refused {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        node = node.with(emitted);
        if let Some((index, failure)) = outcome
            .refused
            .iter()
            .enumerate()
            .find_map(|(index, refused)| refused.as_ref().map(|failure| (index, failure.clone())))
        {
            let name = calls.get(index).copied().unwrap_or("call");
            sites.push(node);
            return Err(Abort {
                at: format!("{UNORDERED_ROW}/calls/{index}-{name}"),
                failure,
            });
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
            .map_err(|failure| stop(format!("{UNORDERED_ROW}/attachment"), failure))?;
        // Only the writer's own texel is compared; the whole level is a site of its own, loosened
        // by the accepted difference the order rests on, so the recording keeps Metal's texels
        // without this chip's scheduling becoming a requirement. See the case's documentation.
        // The whole level is read, as the probe read it, and the texel is named by its origin so
        // a reader of a diff sees which texel this is.
        let at = (COVER[1] * SIDE + COVER[0]) as usize * TARGET_TEXEL as usize;
        let cover = texels
            .get(at..at + TARGET_TEXEL as usize)
            .ok_or_else(|| {
                stop(
                    format!("{UNORDERED_ROW}/cover"),
                    Failure::named(
                        "ShortReadBack",
                        format!("{} bytes hold no texel at {COVER:?}", texels.len()),
                    ),
                )
            })?
            .to_vec();
        node = node.with(Node::leaf("cover", Observed::bytes(cover, cover_layout())));
        node = node.with(Node::leaf(
            ATTACHMENT_SITE,
            Observed::bytes(texels, whole_layout()),
        ));
        let after = side
            .read_buffer(control, 0, words.len())
            .map_err(|failure| stop(format!("{UNORDERED_ROW}/control-after"), failure))?;
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
