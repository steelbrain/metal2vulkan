//! Object and mesh cases over the owned fixtures `engine/mesh_draws.ll` and
//! `engine/mesh_barrier.ll`, each with what Metal drew for it: the words of Metal's own probe of
//! the same functions (`runtime_lane::mesh_draws`'s answers, measured on the M3 Max and an M4
//! alike), each mesh threadgroup covering one pixel of an 8x8 pass of two `RGBA32Uint` targets
//! with the builtins it read, and the device write the M3 Max measured with rasterization off. A
//! mesh case observed through the generated observer fragment reads what the fixture's own
//! fragment function writes to its first target, so Metal's answer is the same words. An object
//! case observed through its launch draws the rows its generated mesh function draws
//! ([`crate::case::MeshObservation::Launch`]), and a mesh case under a generated object function
//! draws the payload that function writes ([`crate::case::ObjectLaunch`]): those words follow from
//! the fixture's source, and the Metal executor's test holds them to what the M3 Max drew.
//! The Metal executor's tests (`metal::tests::mesh_draws`) hold the oracle to them, and the engine
//! candidate's (`candidate::engine::runtime_run::tests`) hold the runtime's mesh path to them.

use crate::case::{AuthoredCase, Stage};
use crate::source::SourceRow;
use base64::Engine as _;

/// An owned object or mesh case over a committed public row, and the bytes Metal drew for its
/// output.
pub(crate) struct MeshFixtureCase {
    pub case: AuthoredCase,
    pub metal: Vec<u8>,
}

/// The public row of `entry` at `stage` in the owned fixture `engine/<module>.ll`.
fn row(module: &str, stage: Stage, entry: &str) -> SourceRow {
    let label = format!("public/engine/{module}.ll");
    let air_ll = std::fs::read_to_string(
        metal2vulkan_devtools::validation_dir()
            .join("fixtures")
            .join(&label),
    )
    .unwrap();
    SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: stage.metadata_label().into(),
        entry: entry.into(),
        air_ll,
        blob_b64: None,
        lib_sha256s: Vec::new(),
        label,
    }
}

/// A case of `row` from the JSON fields `fields`, its identity computed.
fn case_of(row: &SourceRow, fields: serde_json::Value) -> AuthoredCase {
    let mut json = serde_json::json!({
        "air_sha256": row.air_sha256,
        "case_id": "",
        "name": format!("{}-mesh-draw", row.entry),
        "entry": row.entry,
        "stage": serde_json::to_value(
            if row.stage == Stage::Object.metadata_label() { Stage::Object } else { Stage::Mesh }
        ).unwrap(),
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
    });
    json.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let mut case: AuthoredCase = serde_json::from_value(json).unwrap();
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(case.validate_literal_resources(), Ok(()));
    case
}

/// Two cleared 8x8 `RGBA32Uint` targets, and the selection of target `index`.
fn targets(index: u32) -> serde_json::Value {
    let target = |index| {
        serde_json::json!({
            "index": index,
            "format": "rgba32_uint",
            "dimensions": [8, 8],
            "initial_bytes_b64":
                base64::engine::general_purpose::STANDARD.encode([0u8; 8 * 8 * 16]),
        })
    };
    serde_json::json!({
        "render_targets": [target(0), target(1)],
        "output": {"kind": "render_target", "index": index, "origin": [0, 0], "dimensions": [8, 8]},
    })
}

/// The 8x8 target words Metal's probe drew for `pixels`, each `(x, y, words)`.
fn drawn(pixels: &[(usize, usize, [u32; 4])]) -> Vec<u8> {
    let mut words = vec![0u32; 8 * 8 * 4];
    for &(x, y, texel) in pixels {
        words[4 * (8 * y + x)..][..4].copy_from_slice(&texel);
    }
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A mesh function without an object function, drawn as `drawMeshThreadgroups:` of 3 by 2
/// threadgroups of 4 threads: each pixel is its threadgroup's position, its first target's words
/// that position and the grid's threadgroups (the probe's `grid-groups-3x2-by-4`).
pub(crate) fn grid() -> MeshFixtureCase {
    let row = row("mesh_draws", Stage::Mesh, "mesh_draws_grid");
    let mut fields = targets(0);
    fields["mesh_draw"] = serde_json::json!({
        "fragment": "mesh_draws_fragment",
        "grid": {"threadgroups": [3, 2, 1]},
        "threads_per_object_threadgroup": [1, 1, 1],
        "threads_per_mesh_threadgroup": [4, 1, 1],
    });
    let case = case_of(&row, fields);
    let pixels = (0..2)
        .flat_map(|y| (0..3).map(move |x| (x, y, [x as u32, y as u32, 3, 2])))
        .collect::<Vec<_>>();
    MeshFixtureCase {
        case,
        metal: drawn(&pixels),
    }
}

/// An object case: `mesh_draws_object` with its `ctl` `(2, 1)` at object buffer 0, under 3 object
/// threadgroups of 4 threads and mesh threadgroups of 2, launches `2 + x` by 1 mesh threadgroups
/// from object threadgroup `x`, each covering its own pixel with the payload's object words (the
/// probe's `object-groups-3-by-4-mesh-2`).
pub(crate) fn object() -> MeshFixtureCase {
    let row = row("mesh_draws", Stage::Object, "mesh_draws_object");
    let mut fields = targets(0);
    fields["buffers"] = serde_json::json!([{
        "binding": 0,
        "role": "input",
        "bytes_b64": base64::engine::general_purpose::STANDARD
            .encode([2u32, 1, 0, 0].map(u32::to_le_bytes).concat()),
    }]);
    fields["mesh_draw"] = serde_json::json!({
        "mesh": "mesh_draws_mesh",
        "fragment": "mesh_draws_fragment",
        "grid": {"threadgroups": [3, 1, 1]},
        "threads_per_object_threadgroup": [4, 1, 1],
        "threads_per_mesh_threadgroup": [2, 1, 1],
    });
    let case = case_of(&row, fields);
    let object = |x: u32| [x, 65539, 1049604, 65548];
    let pixels = [
        (0, 0, object(0)),
        (1, 0, object(0)),
        (0, 2, object(1)),
        (1, 2, object(1)),
        (2, 2, object(1)),
        (0, 4, object(2)),
        (1, 4, object(2)),
        (2, 4, object(2)),
        (3, 4, object(2)),
    ];
    MeshFixtureCase {
        case,
        metal: drawn(&pixels),
    }
}

/// [`grid`] observed through the generated observer fragment in place of its fragment partner: the
/// first primitive data field (`a`, a `uint4` read flat), which `mesh_draws_fragment` writes to
/// its first target, into the one target the observer writes.
pub(crate) fn observed() -> MeshFixtureCase {
    let row = row("mesh_draws", Stage::Mesh, "mesh_draws_grid");
    let mut fields = targets(0);
    fields["render_targets"].as_array_mut().unwrap().truncate(1);
    fields["name"] = "mesh_draws_grid-observed-mesh-draw".into();
    fields["mesh_draw"] = serde_json::json!({
        "grid": {"threadgroups": [3, 2, 1]},
        "threads_per_object_threadgroup": [1, 1, 1],
        "threads_per_mesh_threadgroup": [4, 1, 1],
        "observation": {"kind": "primitive_data", "index": 0},
    });
    MeshFixtureCase {
        case: case_of(&row, fields),
        metal: grid().metal,
    }
}

/// `mesh_draws_object` observed through its launch, in its mesh and fragment partners' places:
/// with its `ctl` `(2, 3)` at object buffer 0, one object threadgroup of 4 threads launches 2 by 3
/// mesh threadgroups of 2, each drawing row `x + 2y` of one 8x8 target: the grid `(2, 3, 1)` and
/// the row, then the payload's two `uint4`s (the object words of position 0 in a grid of 1, 4
/// threads a threadgroup and a grid of 4 threads, and the grid words `(0, 2, 3, 0)`). Rows 6 and
/// 7, and every pixel past the payload, keep the target's zeros.
pub(crate) fn launched() -> MeshFixtureCase {
    let row = row("mesh_draws", Stage::Object, "mesh_draws_object");
    let mut fields = targets(0);
    fields["render_targets"].as_array_mut().unwrap().truncate(1);
    fields["name"] = "mesh_draws_object-launched-mesh-draw".into();
    fields["buffers"] = serde_json::json!([{
        "binding": 0,
        "role": "input",
        "bytes_b64": base64::engine::general_purpose::STANDARD
            .encode([2u32, 3, 0, 0].map(u32::to_le_bytes).concat()),
    }]);
    fields["mesh_draw"] = serde_json::json!({
        "grid": {"threadgroups": [1, 1, 1]},
        "threads_per_object_threadgroup": [4, 1, 1],
        "threads_per_mesh_threadgroup": [2, 1, 1],
        "observation": {"kind": "launch"},
    });
    let pixels = (0..6u32)
        .flat_map(|row| {
            [
                (0, row as usize, [2, 3, 1, row]),
                (1, row as usize, [0, 65537, 1049604, 65540]),
                (2, row as usize, [0, 2, 3, 0]),
            ]
        })
        .collect::<Vec<_>>();
    MeshFixtureCase {
        case: case_of(&row, fields),
        metal: drawn(&pixels),
    }
}

/// `mesh_draws_mesh` under a generated object function in its object partner's place, with its
/// fragment partner: one object threadgroup writes the payload `object` `(11, 12, 13, 14)` and
/// `grid.x` 1, and launches 2 by 1 mesh threadgroups of 2 threads, each covering pixel
/// `(x, 2)` (`grid.x * 2 + y`) with the payload's object words in the first target.
pub(crate) fn payload() -> MeshFixtureCase {
    let row = row("mesh_draws", Stage::Mesh, "mesh_draws_mesh");
    let mut fields = targets(0);
    fields["name"] = "mesh_draws_mesh-payload-mesh-draw".into();
    fields["mesh_draw"] = serde_json::json!({
        "fragment": "mesh_draws_fragment",
        "grid": {"threadgroups": [1, 1, 1]},
        "threads_per_object_threadgroup": [1, 1, 1],
        "threads_per_mesh_threadgroup": [2, 1, 1],
        "object_launch": {
            "payload_b64": base64::engine::general_purpose::STANDARD
                .encode([11u32, 12, 13, 14, 1, 0, 0, 0].map(u32::to_le_bytes).concat()),
            "mesh_threadgroups": [2, 1, 1],
        },
    });
    MeshFixtureCase {
        case: case_of(&row, fields),
        metal: drawn(&[(0, 2, [11, 12, 13, 14]), (1, 2, [11, 12, 13, 14])]),
    }
}

/// One cleared 2x2 target of `format` (`rgba32_uint` or `rgba32_float`, 16 bytes a texel), its
/// whole selected, under `name`, and an object launch of `payload` launching `mesh_threadgroups`
/// for a draw of one object threadgroup of one thread, of `threads` mesh threads, observing
/// `observation`.
fn observed_launch(
    name: &str,
    format: &str,
    payload: &[u8],
    mesh_threadgroups: [u32; 3],
    threads: u32,
    observation: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "render_targets": [{
            "index": 0,
            "format": format,
            "dimensions": [2, 2],
            "initial_bytes_b64": base64::engine::general_purpose::STANDARD.encode([0u8; 2 * 2 * 16]),
        }],
        "output": {"kind": "render_target", "index": 0, "origin": [0, 0], "dimensions": [2, 2]},
        "mesh_draw": {
            "grid": {"threadgroups": [1, 1, 1]},
            "threads_per_object_threadgroup": [1, 1, 1],
            "threads_per_mesh_threadgroup": [threads, 1, 1],
            "observation": observation,
            "object_launch": {
                "payload_b64": base64::engine::general_purpose::STANDARD.encode(payload),
                "mesh_threadgroups": mesh_threadgroups,
            },
        },
    })
}

/// `mesh_barrier_mesh` under a generated object function whose payload `ctl` is `(0, 0, 5, 9)`,
/// launching 3 by 2 mesh threadgroups, observed through its primitive data field `a` (both stages
/// generated): each threadgroup covers pixel `(x, y)` of one 8x8 target with `(x, y, 5, 9)`.
pub(crate) fn payload_barrier() -> MeshFixtureCase {
    let row = row("mesh_barrier", Stage::Mesh, "mesh_barrier_mesh");
    let mut fields = targets(0);
    fields["render_targets"].as_array_mut().unwrap().truncate(1);
    fields["name"] = "mesh_barrier_mesh-payload-mesh-draw".into();
    fields["mesh_draw"] = serde_json::json!({
        "grid": {"threadgroups": [1, 1, 1]},
        "threads_per_object_threadgroup": [1, 1, 1],
        "threads_per_mesh_threadgroup": [1, 1, 1],
        "observation": {"kind": "primitive_data", "index": 0},
        "object_launch": {
            "payload_b64": base64::engine::general_purpose::STANDARD
                .encode([0u32, 0, 5, 9].map(u32::to_le_bytes).concat()),
            "mesh_threadgroups": [3, 2, 1],
        },
    });
    let pixels = (0..2u32)
        .flat_map(|y| (0..3u32).map(move |x| (x as usize, y as usize, [x, y, 5, 9])))
        .collect::<Vec<_>>();
    MeshFixtureCase {
        case: case_of(&row, fields),
        metal: drawn(&pixels),
    }
}

/// `mesh_order_mesh` under a generated object function whose payload is `base` 16 and `scale` 2,
/// launching one mesh threadgroup of 4 threads, with its `ctl` at mesh buffer 0 writing one
/// primitive slot (one full-screen triangle, its three indices written, none culled), observed
/// through its primitive data field `id`: `16 * 2 + 4 * 0 + 0 + 1`, 33, over every pixel of a 2x2
/// target, as `(33, 0, 0, 1)`.
///
/// Drawn on a GPU only: its mesh function aborts the CPU ICD's compiler (docs/FAST-PATHS.md's
/// known gaps).
#[cfg(m2v_gpu)]
pub(crate) fn payload_order() -> MeshFixtureCase {
    let row = row("mesh_order", Stage::Mesh, "mesh_order_mesh");
    let mut fields = observed_launch(
        "mesh_order_mesh-payload-mesh-draw",
        "rgba32_uint",
        &[16u32, 2].map(u32::to_le_bytes).concat(),
        [1, 1, 1],
        4,
        serde_json::json!({"kind": "primitive_data", "index": 0}),
    );
    // nprims, count, cullmask, idxmask, meshgrid, and three pads.
    fields["buffers"] = serde_json::json!([{
        "binding": 0,
        "role": "input",
        "bytes_b64": base64::engine::general_purpose::STANDARD
            .encode([1u32, 1, 0, 1, 1, 0, 0, 0].map(u32::to_le_bytes).concat()),
    }]);
    MeshFixtureCase {
        case: case_of(&row, fields),
        metal: [33u32, 0, 0, 1]
            .repeat(4)
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect(),
    }
}

/// `mesh_outputs_triangles` under a generated object function whose payload is four `float4`
/// positions, 64 bytes, though the mesh function declares one 16-byte element and indexes past it:
/// `(-1, -1)`, `(3, -1)`, `(-1, 3)` and `(-1, -1)` (z 0.5, w 1). Its 8 threads set vertex `k` to
/// position `k & 3` and three triangles of indices `(1, 2, 3)`, `(4, 1, 2)` and `(3, 4, 1)`: two
/// cover clip space, the third is degenerate. Observed through its primitive data field `pc`, which
/// every primitive sets to `(1, 1)`: every pixel of a 2x2 target is `(1, 1, 0, 1)`. Metal sizes
/// the payload by the type the generated object function declares, so the mesh function's reads
/// past its own element read it.
pub(crate) fn payload_outputs() -> MeshFixtureCase {
    let row = row("mesh_outputs", Stage::Mesh, "mesh_outputs_triangles");
    let positions = [
        [-1.0f32, -1.0, 0.5, 1.0],
        [3.0, -1.0, 0.5, 1.0],
        [-1.0, 3.0, 0.5, 1.0],
        [-1.0, -1.0, 0.5, 1.0],
    ];
    let fields = observed_launch(
        "mesh_outputs_triangles-payload-mesh-draw",
        "rgba32_float",
        &positions
            .iter()
            .flatten()
            .flat_map(|lane| lane.to_le_bytes())
            .collect::<Vec<_>>(),
        [1, 1, 1],
        8,
        serde_json::json!({"kind": "primitive_data", "index": 0}),
    );
    MeshFixtureCase {
        case: case_of(&row, fields),
        metal: [1.0f32, 1.0, 0.0, 1.0]
            .repeat(4)
            .iter()
            .flat_map(|lane| lane.to_le_bytes())
            .collect(),
    }
}

/// A mesh case without rasterization: no fragment function, and the mesh function's device write
/// lands (`mesh_barrier_writer` stores `(3, 2, 1, 7)` to buffer 0), as the M3 Max measured.
pub(crate) fn device_write() -> MeshFixtureCase {
    let row = row("mesh_barrier", Stage::Mesh, "mesh_barrier_writer");
    let case = case_of(
        &row,
        serde_json::json!({
            "buffers": [{
                "binding": 0,
                "role": "output",
                "initial_bytes_b64": base64::engine::general_purpose::STANDARD.encode([0u8; 16]),
            }],
            "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": 16},
            "mesh_draw": {
                "grid": {"threadgroups": [1, 1, 1]},
                "threads_per_object_threadgroup": [1, 1, 1],
                "threads_per_mesh_threadgroup": [1, 1, 1],
                "rasterization": false,
            },
        }),
    );
    MeshFixtureCase {
        case,
        metal: [3u32, 2, 1, 7].map(u32::to_le_bytes).concat(),
    }
}
