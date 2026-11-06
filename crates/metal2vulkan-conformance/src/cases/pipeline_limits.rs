//! What a mesh or compute pipeline's creation answers for a descriptor's limits: refused, or made
//! and then what the state's limit getters answer ([`crate::pipeline_limits`]).
//!
//! The mesh cases are the rows of `probes/mesh_pipeline_limits.m` that Metal answers, over the
//! owned fixture `mesh_pipeline_limits` (the probe's functions): the descriptor's thread maxima,
//! mesh-grid maximum, required sizes, payload length and amplification count, alone and beside
//! the functions' own maxima. The compute cases are the rows of `probes/compute_pipeline_limits.m`
//! over the owned fixture `compute_thread_limit`: a descriptor's `maxTotalThreadsPerThreadgroup`
//! for a kernel without a maximum and for one of `[[max_total_threads_per_threadgroup(64)]]`. The
//! probes' rows whose descriptor validation ends the process (a required size with an axis of 0,
//! of more than 1024 threads or more than its stage's maximum, an object size without an object
//! function, an amplification count of 0) are not cases: the translator's tests hold the runtime's
//! typed refusals of them.
//!
//! # Why this family is safe
//!
//! Each case makes one library and one pipeline, rasterization disabled, and draws nothing: no
//! resource, command buffer or queue.

use crate::case::{Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::pipeline_limits::{ComputeLimitsSpec, MeshLimitsSpec};
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites};

/// The site the library's make sits at.
const LIBRARY: &str = "library";

/// The site the creation's answer sits at.
const CREATED: &str = "created";

/// The mesh cases' fixture and functions.
const MESH_FIXTURE: &str = "mesh_pipeline_limits";
const OBJECT: &str = "mesh_pipeline_limits_object";
const OBJECT_64: &str = "mesh_pipeline_limits_object_64";
const OBJECT_GRID_4: &str = "mesh_pipeline_limits_object_grid_4";
const MESH: &str = "mesh_pipeline_limits_mesh";
const MESH_64: &str = "mesh_pipeline_limits_mesh_64";
const ALONE: &str = "mesh_pipeline_limits_mesh_alone";

/// The compute cases' fixture and kernels.
const COMPUTE_FIXTURE: &str = "compute_thread_limit";
const PLAIN: &str = "compute_thread_limit_plain";
const KERNEL_64: &str = "compute_thread_limit_64";

/// What one case makes.
#[derive(Clone, Copy, Debug)]
enum Made {
    Mesh(MeshLimitsSpec<'static>),
    Compute(ComputeLimitsSpec<'static>),
}

/// A case over one creation's answer.
pub struct PipelineLimitsCase {
    name: &'static str,
    made: Made,
}

/// The descriptor of `object` and `mesh`, changed by `with`.
const fn mesh(
    object: Option<&'static str>,
    mesh: &'static str,
    max: [u32; 3],
    payload: u32,
    amplification: u32,
    required: [[u32; 3]; 2],
) -> Made {
    let mut spec = MeshLimitsSpec::new(object, mesh);
    spec.max_object = max[0];
    spec.max_mesh = max[1];
    spec.max_grid = max[2];
    spec.payload = payload;
    spec.amplification = amplification;
    spec.required_object = required[0];
    spec.required_mesh = required[1];
    Made::Mesh(spec)
}

const Z: [u32; 3] = [0; 3];
const NONE: [[u32; 3]; 2] = [Z, Z];

/// Every case of the family: the probe's name for its row, after the family's prefix.
pub fn all() -> Vec<PipelineLimitsCase> {
    let o = Some(OBJECT);
    let o64 = Some(OBJECT_64);
    let grid4 = Some(OBJECT_GRID_4);
    let rows: [(&'static str, Made); 33] = [
        // Made: Metal's defaults, and each limit alone.
        ("mesh-limits-plain", mesh(o, MESH, [0, 0, 0], 0, 1, NONE)),
        (
            "mesh-limits-no-object",
            mesh(None, ALONE, [0, 0, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-max-64-32",
            mesh(o, MESH, [64, 32, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-attribute-64",
            mesh(o64, MESH, [0, 0, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-attribute-64-max-64",
            mesh(o64, MESH, [64, 0, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-required-object-32-attribute-64",
            mesh(o64, MESH, [0, 0, 0], 0, 1, [[32, 1, 1], Z]),
        ),
        (
            "mesh-limits-mesh-attribute-64",
            mesh(o, MESH_64, [0, 0, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-mesh-attribute-64-max-64",
            mesh(o, MESH_64, [0, 64, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-required-mesh-32-attribute-64",
            mesh(o, MESH_64, [0, 0, 0], 0, 1, [Z, [32, 1, 1]]),
        ),
        ("mesh-limits-grid-8", mesh(o, MESH, [0, 0, 8], 0, 1, NONE)),
        (
            "mesh-limits-grid-2097152",
            mesh(o, MESH, [0, 0, 2_097_152], 0, 1, NONE),
        ),
        (
            "mesh-limits-grid-8-no-object",
            mesh(None, ALONE, [0, 0, 8], 0, 1, NONE),
        ),
        (
            "mesh-limits-grid-attribute-4",
            mesh(grid4, MESH, [0, 0, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-grid-attribute-4-grid-8",
            mesh(grid4, MESH, [0, 0, 8], 0, 1, NONE),
        ),
        (
            "mesh-limits-grid-attribute-4-grid-2",
            mesh(grid4, MESH, [0, 0, 2], 0, 1, NONE),
        ),
        (
            "mesh-limits-payload-64",
            mesh(o, MESH, [0, 0, 0], 64, 1, NONE),
        ),
        (
            "mesh-limits-payload-16384",
            mesh(o, MESH, [0, 0, 0], 16384, 1, NONE),
        ),
        (
            "mesh-limits-amplification-2",
            mesh(o, MESH, [0, 0, 0], 0, 2, NONE),
        ),
        (
            "mesh-limits-required-object-32x2",
            mesh(o, MESH, [0, 0, 0], 0, 1, [[32, 2, 1], Z]),
        ),
        (
            "mesh-limits-required-mesh-16x2",
            mesh(o, MESH, [0, 0, 0], 0, 1, [Z, [16, 2, 1]]),
        ),
        (
            "mesh-limits-required-object-32-max-64",
            mesh(o, MESH, [64, 0, 0], 0, 1, [[32, 1, 1], Z]),
        ),
        (
            "mesh-limits-required-object-64-max-64",
            mesh(o, MESH, [64, 0, 0], 0, 1, [[64, 1, 1], Z]),
        ),
        (
            "mesh-limits-required-mesh-64-max-64",
            mesh(o, MESH, [0, 64, 0], 0, 1, [Z, [64, 1, 1]]),
        ),
        // Refused: a maximum past 1024, a function's maximum beside another of the
        // descriptor's, a mesh stage's required size of fewer threads than its maximum, and a
        // payload past 16384 bytes.
        (
            "mesh-limits-max-object-2048",
            mesh(o, MESH, [2048, 0, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-max-mesh-2048",
            mesh(o, MESH, [0, 2048, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-max-mesh-2048-no-object",
            mesh(None, ALONE, [0, 2048, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-attribute-64-max-32",
            mesh(o64, MESH, [32, 0, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-attribute-64-max-128",
            mesh(o64, MESH, [128, 0, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-mesh-attribute-64-max-32",
            mesh(o, MESH_64, [0, 32, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-mesh-attribute-64-max-128",
            mesh(o, MESH_64, [0, 128, 0], 0, 1, NONE),
        ),
        (
            "mesh-limits-required-mesh-32-max-64",
            mesh(o, MESH, [0, 64, 0], 0, 1, [Z, [32, 1, 1]]),
        ),
        (
            "mesh-limits-payload-16385",
            mesh(o, MESH, [0, 0, 0], 16385, 1, NONE),
        ),
        (
            "mesh-limits-payload-16385-no-object",
            mesh(None, ALONE, [0, 0, 0], 16385, 1, NONE),
        ),
    ];
    let compute = |entry, max_total_threads| {
        Made::Compute(ComputeLimitsSpec {
            entry,
            max_total_threads,
        })
    };
    let kernels: [(&'static str, Made); 10] = [
        ("compute-limits-plain", compute(PLAIN, 0)),
        ("compute-limits-plain-max-96", compute(PLAIN, 96)),
        ("compute-limits-plain-max-1024", compute(PLAIN, 1024)),
        ("compute-limits-plain-max-1025", compute(PLAIN, 1025)),
        ("compute-limits-plain-max-2048", compute(PLAIN, 2048)),
        ("compute-limits-attribute-64", compute(KERNEL_64, 0)),
        ("compute-limits-attribute-64-max-64", compute(KERNEL_64, 64)),
        ("compute-limits-attribute-64-max-32", compute(KERNEL_64, 32)),
        (
            "compute-limits-attribute-64-max-128",
            compute(KERNEL_64, 128),
        ),
        (
            "compute-limits-attribute-64-max-2048",
            compute(KERNEL_64, 2048),
        ),
    ];
    rows.into_iter()
        .chain(kernels)
        .map(|(name, made)| PipelineLimitsCase { name, made })
        .collect()
}

impl PipelineLimitsCase {
    fn fixture(&self) -> Fixture {
        match self.made {
            Made::Mesh(spec) => Fixture {
                name: MESH_FIXTURE,
                entry: spec.object.unwrap_or(spec.mesh),
            },
            Made::Compute(spec) => Fixture {
                name: COMPUTE_FIXTURE,
                entry: spec.entry,
            },
        }
    }

    fn words(&self) -> Vec<u8> {
        match self.made {
            Made::Mesh(spec) => spec.words(),
            Made::Compute(spec) => spec.words(),
        }
    }
}

impl Case for PipelineLimitsCase {
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
                    fixture: format!("{}{extension}", fixture.name),
                    entry: fixture.entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("descriptor", &self.words()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self.fixture().loaded().map_err(|why| stop(LIBRARY, why))?;
        let library = crate::case::made(sites, LIBRARY, side.library(&loaded.source()))?;
        let answered = match self.made {
            Made::Mesh(spec) => side
                .mesh_pipeline_limits(library, &spec)
                .map(|created| created.node(CREATED)),
            Made::Compute(spec) => side
                .compute_pipeline_limits(library, &spec)
                .map(|created| created.node(CREATED)),
        };
        match answered {
            Ok(node) => sites.push(node),
            Err(failure) => {
                sites.push(Node::leaf(CREATED, Observed::value(failure.recorded())));
                return Err(Abort {
                    at: CREATED.to_owned(),
                    failure,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No two cases share a name or a descriptor, so no recording of one is read against another.
    #[test]
    fn every_case_states_its_own_descriptor() {
        let cases = all();
        let names: std::collections::BTreeSet<&str> = cases.iter().map(|c| c.name).collect();
        let words: std::collections::BTreeSet<Vec<u8>> = cases.iter().map(|c| c.words()).collect();
        assert_eq!(names.len(), cases.len());
        assert_eq!(words.len(), cases.len());
    }
}
