//! What a compute pipeline **descriptor** says about itself, and what `reset` puts back.
//!
//! `MTLComputePipelineDescriptor` was exercised at five of its sixteen items, and every one of the
//! five by a case that made a **state** from the descriptor and asked the state
//! ([`crate::cases::pipeline_facts`]) or that dispatched with it: the label, the compute function,
//! the stage-input descriptor, the indirect-command support and the required threadgroup size. The
//! other eleven are what a program does **not** set — so what Metal's defaults decide — plus the
//! per-buffer mutability, the deprecated spelling of the preloaded libraries, the lists of supplied
//! objects, and `reset`.
//!
//! The runtime holds each of those defaults as a literal in
//! [`ComputePipelineDescriptor::new`](metal2vulkan::pipeline::ComputePipelineDescriptor::new), read
//! off Apple's documentation, and nothing in the tree ran Metal to check one: the call-stack depth
//! of 1, the absent thread limit, the absent required threadgroup size, the empty buffer map, the
//! three flags off. [`crate::compute_descriptors`] says which claim each getter is; this family is
//! the three runs that hold them.
//!
//! # The shapes, and why each is here
//!
//! - **the descriptor of a kernel alone** (`compute-descriptor-defaults`): Metal's default for
//!   every other field, which is the one shape that can say what a program gets for free;
//! - **the descriptor with every field stated** (`compute-descriptor-set`): each field away from
//!   its default and no two stated numbers equal
//!   ([`crate::compute_descriptors::AWAY`]), read straight back. This is where an enumeration whose
//!   ordinals we numbered differently from Metal reads wrong rather than passing, and where the
//!   deprecated `insertLibraries` is shown to be the same property as `preloadedLibraries`;
//! - **the same descriptor after `reset`** (`compute-descriptor-reset`): every field stated, then
//!   `reset`, then the compute function set again. Its answer must be the first shape's, which is
//!   the only way to say "a reset descriptor is a fresh one" with both shapes in one recording. It
//!   reads every field the first shape does **except** the three lists of supplied objects, and
//!   that exclusion is itself a measurement: Metal's `reset` leaves `preloadedLibraries` **nil**
//!   where its `init` leaves a non-nil empty array, which `objc2-metal` 0.3.2 declares cannot
//!   happen and panics on ([`crate::compute_descriptors::SuppliedLists`]).
//!
//! The runtime has no mutable descriptor, so its `reset` is `ComputePipelineDescriptor::new`
//! again (the coverage map says so), and the case makes our side build the stated descriptor and
//! then replace it — rather than skip the setting, which would leave nothing for the reset to undo.
//!
//! # Exactness
//!
//! Every field is a number, a bool, a name or an ordinal. The family declares no tolerance and
//! announces nothing: no case touches a texture, a sampler, a float or the GPU. The one loosening
//! is `compute-descriptor-reset`'s three thread-shape facts, which Metal's `reset` keeps on the M3
//! Max where the runtime's fresh descriptor answers the defaults its header documents: each may be
//! that default alone, under the accepted difference `docs/API.md` section 10 carries for it
//! ([`RESET_IS_ACCEPTED`]).
//!
//! Four items are compared **at their default alone** — `binaryArchives`, the two spellings of the
//! preloaded libraries and `linkedFunctions` — because the runtime takes each of them somewhere
//! other than the descriptor and a count is the one shape both sides can state. The three per-stage
//! ones are compared on the first two shapes and not the third.
//! [`crate::compute_descriptors`] gives the reasoning; nothing is widened for it.
//!
//! # Why this family is safe
//!
//! No shape compiles a pipeline, allocates a resource, makes a command buffer or reaches a queue:
//! each makes one library, one function of it, and a descriptor it reads and drops. The one setter
//! of Metal's that can end a process here is `setLabel:` with nil, which no shape sends.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{Abort, Case, Failure, Input, MetalSource, Side};
use crate::compute_descriptors::DescriptorSpec;
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render_descriptors::Shape;
use crate::transcript::{Digest, Node, Observed, Sites, Value};

/// The site the library's make sits at.
const LIBRARY: &str = "library";

/// The site the descriptor's own answers sit at.
const FACTS: &str = "facts";

/// The fixture whose kernel takes a `threadgroup uint *`: `function_bindings`' own, which
/// [`crate::cases::pipeline_facts`] also borrows, so this family needs no fixture of its own.
const FIXTURE: &str = "a27_ob1_threadgroup";
const KERNEL: &str = "ob1_threadgroup";

/// The bold lead-in of the accepted difference in `docs/API.md` section 10 that a reset
/// descriptor's thread shape rests on: the runtime's `reset` is a fresh descriptor, as
/// `MTLComputePipeline.h` documents it, where Metal's on the M3 Max (macOS 27.0.1) keeps the three
/// thread-shape properties the program set. The slug of these words must be an entry of the
/// document or the case fails.
pub const RESET_IS_ACCEPTED: &str = "A reset compute pipeline descriptor is a fresh one";

/// The thread-shape facts Metal's `reset` keeps, under [`FACTS`], with the default the runtime's
/// fresh descriptor answers at each.
fn kept_by_metals_reset() -> [(&'static str, Value); 5] {
    [
        (
            "threadgroup_size_is_multiple_of_execution_width",
            Value::Bool(false),
        ),
        ("max_total_threads_per_threadgroup", Value::Uint(0)),
        ("required_threads_per_threadgroup/width", Value::Uint(0)),
        ("required_threads_per_threadgroup/height", Value::Uint(0)),
        ("required_threads_per_threadgroup/depth", Value::Uint(0)),
    ]
}

/// A case over one compute pipeline descriptor's own facts.
pub struct ComputeDescriptorCase {
    name: &'static str,
    spec: DescriptorSpec<'static>,
}

/// Every case of the family.
pub fn all() -> Vec<ComputeDescriptorCase> {
    vec![
        // The floor: a descriptor carrying one kernel. Every other field is Metal's own default,
        // and no case before this one asked Metal for any of them.
        ComputeDescriptorCase {
            name: "compute-descriptor-defaults",
            spec: DescriptorSpec::new(KERNEL, Shape::Defaults),
        },
        // Every field stated, each away from its default, read straight back: the round trip that
        // a getter wired to a neighbour's field fails.
        ComputeDescriptorCase {
            name: "compute-descriptor-set",
            spec: DescriptorSpec::new(KERNEL, Shape::Set),
        },
        // The same, then `reset`, then the kernel again: the answer must be the first case's,
        // field for field.
        ComputeDescriptorCase {
            name: "compute-descriptor-reset",
            spec: DescriptorSpec::new(KERNEL, Shape::Reset),
        },
    ]
}

impl ComputeDescriptorCase {
    fn loaded(&self) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: KERNEL,
        }
    }
}

impl Case for ComputeDescriptorCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        if !matches!(self.spec.shape, Shape::Reset) {
            return Vec::new();
        }
        let justified = Justification::accepted(crate::allowance::accepted_slug(RESET_IS_ACCEPTED));
        kept_by_metals_reset()
            .into_iter()
            .map(|(site, default)| {
                Allowance::new(
                    format!("case/{FACTS}/{site}"),
                    Loosening::AnyOf(vec![default]),
                    justified.clone(),
                )
            })
            .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.loaded();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: KERNEL.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.loaded();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The spec is the whole of the rest of what the case feeds: the kernel, the shape, and
            // — where the shape states anything — the stated table itself, so a table that moves
            // invalidates every recording of the family rather than passing against an older one.
            Input::bytes("descriptor", &self.spec.words()),
            // The slot bound the facts are read over. A record of three buffer slots compared
            // against a record of two is a different observation, so the bound is an input.
            Input::bytes(
                "slots",
                &crate::render_descriptors::READ_BUFFERS.to_le_bytes(),
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self.loaded().loaded().map_err(|why| stop(LIBRARY, why))?;
        let library = crate::case::made(sites, LIBRARY, side.library(&loaded.source()))?;
        match side.compute_descriptor_facts(library, &self.spec) {
            Ok(facts) => sites.push(facts.node(FACTS)),
            Err(failure) => {
                sites.push(Node::leaf(FACTS, Observed::value(failure.recorded())));
                return Err(Abort {
                    at: FACTS.to_owned(),
                    failure,
                });
            }
        }
        Ok(())
    }
}

/// The stated table holds no two equal numbers among the ones a getter could be crossed with, and
/// every slot it names is inside the bound the facts are read over. Those two properties are what
/// make the set shape a measurement rather than a shape that passes by coincidence.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_descriptors::AWAY;

    /// Every number the set shape states is distinct, so a side answering one getter out of
    /// another's field reads a different value — including a side that read the required
    /// threadgroup's width where its height belongs.
    #[test]
    fn the_stated_numbers_are_all_different() {
        let numbers = [
            AWAY.max_total_threads_per_threadgroup,
            AWAY.max_call_stack_depth,
            AWAY.required_threads_per_threadgroup[0],
            AWAY.required_threads_per_threadgroup[1],
            AWAY.required_threads_per_threadgroup[2],
        ];
        let unique: std::collections::BTreeSet<u32> = numbers.iter().copied().collect();
        assert_eq!(
            unique.len(),
            numbers.len(),
            "two stated numbers are equal, so a crossed getter would pass: {numbers:?}"
        );
    }

    /// The reset case alone loosens anything: each of the three thread-shape facts Metal's `reset`
    /// keeps, to the default the runtime's fresh descriptor answers there, under an entry
    /// `docs/API.md` section 10 holds; the defaults and the stated shapes compare exactly.
    #[test]
    fn only_the_reset_loosens_and_only_the_kept_thread_shape() {
        let entry = crate::allowance::accepted_slug(RESET_IS_ACCEPTED);
        assert!(
            crate::allowance::accepted_differences().contains(&entry),
            "docs/API.md section 10 holds no entry {entry}"
        );
        for case in all() {
            let allowances = case.allowances();
            if !matches!(case.spec.shape, Shape::Reset) {
                assert!(allowances.is_empty(), "{}", case.name());
                continue;
            }
            let paths: Vec<&str> = allowances.iter().map(|a| a.path.as_str()).collect();
            assert_eq!(
                paths,
                [
                    "case/facts/threadgroup_size_is_multiple_of_execution_width",
                    "case/facts/max_total_threads_per_threadgroup",
                    "case/facts/required_threads_per_threadgroup/width",
                    "case/facts/required_threads_per_threadgroup/height",
                    "case/facts/required_threads_per_threadgroup/depth",
                ]
            );
            for allowance in &allowances {
                assert_eq!(
                    allowance.justified_by,
                    Justification::AcceptedDifference(entry.clone()),
                    "{}",
                    allowance.path
                );
                // Metal's kept value is outside it, so the allowance takes the default alone.
                let Loosening::AnyOf(values) = &allowance.loosening else {
                    panic!("{}: {:?}", allowance.path, allowance.loosening);
                };
                assert_eq!(values.len(), 1, "{}", allowance.path);
                assert!(
                    !allowance.within(&values[0], &Value::Uint(96))
                        && !allowance.within(&values[0], &Value::Bool(true)),
                    "{}",
                    allowance.path
                );
            }
        }
    }

    /// Every buffer slot the stated table names is inside the bound the facts are read over, so a
    /// stated mutability always appears in the record.
    #[test]
    fn every_stated_slot_is_read_back() {
        for (index, _) in AWAY.buffers {
            assert!(
                *index < crate::render_descriptors::READ_BUFFERS,
                "buffer {index} is stated and never read"
            );
        }
    }

    /// The three shapes fingerprint differently, so no recording of one is read against another.
    #[test]
    fn the_three_shapes_have_three_fingerprints() {
        let words: std::collections::BTreeSet<Vec<u8>> =
            all().iter().map(|case| case.spec.words()).collect();
        assert_eq!(words.len(), all().len());
    }
}
