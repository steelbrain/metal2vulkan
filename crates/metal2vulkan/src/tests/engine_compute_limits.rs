//! A compute pipeline descriptor's `maxTotalThreadsPerThreadgroup` beside its kernel's own.
//!
//! The owned fixture `engine/compute_thread_limit` holds a kernel of
//! `[[max_total_threads_per_threadgroup(64)]]` and one without. On every committed profile each
//! descriptor maximum is made or refused as an M3 Max's creation answers it
//! (`compute_pipeline_limits.m`, plain and under API validation alike): one above 1024 is refused for
//! the kernel without a maximum (`ComputePipelineProblem::MaxThreadsOverLimit`), one other than the
//! kernel's own for the other (`MaxThreadsDiffer`, above 1024 too), and the rest are made, the
//! pipeline answering the least of the maxima within the device. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use super::engine_mesh::{device, fixture};
use crate::budget::Budget;
use crate::error::{ComputePipelineProblem as P, DescriptorError, Error};
use crate::library::Library;
use crate::pipeline::descriptor::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use crate::spirv::testing::committed_profiles;

/// Each descriptor maximum of each kernel, and Metal's answer: the state's
/// `maxTotalThreadsPerThreadgroup`, or its creation's error.
const CASES: [(&str, Option<u32>, Result<u32, P>); 11] = [
    ("compute_thread_limit_plain", None, Ok(1024)),
    ("compute_thread_limit_plain", Some(96), Ok(96)),
    ("compute_thread_limit_plain", Some(1024), Ok(1024)),
    (
        "compute_thread_limit_plain",
        Some(1025),
        Err(P::MaxThreadsOverLimit {
            max: 1025,
            limit: 1024,
        }),
    ),
    (
        "compute_thread_limit_plain",
        Some(2048),
        Err(P::MaxThreadsOverLimit {
            max: 2048,
            limit: 1024,
        }),
    ),
    ("compute_thread_limit_64", None, Ok(64)),
    ("compute_thread_limit_64", Some(64), Ok(64)),
    (
        "compute_thread_limit_64",
        Some(32),
        Err(P::MaxThreadsDiffer {
            specified: 64,
            max: 32,
        }),
    ),
    (
        "compute_thread_limit_64",
        Some(128),
        Err(P::MaxThreadsDiffer {
            specified: 64,
            max: 128,
        }),
    ),
    (
        "compute_thread_limit_64",
        Some(2048),
        Err(P::MaxThreadsDiffer {
            specified: 64,
            max: 2048,
        }),
    ),
    ("compute_thread_limit_plain", Some(0), Ok(1024)),
];

/// Every case on every committed profile: a made pipeline answers Metal's maximum
/// (`metal_max_total_threads_per_threadgroup`) and at most it within the device, a refused one
/// Metal's rule.
#[test]
fn a_compute_descriptors_thread_maximum_is_refused_as_metal_refuses_it() {
    let Some(bc) = fixture("compute_thread_limit") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    for (stem, _) in committed_profiles().expect("committed profiles") {
        let dev = device(&stem);
        for (kernel, max, want) in CASES {
            let what = format!("{stem} {kernel} {max:?}");
            let mut d = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(lib.function(kernel).expect("function")),
            ));
            d.max_total_threads_per_threadgroup = max;
            let made = dev.compile_compute_pipeline(&d, &Budget::standard());
            match (made, want) {
                (Ok(p), Ok(threads)) => {
                    let info = &p.metal_info;
                    assert_eq!(
                        info.metal_max_total_threads_per_threadgroup, threads,
                        "{what}"
                    );
                    assert!(
                        info.max_total_threads_per_threadgroup <= threads,
                        "{what}: {info:?}"
                    );
                }
                (
                    Err(Error::InvalidDescriptor(DescriptorError::ComputePipeline { problem })),
                    Err(p),
                ) => {
                    assert_eq!(problem, p, "{what}")
                }
                (got, want) => panic!("{what}: {got:?}, Metal {want:?}", got = got.err()),
            }
        }
    }
}
