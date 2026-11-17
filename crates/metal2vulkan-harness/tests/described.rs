//! A description's workloads over a metallib run on the old runtime twice, from the same starting
//! contents, and their outputs match; one of them matches its CPU reference too. The metallib is
//! the authored fixture's kernels in a synthetic container, so the path a described workload takes
//! runs without any kernel the repository does not own. Each function's bitcode is a module
//! holding that kernel alone, as a metallib's is: the translator checks that a metallib function's
//! module declares one entry point.
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};
use metal2vulkan_devtools::scratch::Owned;
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble_text};
use metal2vulkan_harness::plan::seeded_words;
use metal2vulkan_harness::run::{compare_outputs, Harness, Runs, Session};
use metal2vulkan_harness::{authored, description, old};

/// The description: each fill, both storage modes, a binding at an offset, bindings that stay
/// set for the dispatches after, both dispatch calls, `repeat`, and both comparisons.
const DESCRIPTION: &str = "\
# Synthetic workloads over the authored kernels.
workload scale kernels.metallib
buffer 65536 private unit 7
buffer 65536 shared zero
set-buffer 0 0
set-buffer 1 1
set-bytes 2 8 hex 000000400000803f   # float2(2, 1)
dispatch scale_offset threadgroups 16 1 1 256 1 1
check 1 float 0

workload chain kernels.metallib
buffer 1024 private words 9
set-buffer 0 0
set-bytes 1 4 words 3
dispatch lcg_step threadgroups 1 1 1 256 1 1 repeat 64
check 0 exact

workload filter kernels.metallib
buffer 65536 private unit 11
buffer 65536 private zero
set-buffer 0 0
set-buffer 1 1
set-bytes 2 8 hex 2800000066000000   # uint2(40, 102): 4080 texels of 4096
dispatch binomial3x3 threads 40 102 1 16 16 1
check 1 float 0.0001

workload reduce kernels.metallib
buffer 16400 private words 5
buffer 16 shared zero
set-buffer 0 0 16
set-buffer 1 1
set-bytes 2 8 hex 0010000000100000   # uint2(4096, 4096)
dispatch reduce_sum threadgroups 1 1 1 256 1 1
check 1 exact
";

#[test]
fn described_workloads_match_between_two_runs_of_the_old_runtime() {
    let air = authored::library_air().unwrap();
    let ll = std::fs::read_to_string(fixtures_dir().join("harness/workloads.ll")).unwrap();
    let functions: Vec<MtlbFunction> = ["scale_offset", "lcg_step", "binomial3x3", "reduce_sum"]
        .map(|name| MtlbFunction {
            name,
            type_code: 2,
            bitcode: wrap(&llvm_assemble_text(&single_entry(&ll, name)).unwrap()),
            constants: Vec::new(),
            public: Vec::new(),
            tags: Vec::new(),
        })
        .into();
    let scratch = Owned::made("harness-described").unwrap();
    std::fs::write(
        scratch.path().join("kernels.metallib"),
        metallib(&functions),
    )
    .unwrap();
    let described = description::parse(DESCRIPTION, scratch.path()).unwrap();
    let mut sessions = [0, 1].map(|_| Harness::new(old::system_default().unwrap(), &air).unwrap());
    let runs = Runs {
        warmup: 1,
        timed: 2,
    };
    for workload in &described {
        let plan = workload.plan().unwrap();
        let [first, second] = sessions.each_mut().map(|session| {
            (session.run(&plan, runs)).unwrap_or_else(|why| panic!("{}: {why}", plan.name))
        });
        compare_outputs(&plan, &first, &second).unwrap();
        assert_eq!(first.gpu_us.len(), 2, "{}", plan.name);
        if plan.name == "chain" {
            // 64 steps of the generator, each with the one increment set before the first.
            let increment = seeded_words(3, 1)[0];
            let want: Vec<u32> = (seeded_words(9, 256).into_iter())
                .map(|word| {
                    (0..64).fold(word, |x, _| {
                        x.wrapping_mul(1_664_525).wrapping_add(increment)
                    })
                })
                .collect();
            let got: Vec<u32> = (first.outputs[0].as_chunks::<4>().0.iter())
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            assert_eq!(got, want);
        }
    }
    assert_eq!(described.len(), 4);
}

/// The fixture's LLVM text `ll` with `kernel` its only entry point: every other kernel's
/// definition, its `!air.kernel` node and the threadgroup globals named after it left out.
fn single_entry(ll: &str, kernel: &str) -> String {
    /// The name of the function `line` defines, where it starts a definition.
    fn defined(line: &str) -> Option<&str> {
        let rest = line.strip_prefix("define ")?;
        Some(rest.split_once('@')?.1.split_once('(')?.0)
    }
    let others: Vec<&str> = (ll.lines().filter_map(defined))
        .filter(|&name| name != kernel)
        .collect();
    let names_other_entry =
        |line: &str| (others.iter()).any(|other| line.contains(&format!("ptr @{other},")));
    let dropped: Vec<&str> = (ll.lines())
        .filter(|line| line.starts_with('!') && names_other_entry(line))
        .filter_map(|line| Some(line.split_once(" = ")?.0))
        .collect();
    let mut out = String::new();
    let mut in_other = false;
    for line in ll.lines() {
        if in_other {
            in_other = line != "}";
            continue;
        }
        if defined(line).is_some_and(|name| name != kernel) {
            in_other = true;
            continue;
        }
        let other_global = line.starts_with('@') && others.iter().any(|other| line.contains(other));
        if other_global || (line.starts_with('!') && names_other_entry(line)) {
            continue;
        }
        match (line.strip_prefix("!air.kernel = !{")).and_then(|list| list.strip_suffix('}')) {
            Some(list) => {
                let kept: Vec<&str> = (list.split(", "))
                    .filter(|node| !dropped.contains(node))
                    .collect();
                out.push_str(&format!("!air.kernel = !{{{}}}\n", kept.join(", ")));
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    assert_eq!(
        out.lines().filter(|line| defined(line).is_some()).count(),
        1,
        "{kernel} alone"
    );
    out
}
