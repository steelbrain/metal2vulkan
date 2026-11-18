//! A description's workloads over a metallib run on the old runtime and on the new, from the same
//! starting contents, and their outputs match, where the new runtime does not refuse one with a
//! typed refusal, which the test prints; one of them matches its CPU reference too. The metallib
//! is the authored fixture's kernels in a synthetic container, so the path a described workload
//! takes runs without any kernel the repository does not own. Each function's bitcode is a module
//! holding that kernel alone, as a metallib's is: the translator checks that a metallib function's
//! module declares one entry point.
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};
use metal2vulkan_devtools::scratch::Owned;
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble_text};
use metal2vulkan_harness::plan::seeded_words;
use metal2vulkan_harness::run::{run_each, Harness, Runs, Session};
use metal2vulkan_harness::{authored, description, old, rt};

/// The description: each fill, both storage modes, a binding at an offset, bindings that stay
/// set for the dispatches after, both dispatch calls, `repeat`, both comparisons, a threadgroup
/// memory length, textures of each source with their bytes written by the CPU, a sampler
/// state, a texture copy from a slice of an array into a 2D texture, and a Private texture
/// written from a buffer and read into another around a fill.
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

workload groups kernels.metallib
buffer 16384 private words 13
buffer 64 shared zero
set-buffer 0 0
set-buffer 1 1
threadgroup-length 0 1024   # a word for each of 256 threads
dispatch group_sums threadgroups 16 1 1 256 1 1
check 1 exact

workload texels kernels.metallib
texture 2 70 32 16 1 1 shared plain words 17   # RGBA8Unorm, ShaderRead
buffer 8192 shared zero
set-texture 0 0
set-buffer 0 0
set-bytes 1 4 hex 20000000   # its width, 32
dispatch texels_out threadgroups 2 1 1 16 16 1
check 0 exact

workload heaped kernels.metallib
texture 2 70 32 16 1 1 shared heap words 17
buffer 8192 shared zero
set-texture 0 0
set-buffer 0 0
set-bytes 1 4 hex 20000000
dispatch texels_out threadgroups 2 1 1 16 16 1
check 0 exact

workload over kernels.metallib
texture 2 70 32 16 1 1 shared over-buffer words 17
buffer 8192 shared zero
set-texture 0 0
set-buffer 0 0
set-bytes 1 4 hex 20000000
dispatch texels_out threadgroups 2 1 1 16 16 1
check 0 exact

workload sampled kernels.metallib
texture 2 70 32 32 1 1 managed plain words 19
texture 2 115 16 16 1 3 private plain undefined   # RGBA16Float, ShaderRead | ShaderWrite
buffer 4096 shared zero
set-texture 0 0
set-texture 1 1
set-sampler 0 nearest clamp-to-zero normalized
dispatch sample_down threadgroups 1 1 1 16 16 1
set-texture 0 1
set-buffer 0 0
set-bytes 1 4 hex 10000000   # its width, 16
dispatch texels_out threadgroups 1 1 1 16 16 1
check 0 exact

workload copied kernels.metallib
texture 3 70 32 16 2 1 shared plain words 23   # RGBA8Unorm, a 2D array of 2
texture 2 70 32 16 1 1 shared plain words 17
buffer 8192 shared zero
copy-texture 0 1 4 2 8 6 1 0 20 9   # 8x6 texels at (4, 2) of slice 1 to (20, 9)
set-texture 0 1
set-buffer 0 0
set-bytes 1 4 hex 20000000
dispatch texels_out threadgroups 2 1 1 16 16 1
check 0 exact

workload staged kernels.metallib
texture 2 70 32 16 1 1 private plain undefined   # RGBA8Unorm, written by a copy alone
buffer 4096 shared words 29
buffer 8192 shared zero
buffer 1024 shared words 31
copy-buffer-to-texture 0 0 128 0 32 16 0 0 0 0
fill-buffer 2 4 20 171
copy-texture-to-buffer 0 0 4 2 8 6 2 256 64 0   # 8x6 texels at (4, 2), rows 64 apart
set-texture 0 0
set-buffer 0 1
set-bytes 1 4 hex 20000000
dispatch texels_out threadgroups 2 1 1 16 16 1
check 1 exact
check 2 exact
";

#[test]
fn described_workloads_match_between_the_old_runtime_and_the_new() {
    let air = authored::library_air().unwrap();
    let ll = std::fs::read_to_string(fixtures_dir().join("harness/workloads.ll")).unwrap();
    let functions: Vec<MtlbFunction> = [
        "scale_offset",
        "lcg_step",
        "binomial3x3",
        "reduce_sum",
        "group_sums",
        "texels_out",
        "sample_down",
    ]
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
    let mut sessions: [(&str, Box<dyn Session>); 2] = [
        (
            "old",
            Box::new(Harness::new(old::system_default().unwrap(), &air).unwrap()),
        ),
        (
            "new",
            Box::new(Harness::new(rt::system_default().unwrap(), &air).unwrap()),
        ),
    ];
    let runs = Runs {
        warmup: 1,
        settle: false,
        timed: 2,
    };
    let mut texels = None;
    for workload in &described {
        let plan = workload.plan().unwrap();
        let each = (run_each(&plan, &mut sessions, runs))
            .unwrap_or_else(|why| panic!("{}: {why}", plan.name));
        let first = (each[0].as_ref())
            .unwrap_or_else(|refused| panic!("{}: the old runtime refused: {refused}", plan.name));
        if let Err(refused) = &each[1] {
            println!("{}: REFUSED on the new runtime: {refused}", plan.name);
        }
        // The new runtime makes no heap and no texture over a buffer yet.
        let missing = match plan.name.as_str() {
            "heaped" => Some("makes no heap"),
            "over" => Some("makes no texture over a buffer"),
            _ => None,
        };
        if let Some(missing) = missing {
            let refused = (each[1].as_ref().err()).map(ToString::to_string);
            assert!(
                refused.as_deref().is_some_and(|why| why.contains(missing)),
                "{}: {refused:?}",
                plan.name
            );
        }
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
        if plan.name == "texels" {
            // Each texel's four bytes as `float`s of [0, 1], read back where the CPU wrote them.
            let bytes: Vec<u8> = (seeded_words(17, 512).iter())
                .flat_map(|word| word.to_le_bytes())
                .collect();
            let got: Vec<f32> = (first.outputs[0].as_chunks::<4>().0.iter())
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            assert_eq!(got.len(), bytes.len());
            for (at, (&got, &byte)) in got.iter().zip(&bytes).enumerate() {
                let want = f32::from(byte) / 255.0;
                assert!((got - want).abs() <= 1e-6, "channel {at}: {got} for {want}");
            }
            texels = Some(first.outputs[0].clone());
        }
        if plan.name == "staged" {
            // The texture is buffer 0's first 2048 bytes; buffer 2 its words with 4..24 filled
            // and the region's rows at 256, 64 bytes apart.
            let texture = &seeded_bytes(29, 1024)[..2048];
            let got: Vec<f32> = (first.outputs[0].as_chunks::<4>().0.iter())
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            assert_eq!(got.len(), texture.len());
            for (at, (&got, &byte)) in got.iter().zip(texture).enumerate() {
                let want = f32::from(byte) / 255.0;
                assert!(
                    (got - want).abs() <= 1e-6,
                    "staged: channel {at}: {got} for {want}"
                );
            }
            let mut want = seeded_bytes(31, 256);
            want[4..24].fill(171);
            for y in 0..6 {
                let (from, to) = (((2 + y) * 32 + 4) * 4, 256 + y * 64);
                want[to..to + 32].copy_from_slice(&texture[from..from + 32]);
            }
            assert_eq!(first.outputs[1], want, "staged: buffer 2");
        }
        if plan.name == "copied" {
            // The 2D texture's bytes, with the region of the array's slice 1 copied over them.
            let words = |seed, count| -> Vec<u8> {
                (seeded_words(seed, count).iter())
                    .flat_map(|word| word.to_le_bytes())
                    .collect()
            };
            let (array, mut bytes) = (words(23, 1024), words(17, 512));
            let row = 32 * 4;
            for y in 0..6 {
                let from = 2048 + (2 + y) * row + 4 * 4;
                let to = (9 + y) * row + 20 * 4;
                bytes[to..to + 8 * 4].copy_from_slice(&array[from..from + 8 * 4]);
            }
            let got: Vec<f32> = (first.outputs[0].as_chunks::<4>().0.iter())
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            assert_eq!(got.len(), bytes.len());
            for (at, (&got, &byte)) in got.iter().zip(&bytes).enumerate() {
                let want = f32::from(byte) / 255.0;
                assert!(
                    (got - want).abs() <= 1e-6,
                    "copied: channel {at}: {got} for {want}"
                );
            }
        }
        if plan.name == "heaped" || plan.name == "over" {
            // The same texels as the device's own texture of them, on the old runtime.
            assert_eq!(Some(&first.outputs[0]), texels.as_ref(), "{}", plan.name);
        }
        if plan.name == "groups" {
            // Each threadgroup of 256 threads sums its words through the length the host set.
            let want: Vec<u32> = (seeded_words(13, 4096).chunks(256))
                .map(|words| words.iter().fold(0u32, |sum, &word| sum.wrapping_add(word)))
                .collect();
            let got: Vec<u32> = (first.outputs[0].as_chunks::<4>().0.iter())
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            assert_eq!(got, want);
        }
    }
    assert_eq!(described.len(), 11);
}

/// The bytes of `count` of SplitMix64's words from `seed`, as a description's `words` fill.
fn seeded_bytes(seed: u64, count: usize) -> Vec<u8> {
    (seeded_words(seed, count).iter())
        .flat_map(|word| word.to_le_bytes())
        .collect()
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
