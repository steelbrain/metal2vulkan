//! End to end: a kernel that applies one large AIR function twice, with different constants, to two
//! outputs, the shape of corpus row 69a35b6f (`generateColorCubes`, whose pipeline took 7.6 s to
//! create on an M3 Max through MoltenVK and 24 to 27 s on an RTX 5080 Laptop GPU because the driver
//! compiled the function's body twice), translates with the body written once, as a loop of two
//! trips, on every profile, and computes what the body written twice computes. The regression check
//! is the module's shape: the instructions a driver compiles for the entry
//! (`spirv::roll::entry_weight`) are one body's, as for a kernel that applies the function once,
//! where two bodies are above the size that rolls.
//!
//! The body is a chain of `float` multiplies and adds, which the interpreter computes exactly.
//! `llvm-as` assembles the module; without it the test skips visibly.

use crate::device::DevicePolicy;
use crate::lower::select::Plan;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile_kept, link_and_validate, llvm_assemble, meter, profile, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::roll::{entry_weight, ROLL_ABOVE};

/// The `float` operations of the body: two inlined copies are above `ROLL_ABOVE`.
const OPS: usize = 13_000;

/// The operations before the body's first selection; each later stretch is one longer than the
/// one before, and [`halves`] picks each operation, so no stretch of the body repeats another
/// (only the applications repeat).
const SEGMENT: usize = 500;

/// The kernel's input, made opaque (canonicalization folds no float arithmetic).
const X: f32 = 1.5;

/// The applications of `@chain`, each reading the one before's result.
const CHAIN: usize = 6;

/// `@body(x, c, out)`: `x + c`, then `* 0.5` or `+ 0.25` as [`halves`] picks, `OPS - 1` times,
/// and after stretches of `SEGMENT`, `SEGMENT + 1`, ... operations a selection that stores `-1`
/// to word 0 when the value is below `-2^30` (never: the value stays positive), a construct as a
/// real function's are (a store keeps it a branch); `@twice` computes `body(X, 1)` and `body(X, 2)`,
/// then stores them to words 0 and 1 of its buffer, so the first call's result is read after both
/// (the loop keeps it by trip); `@once` computes `body(X, 1)` alone and stores it to word 0.
/// `@chain` applies the body [`CHAIN`] times, each application to the one before's result, with
/// the constants 1, 2, ..., and stores the last result to word 0; `@lanes` loads a `float4` from
/// words 0 to 3, adds `0.5` to each lane, and applies the body to each lane in turn, with the
/// constant 1, storing the results to words 4 to 7.
fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    s.push_str(
        "define internal fastcc float @body(float %x, float %c, ptr addrspace(1) %out) {\nb0:\n",
    );
    s.push_str("  %v0 = fadd float %x, %c\n");
    let (mut next, mut stretch) = (SEGMENT, SEGMENT);
    for i in 1..OPS {
        let (op, k) = if halves(i) {
            ("fmul", "0x3FE0000000000000")
        } else {
            ("fadd", "0x3FD0000000000000")
        };
        s.push_str(&format!("  %v{i} = {op} float %v{}, {k}\n", i - 1));
        if i == next && i + 1 < OPS {
            stretch += 1;
            next += stretch;
            s.push_str(&format!(
                "  %t{i} = fcmp olt float %v{i}, 0xC1D0000000000000\n  br i1 %t{i}, label %r{i}, label %j{i}\nr{i}:\n  store float -1.0, ptr addrspace(1) %out, align 4\n  br label %j{i}\nj{i}:\n"
            ));
        }
    }
    s.push_str(&format!("  ret float %v{}\n}}\n\n", OPS - 1));
    s.push_str(&format!(
        "define void @twice(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n  %x = fadd float 0x{:016X}, -0.0\n  %a = call fastcc float @body(float %x, float 1.0, ptr addrspace(1) %0)\n  %b = call fastcc float @body(float %x, float 2.0, ptr addrspace(1) %0)\n  store float %a, ptr addrspace(1) %0, align 4\n  %p1 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1\n  store float %b, ptr addrspace(1) %p1, align 4\n  ret void\n}}\n\n",
        f64::from(X).to_bits()
    ));
    s.push_str(&format!(
        "define void @once(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n  %x = fadd float 0x{:016X}, -0.0\n  %a = call fastcc float @body(float %x, float 1.0, ptr addrspace(1) %0)\n  store float %a, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n",
        f64::from(X).to_bits()
    ));
    s.push_str(
        "define void @chain(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {\nentry:\n",
    );
    s.push_str(&format!(
        "  %c0 = fadd float 0x{:016X}, -0.0\n",
        f64::from(X).to_bits()
    ));
    for j in 1..=CHAIN {
        s.push_str(&format!(
            "  %c{j} = call fastcc float @body(float %c{}, float {j}.0, ptr addrspace(1) %0)\n",
            j - 1
        ));
    }
    s.push_str(&format!(
        "  store float %c{CHAIN}, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n"
    ));
    s.push_str("define void @lanes(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {\nentry:\n  %l = load <4 x float>, ptr addrspace(1) %0, align 4\n  %v = fadd <4 x float> %l, <float 0.5, float 0.5, float 0.5, float 0.5>\n");
    for j in 0..4 {
        s.push_str(&format!(
            "  %e{j} = extractelement <4 x float> %v, i32 {j}\n  %r{j} = call fastcc float @body(float %e{j}, float 1.0, ptr addrspace(1) %0)\n"
        ));
    }
    for j in 0..4 {
        s.push_str(&format!(
            "  %p{j} = getelementptr inbounds float, ptr addrspace(1) %0, i64 {}\n  store float %r{j}, ptr addrspace(1) %p{j}, align 4\n",
            4 + j
        ));
    }
    s.push_str("  ret void\n}\n\n");
    s.push_str("!llvm.module.flags = !{!0}\n!air.kernel = !{!20, !21, !22, !23}\n!air.version = !{!17}\n!air.language_version = !{!18}\n\n");
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n!20 = !{ptr @twice, !10, !11}\n!21 = !{ptr @once, !10, !11}\n!22 = !{ptr @chain, !10, !11}\n!23 = !{ptr @lanes, !10, !11}\n");
    s
}

/// What `@body(X, c)` returns.
fn body(c: f32) -> f32 {
    apply(X, c)
}

/// What `@body(x, c)` returns.
fn apply(x: f32, c: f32) -> f32 {
    let mut v = x + c;
    for i in 1..OPS {
        v = if halves(i) { v * 0.5 } else { v + 0.25 };
    }
    v
}

/// Whether operation `i` of the body halves the value (else it adds `0.25`): a bit of a
/// multiplicative hash of `i`. Operations that alternated made every stretch the same code as
/// the next one shifted by a stretch's length (a window holding the end of one stretch, its
/// selection and the start of the next compares equal to the window one stretch on), which the
/// roll pass writes as a loop once such a window is above its floor: the stretches of `@once` would
/// roll, where the test needs the body to hold no repeated region.
fn halves(i: usize) -> bool {
    ((i as u64).wrapping_mul(2_654_435_761) >> 16) & 1 == 1
}

/// The module assembled by `llvm-as`, or `None` (the test skips visibly) without it.
fn assembled() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-roll");
    let dir = scratch.path();
    let ll = dir.join("twice.ll");
    std::fs::write(&ll, module()).expect("write");
    let assembled = llvm_assemble(&ll);
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the module: {e}"),
    }
}

/// Run `m` (planned as `plan`) over one buffer of `words`, returning the buffer's words after.
fn run_words(m: &TypedModule, plan: &Plan, words: &[u32], stem: &str) -> Vec<u32> {
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec: Default::default(),
    };
    let out = run(m, &Config::default(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
    let b = out.buffers.first().expect("buffer");
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn a_function_applied_twice_is_compiled_once_on_every_profile() {
    let Some(bitcode) = assembled() else {
        return;
    };
    let floor = profile("runtime-floor", DevicePolicy::default()).expect("profile");
    let mvk = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    // Both compile the AIR inliner's two copies, and both roll them.
    let want = [body(1.0).to_bits(), body(2.0).to_bits()];
    for (p, stem) in [(&floor, "runtime-floor"), (&mvk, "moltenvk-m3")] {
        let (_, _, once) = compile_kept(&bitcode, "once", p).expect("once");
        let (art, plan, kept) = compile_kept(&bitcode, "twice", p).expect("twice");
        let (w1, w2) = (
            entry_weight(&once, &mut meter()).expect("weight"),
            entry_weight(&kept, &mut meter()).expect("weight"),
        );
        assert!(
            w1 * 2 > ROLL_ABOVE,
            "{stem}: two copies of the body are above the size that rolls: 2 x {w1}"
        );
        assert!(
            w2 < w1 + 256,
            "{stem}: the body is compiled once: {w2} instructions where one application is {w1}"
        );
        assert_eq!(
            kept.loops().len(),
            once.loops().len() + 1,
            "{stem}: one loop of two trips"
        );
        link_and_validate(&art, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let got = run_words(&kept, &plan, &[0, 0], stem);
        assert_eq!(
            got, want,
            "{stem}: body(1.5, 1), body(1.5, 2), and no store to word 2"
        );
    }
}

/// A function applied six times in a chain, each application to the one before's result
/// (`@chain`, the constants 1 to 6), rolls into one loop that carries the result from trip to
/// trip; a function applied to each lane of a vector computed before the applications
/// (`@lanes`) rolls into one loop that reads its trip's lane. On every profile each entry is
/// compiled about once (two applications are above the size that rolls) and computes what its
/// applications written out compute.
#[test]
fn a_chain_of_applications_and_one_by_lane_are_compiled_once_on_every_profile() {
    let Some(bitcode) = assembled() else {
        return;
    };
    let floor = profile("runtime-floor", DevicePolicy::default()).expect("profile");
    let mvk = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let chained = (1..=CHAIN).fold(X, |v, j| apply(v, j as f32));
    let lanes = [0.25f32, 1.0, 2.75, 3.5];
    let cases: [(&str, Vec<u32>, Vec<u32>); 2] = [
        ("chain", vec![0; 8], {
            let mut w = vec![0; 8];
            w[0] = chained.to_bits();
            w
        }),
        (
            "lanes",
            lanes.iter().map(|l| l.to_bits()).chain([0; 4]).collect(),
            lanes
                .iter()
                .map(|l| l.to_bits())
                .chain(lanes.iter().map(|l| apply(l + 0.5, 1.0).to_bits()))
                .collect(),
        ),
    ];
    for (p, stem) in [(&floor, "runtime-floor"), (&mvk, "MoltenVK")] {
        let (_, _, once) = compile_kept(&bitcode, "once", p).expect("once");
        let w1 = entry_weight(&once, &mut meter()).expect("weight");
        assert!(
            w1 * 2 > ROLL_ABOVE,
            "{stem}: two applications are above the size that rolls: 2 x {w1}"
        );
        for (entry, input, want) in &cases {
            let (art, plan, kept) = compile_kept(&bitcode, entry, p).expect(entry);
            let w = entry_weight(&kept, &mut meter()).expect("weight");
            assert!(
                w < w1 + 512,
                "{stem} {entry}: the body is compiled once: {w} instructions where one application is {w1}"
            );
            assert_eq!(
                kept.loops().len(),
                once.loops().len() + 1,
                "{stem} {entry}: one loop"
            );
            link_and_validate(&art, p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
            assert_eq!(
                run_words(&kept, &plan, input, stem),
                *want,
                "{stem} {entry}: the applications written out"
            );
        }
    }
}
