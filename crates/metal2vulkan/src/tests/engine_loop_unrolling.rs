//! A loop's unroll request end to end. The AIR states it on the loop's latch (`!llvm.loop`), as
//! Metal's compiler writes it: `llvm.loop.unroll.full` for a source's `#pragma clang loop
//! unroll(full)`, `llvm.loop.unroll.enable` for `#pragma unroll`, `llvm.loop.unroll.count` for
//! `#pragma unroll N`, and `llvm.loop.unroll.disable` for `#pragma nounroll`. The loop's
//! `OpLoopMerge` asks the driver the same, a hint that changes no answer: `Unroll` for the first
//! two, `PartialCount` and the factor for a count, `DontUnroll` for the last. A loop that asks
//! nothing, or asks what the translator does not take (a count of 0), asks the driver nothing.
//! Each kernel sums the buffer's first four words (1 to 4) into its first. `llvm-as` assembles the
//! module; without it each test skips visibly.

use crate::air::{self, SpecRequest};
use crate::device::{DevicePolicy, DeviceProfile};
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::lower::simd::WidestPin;
use crate::memory::static_root::analyze_memory;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, meter, profile, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spv::Op;

const PROFILES: [&str; 4] = ["amd-rdna3-radv", "lavapipe", "vk12-floor", "moltenvk-m3"];

/// The kernels: name, and the latch's attachment.
const KERNELS: [(&str, &str); 6] = [
    ("unroll_full", ", !llvm.loop !40"),
    ("unroll_disable", ", !llvm.loop !42"),
    ("unroll_enable", ", !llvm.loop !45"),
    ("unroll_count", ", !llvm.loop !47"),
    ("unroll_count_zero", ", !llvm.loop !49"),
    ("unroll_unasked", ""),
];

fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for (name, latch) in KERNELS {
        s.push_str(&format!(
            "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i32 %i
  %x = load i32, ptr addrspace(1) %p, align 4
  %acc.next = add i32 %acc, %x
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 4
  br i1 %done, label %exit, label %loop{latch}
exit:
  store i32 %acc.next, ptr addrspace(1) %0, align 4
  ret void
}}

"
        ));
    }
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"buf\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, (name, _)) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{name}, !10, !11}}\n", 20 + i));
    }
    s.push_str("!40 = distinct !{!40, !41, !44}\n!41 = !{!\"llvm.loop.unroll.full\"}\n");
    s.push_str("!42 = distinct !{!42, !43, !44}\n!43 = !{!\"llvm.loop.unroll.disable\"}\n");
    s.push_str("!44 = !{!\"llvm.loop.mustprogress\"}\n");
    s.push_str("!45 = distinct !{!45, !46, !44}\n!46 = !{!\"llvm.loop.unroll.enable\"}\n");
    s.push_str("!47 = distinct !{!47, !48, !44}\n!48 = !{!\"llvm.loop.unroll.count\", i32 4}\n");
    s.push_str("!49 = distinct !{!49, !50, !44}\n!50 = !{!\"llvm.loop.unroll.count\", i32 0}\n");
    s
}

/// Assemble the module, or `None` (skipped visibly) without `llvm-as`.
fn assemble() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-loop-unrolling");
    let ll = scratch.path().join("unrolling.ll");
    std::fs::write(&ll, module()).expect("write");
    match llvm_assemble(&ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the module: {e}"),
    }
}

/// The `LoopControl` of each `OpLoopMerge` in module `words` and the literals after it, in order.
fn loop_controls(words: &[u32]) -> Vec<Vec<u32>> {
    let mut i = 5;
    let mut controls = Vec::new();
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if op == Op::LoopMerge as u32 {
            controls.push(words[i + 3..i + len].to_vec());
        }
        i += len;
    }
    controls
}

/// Kernel `entry`'s typed module on `p`, run on the interpreter over the words 1, 2, 3, 4: the
/// buffer's first word.
fn interpret(bitcode: &[u8], entry: &str, p: &DeviceProfile) -> u32 {
    let mut m = meter();
    let decoded = air::decode(bitcode, &mut m).expect("decode");
    let req = SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, &mut m).expect("specialize");
    let memory = analyze_memory(&spec, p, &mut m).expect("memory");
    let plan = select(
        &spec,
        &memory,
        p,
        (&LowerInputs::new(), WidestPin::Weighed),
        &mut m,
    )
    .expect("select");
    let module = emit_module(&spec, &plan, &mut m).expect("emit").module;
    let bytes = [1u32, 2, 3, 4]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes,
        }],
        push: vec![0; 4],
        spec: Default::default(),
    };
    let out = run(&module, &Config::default(), res)
        .unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    let b = out.buffers.first().expect("buffer");
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Kernel `entry`'s one loop asks `control` (its `LoopControl` and the literals after it) on
/// every profile, and the kernel sums to 10.
fn asks(entry: &str, control: &[u32]) {
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let what = format!("{entry} on {stem}");
        let a = compile(&bitcode, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        assert_eq!(loop_controls(&a.module.words), [control], "{what}");
        assert_eq!(interpret(&bitcode, entry, &p), 10, "{what}");
    }
}

/// `llvm.loop.unroll.full` asks `Unroll`.
#[test]
fn a_full_unroll_request_asks_unroll() {
    asks("unroll_full", &[0x1]);
}

/// `llvm.loop.unroll.enable` (`#pragma unroll`) asks `Unroll`: fully where the trip count is
/// known, else partly, which is `Unroll`'s "to the degree possible".
#[test]
fn an_enabled_unroll_asks_unroll() {
    asks("unroll_enable", &[0x1]);
}

/// `llvm.loop.unroll.count` 4 asks `PartialCount` 4.
#[test]
fn an_unroll_count_asks_its_partial_count() {
    asks("unroll_count", &[0x100, 4]);
}

/// `llvm.loop.unroll.disable` asks `DontUnroll`.
#[test]
fn a_disabled_unroll_asks_dont_unroll() {
    asks("unroll_disable", &[0x2]);
}

/// A count of 0, which no unrolling takes, asks nothing; nor does a loop with no request.
#[test]
fn a_loop_without_a_request_it_takes_asks_nothing() {
    asks("unroll_count_zero", &[0x0]);
    asks("unroll_unasked", &[0x0]);
}
