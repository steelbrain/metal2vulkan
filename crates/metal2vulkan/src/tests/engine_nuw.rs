//! An integer operation the AIR marks `nuw` end to end. A narrow integer is held zero-extended in
//! a container, and an add, sub, mul or shl that can carry above its width masks its result;
//! one marked `nuw` does not (`ops::int_binary`): every result that would carry is poison. A loop
//! counter's increment is the common one, and its latch then reads the add itself, as the
//! driver's loop analysis needs to find the trip count (a masked `(i + 1) & 0xFFFF` hides it,
//! which keeps an `i16`-counted loop over a private array rolled and the array in scratch on
//! RADV). A plain `add` keeps its mask.
//!
//! Where a poison value meets something that must stay defined, it is masked: a `freeze` answers
//! a value of its type, and a store, a narrow atomic and a bitcast's packing change no byte or
//! lane but the poison one's. Each kernel loads its operand from the buffer (bytes 8 and 9 hold
//! `0xFF`), so nothing folds it. `llvm-as` assembles the module; without it the test skips
//! visibly.

use crate::air::{self, SpecRequest};
use crate::device::DevicePolicy;
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

const PROFILES: [&str; 4] = ["amd-rdna3-radv", "lavapipe", "runtime-floor", "moltenvk-m3"];

/// Two loops over a private `[4 x i32]`, each counted by an `i<bits>` whose increment carries
/// `flag`: the first stores `3 * i`, the second sums the elements (18).
fn loops(name: &str, bits: u8, flag: &str) -> String {
    format!(
        "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{
entry:
  %arr = alloca [4 x i32], align 4
  br label %fill
fill:
  %i = phi i{bits} [ 0, %entry ], [ %i.next, %fill ]
  %iw = zext i{bits} %i to i64
  %p = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 %iw
  %iv = zext i{bits} %i to i32
  %v = mul i32 %iv, 3
  store i32 %v, ptr %p, align 4
  %i.next = add {flag} i{bits} %i, 1
  %i.done = icmp eq i{bits} %i.next, 4
  br i1 %i.done, label %sum, label %fill
sum:
  %j = phi i{bits} [ 0, %fill ], [ %j.next, %sum ]
  %acc = phi i32 [ 0, %fill ], [ %acc.next, %sum ]
  %jw = zext i{bits} %j to i64
  %q = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 %jw
  %x = load i32, ptr %q, align 4
  %acc.next = add i32 %acc, %x
  %j.next = add {flag} i{bits} %j, 1
  %j.done = icmp eq i{bits} %j.next, 4
  br i1 %j.done, label %exit, label %sum
exit:
  store i32 %acc.next, ptr addrspace(1) %0, align 4
  ret void
}}

"
    )
}

/// A kernel whose body (`%0` is the buffer) ends in `ret void`.
fn kernel(name: &str, body: &str) -> String {
    format!(
        "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{
entry:
{body}  ret void
}}

"
    )
}

/// `%s` = `add nuw i<bits>` of the buffer's `0xFF..` at byte 8 and 1: poison.
fn wrapped(bits: u8) -> String {
    format!(
        "  %ip = getelementptr inbounds i8, ptr addrspace(1) %0, i64 8
  %a = load i{bits}, ptr addrspace(1) %ip, align 4
  %s = add nuw i{bits} %a, 1
"
    )
}

/// The `i16` 0x1234 at byte 2 (the neighbour of the field at byte 0).
const NEIGHBOUR: &str = "  %hp = getelementptr inbounds i8, ptr addrspace(1) %0, i64 2
  store i16 4660, ptr addrspace(1) %hp, align 2
";

const KERNELS: [&str; 10] = [
    "nuw_i16_loops",
    "plain_i16_loops",
    "nuw_i8_loops",
    "plain_i8_loops",
    "frozen_wrap",
    "frozen_wrap_chunked",
    "stored_wrap_lane",
    "stored_wrap_scalar",
    "atomic_wrap",
    "bitcast_wrap",
];

fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    s.push_str(&loops("nuw_i16_loops", 16, "nuw"));
    s.push_str(&loops("plain_i16_loops", 16, ""));
    s.push_str(&loops("nuw_i8_loops", 8, "nuw"));
    s.push_str(&loops("plain_i8_loops", 8, ""));
    s.push_str(&kernel(
        "frozen_wrap",
        &format!(
            "{}  %f = freeze i16 %s\n  %r = zext i16 %f to i32\n  store i32 %r, ptr addrspace(1) %0, align 4\n",
            wrapped(16)
        ),
    ));
    s.push_str(&kernel(
        "frozen_wrap_chunked",
        "  %ip = getelementptr inbounds i8, ptr addrspace(1) %0, i64 8
  %a = load i16, ptr addrspace(1) %ip, align 4
  %v = insertelement <8 x i16> zeroinitializer, i16 %a, i32 5
  %s = add nuw <8 x i16> %v, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %f = freeze <8 x i16> %s
  %e = extractelement <8 x i16> %f, i32 5
  %r = zext i16 %e to i32
  store i32 %r, ptr addrspace(1) %0, align 4
",
    ));
    s.push_str(&kernel(
        "stored_wrap_lane",
        &format!(
            "{}  %v = insertelement <2 x i16> <i16 0, i16 4660>, i16 %s, i32 0\n  store <2 x i16> %v, ptr addrspace(1) %0, align 4\n",
            wrapped(16)
        ),
    ));
    s.push_str(&kernel(
        "stored_wrap_scalar",
        &format!(
            "{NEIGHBOUR}{}  store i16 %s, ptr addrspace(1) %0, align 4\n",
            wrapped(16)
        ),
    ));
    s.push_str(&kernel(
        "atomic_wrap",
        &format!(
            "{NEIGHBOUR}{}  %o = atomicrmw xchg ptr addrspace(1) %0, i16 %s monotonic, align 4\n",
            wrapped(16)
        ),
    ));
    s.push_str(&kernel(
        "bitcast_wrap",
        &format!(
            "{}  %v = insertelement <4 x i8> <i8 1, i8 0, i8 52, i8 18>, i8 %s, i32 1\n  %h = bitcast <4 x i8> %v to <2 x i16>\n  %e = extractelement <2 x i16> %h, i32 1\n  %r = zext i16 %e to i32\n  store i32 %r, ptr addrspace(1) %0, align 4\n",
            wrapped(8)
        ),
    ));
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"buf\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, k) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{k}, !10, !11}}\n", 20 + i));
    }
    s
}

/// Assemble the module, or `None` (skipped visibly) without `llvm-as`.
fn assemble() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-nuw");
    let ll = scratch.path().join("nuw.ll");
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

/// The masks to `width` bits in module `words`: `OpBitwiseAnd` with the constant `2^width - 1`,
/// and `OpBitFieldUExtract` of `width` bits from bit 0.
fn masks(words: &[u32], width: u32) -> usize {
    let mut consts = std::collections::HashMap::new();
    let mut ins = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        let a = &words[i + 1..i + len];
        if op == Op::Constant as u32 && a.len() == 3 {
            consts.insert(a[1], a[2]);
        }
        ins.push((op, a));
        i += len;
    }
    let ones = (1u32 << width) - 1;
    let is = |id: u32, v: u32| consts.get(&id) == Some(&v);
    ins.iter()
        .filter(|(op, a)| match *op {
            o if o == Op::BitwiseAnd as u32 => a.len() == 4 && (is(a[2], ones) || is(a[3], ones)),
            o if o == Op::BitFieldUExtract as u32 => a.len() == 5 && is(a[3], 0) && is(a[4], width),
            _ => false,
        })
        .count()
}

/// Kernel `entry`'s typed module on `p`, run on the interpreter over a 16-byte buffer whose bytes
/// 8 and 9 are `0xFF`: the buffer's first word.
fn interpret(bitcode: &[u8], entry: &str, p: &crate::device::DeviceProfile) -> u32 {
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
    let mut bytes = vec![0u8; 16];
    bytes[8] = 0xFF;
    bytes[9] = 0xFF;
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

/// A loop counted by an `i16` or `i8` whose increment is `add nuw` exits on the add itself, with
/// no mask in the module; a plain `add` keeps its mask. Both sum the private array to 18.
#[test]
fn a_nuw_loop_counter_is_not_masked() {
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (entry, width, masked) in [
            ("nuw_i16_loops", 16, false),
            ("plain_i16_loops", 16, true),
            ("nuw_i8_loops", 8, false),
            ("plain_i8_loops", 8, true),
        ] {
            let what = format!("{entry} on {stem}");
            let a = compile(&bitcode, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let n = masks(&a.module.words, width);
            assert_eq!(n > 0, masked, "{what}: {n} masks to {width} bits");
            let got = interpret(&bitcode, entry, &p);
            assert_eq!(got, 18, "{what}");
        }
    }
}

/// A poison value of a `nuw` add stays its own: a `freeze` of it (a chunked vector's lane too) is a
/// value of `i16`, and a store, an atomic exchange and a bitcast's packing of it leave the
/// neighbouring `i16` 0x1234.
#[test]
fn a_poison_nuw_result_reaches_no_neighbour() {
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for entry in [
            "frozen_wrap",
            "frozen_wrap_chunked",
            "stored_wrap_lane",
            "stored_wrap_scalar",
            "atomic_wrap",
            "bitcast_wrap",
        ] {
            let what = format!("{entry} on {stem}");
            let a = compile(&bitcode, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let got = interpret(&bitcode, entry, &p);
            match entry {
                "frozen_wrap" | "frozen_wrap_chunked" => {
                    assert!(got <= 0xFFFF, "{what}: {got:#x}")
                }
                "bitcast_wrap" => assert_eq!(got, 0x1234, "{what}: {got:#x}"),
                _ => assert_eq!(got >> 16, 0x1234, "{what}: {got:#x}"),
            }
        }
    }
}
