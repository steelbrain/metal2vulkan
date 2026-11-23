//! The masks a narrow integer's operations keep, by the profile's `NarrowWrap` fact.
//!
//! An owned kernel computes `i16` and `i8` results through every kind of use: one that reads only
//! the low bits (another wrapping operation, a truncation, a sign extension, a signed divide's
//! operand), one that passes the bits on (a `select`, a phi, a bitwise and or or) and one that
//! reads them (a zero extension, a comparison, a logical shift right). `Device::from_queries`
//! drops the wrap of every result whose bits above its width no use reads and keeps every other.
//! The kernel translates and validates on every committed profile, and answers LLVM's words on the
//! CPU interpreter. `llvm-as` assembles the kernel; without it the tests skip visibly. No GPU is
//! used.

use std::collections::BTreeMap;

use crate::air::ir::BinOp;
use crate::device::{Device, DevicePolicy, DeviceProfile};
use crate::lower::arith::ArithStrategy;
use crate::lower::select::Plan;
use crate::lower::strategy::Strategy;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, description, link_and_validate, llvm_assemble_text, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::committed_profiles;

const ENTRY: &str = "narrow_wraps";

/// The threads: every pair of edges.
const THREADS: u32 = 256;

/// The words each thread writes.
const WORDS: usize = 8;

/// The edge values, as words: each operation reads its low 16 or 8 bits.
const EDGES: [u32; 16] = [
    0x0000_0000,
    0x0000_0001,
    0x0000_7ffe,
    0x0000_7fff,
    0x0000_8000,
    0x0000_8001,
    0x0000_ffff,
    0x0000_fffe,
    0x0000_007f,
    0x0000_0080,
    0x0000_00ff,
    0x0000_0081,
    0x0000_4000,
    0x0000_c000,
    0x0000_0040,
    0x0000_ffc0,
];

/// The owned synthetic kernel: thread `gid` reads edges `x` (`gid & 15`) and `y` (`gid >> 4`)
/// and writes [`want`]'s eight words.
const KERNEL: &str = r#"; Owned synthetic kernel.
source_filename = "narrow_wraps.metal"

define void @narrow_wraps(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {
entry:
  %xi = and i32 %gid, 15
  %yi = lshr i32 %gid, 4
  %xw = zext i32 %xi to i64
  %yw = zext i32 %yi to i64
  %xp = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %xw
  %yp = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %yw
  %x = load i32, ptr addrspace(1) %xp, align 4
  %y = load i32, ptr addrspace(1) %yp, align 4
  %x16 = trunc i32 %x to i16
  %y16 = trunc i32 %y to i16
  %x8 = trunc i32 %x to i8
  %cond = icmp ult i32 %xi, 8
  %a = add i16 %x16, %y16
  %b = mul i16 %a, %y16
  %c = sub i16 %b, %x16
  %d = ashr i16 %c, 3
  %s = shl i16 %x16, 5
  %t = add i16 %y16, 7
  %sel = select i1 %cond, i16 %s, i16 %t
  %q = sdiv i16 %sel, -7
  %q8 = trunc i16 %q to i8
  %r8 = add i8 %q8, %x8
  %u = add i16 %x16, %y16
  %v = mul i16 %x16, %x16
  %w = lshr i16 %v, 4
  %a3 = add i16 %x16, 1234
  %n = and i16 %a3, %y16
  %a2 = sub i16 %y16, %x16
  %o = or i16 %a2, %x16
  br i1 %cond, label %then, label %else

then:
  %p1 = add i16 %x16, 300
  %p3 = mul i16 %x16, 3
  br label %join

else:
  %p2 = sub i16 %y16, 300
  %p4 = shl i16 %y16, 2
  br label %join

join:
  %ph = phi i16 [ %p1, %then ], [ %p2, %else ]
  %ph2 = phi i16 [ %p3, %then ], [ %p4, %else ]
  %w0 = zext i16 %d to i32
  %w1 = sext i8 %r8 to i32
  %ug = icmp ugt i16 %u, 1000
  %w2 = zext i1 %ug to i32
  %w3 = zext i16 %w to i32
  %w4 = sext i16 %ph to i32
  %w5 = zext i16 %ph2 to i32
  %w6 = sext i16 %n to i32
  %w7 = zext i16 %o to i32
  %base = mul i32 %gid, 8
  %bw = zext i32 %base to i64
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %bw
  store i32 %w0, ptr addrspace(1) %p0, align 4
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 1
  store i32 %w1, ptr addrspace(1) %q1, align 4
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 2
  store i32 %w2, ptr addrspace(1) %q2, align 4
  %q3 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 3
  store i32 %w3, ptr addrspace(1) %q3, align 4
  %q4 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 4
  store i32 %w4, ptr addrspace(1) %q4, align 4
  %q5 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 5
  store i32 %w5, ptr addrspace(1) %q5, align 4
  %q6 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 6
  store i32 %w6, ptr addrspace(1) %q6, align 4
  %q7 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 7
  store i32 %w7, ptr addrspace(1) %q7, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @narrow_wraps, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// The narrow operations in the kernel's order, each with whether its result is masked under
/// `NarrowWrap::Read`: a result only another wrapping operation, a truncation, a sign extension or
/// a signed divide reads, directly or through a `select`, a phi or a bitwise and, is not; one a
/// zero extension, a comparison or a logical shift right reads, directly or through a phi or a
/// bitwise or, is. An operation that never masks (`lshr`, `and`, `or`) stays as it is.
const READ_WRAPS: [(BinOp, bool); 19] = [
    (BinOp::Add, false),  // %a, by %b
    (BinOp::Mul, false),  // %b, by %c
    (BinOp::Sub, false),  // %c, by %d
    (BinOp::AShr, true),  // %d, by its zext
    (BinOp::Shl, false),  // %s, through the select by %q
    (BinOp::Add, false),  // %t, through the select by %q
    (BinOp::SDiv, false), // %q, by its trunc
    (BinOp::Add, false),  // %r8, by its sext
    (BinOp::Add, true),   // %u, by the icmp
    (BinOp::Mul, true),   // %v, by the lshr
    (BinOp::LShr, true),  // %w
    (BinOp::Add, false),  // %a3, through the and by its sext
    (BinOp::And, true),   // %n
    (BinOp::Sub, true),   // %a2, through the or by its zext
    (BinOp::Or, true),    // %o
    (BinOp::Add, false),  // %p1, through %ph by its sext
    (BinOp::Mul, true),   // %p3, through %ph2 by its zext
    (BinOp::Sub, false),  // %p2, through %ph by its sext
    (BinOp::Shl, true),   // %p4, through %ph2 by its zext
];

/// Thread `gid`'s eight words, from LLVM's definitions.
fn want(gid: u32) -> [u32; WORDS] {
    let xi = gid & 15;
    let (x, y) = (EDGES[xi as usize], EDGES[(gid >> 4) as usize]);
    let (x16, y16, x8) = (x as u16, y as u16, x as u8);
    let cond = xi < 8;
    let a = x16.wrapping_add(y16);
    let b = a.wrapping_mul(y16);
    let c = b.wrapping_sub(x16);
    let d = ((c as i16) >> 3) as u16;
    let sel = if cond {
        x16.wrapping_shl(5)
    } else {
        y16.wrapping_add(7)
    };
    let q = ((sel as i16) / -7) as u16;
    let r8 = (q as u8).wrapping_add(x8);
    let u = x16.wrapping_add(y16);
    let w = x16.wrapping_mul(x16) >> 4;
    let n = x16.wrapping_add(1234) & y16;
    let o = y16.wrapping_sub(x16) | x16;
    let ph = if cond {
        x16.wrapping_add(300)
    } else {
        y16.wrapping_sub(300)
    };
    let ph2 = if cond {
        x16.wrapping_mul(3)
    } else {
        y16.wrapping_shl(2)
    };
    [
        u32::from(d),
        r8 as i8 as u32,
        u32::from(u > 1000),
        u32::from(w),
        ph as i16 as u32,
        u32::from(ph2),
        n as i16 as u32,
        u32::from(o),
    ]
}

/// The kernel's bitcode, or `None` (skipped visibly) without `llvm-as`.
fn bitcode() -> Option<Vec<u8>> {
    match llvm_assemble_text(KERNEL) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {ENTRY}: {why}"),
    }
}

/// `Device::from_queries`' profile of committed description `stem`.
fn queried(stem: &str) -> DeviceProfile {
    let desc = description(stem).expect("description");
    let device = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
    device.profile().clone()
}

/// Each narrow integer operation of `plan`, in the kernel's order, with whether it masks.
fn wraps(plan: &Plan) -> Vec<(BinOp, bool)> {
    plan.strategy
        .iter()
        .filter_map(|s| match s {
            Some(Strategy::Arith(ArithStrategy::Int { op, form, wrap, .. }))
                if form.legalized() =>
            {
                Some((*op, *wrap))
            }
            _ => None,
        })
        .collect()
}

/// On every committed profile, `Device::from_queries`' drops exactly the unread narrow results'
/// wraps; the kernel translates and validates.
#[test]
fn from_queries_masks_only_the_narrow_bits_a_use_reads() {
    let Some(bc) = bitcode() else {
        return;
    };
    let read: Vec<_> = READ_WRAPS.to_vec();
    for (stem, p) in &committed_profiles().expect("profiles") {
        let (plan, _) = typed_module(&bc, ENTRY, p).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(wraps(&plan), read, "{stem}");
        let a = compile(&bc, ENTRY, p).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
    }
}

/// The module answers LLVM's words over every pair of edges on the CPU interpreter.
#[test]
fn unmasked_narrow_results_answer_the_same_words() {
    let Some(bc) = bitcode() else {
        return;
    };
    let input: Vec<u8> = EDGES.iter().flat_map(|w| w.to_le_bytes()).collect();
    let expected: Vec<u32> = (0..THREADS).flat_map(want).collect();
    let ctor = "nvidia-rtx50-linux";
    let p = queried(ctor);
    let (plan, m) = typed_module(&bc, ENTRY, &p).unwrap_or_else(|e| panic!("{ctor}: {e:?}"));
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([THREADS, 1, 1]));
    let buffers = [input, vec![0xAB; THREADS as usize * WORDS * 4]];
    let res = Resources {
        buffers: buffers
            .iter()
            .enumerate()
            .map(|(i, bytes)| Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Buffer,
                    index: i as u32,
                }),
                address: None,
                bytes: bytes.clone(),
            })
            .collect(),
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let cfg = Config {
        order: Order::Forward,
        ..Config::default()
    };
    let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{ctor}: {e:?}"));
    let got: Vec<u32> = out.buffers[1]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    for (i, (g, w)) in got.iter().zip(&expected).enumerate() {
        let (gid, k) = (i / WORDS, i % WORDS);
        assert_eq!(g, w, "{ctor}: thread {gid} word {k}");
    }
}
