//! Metal's `frexp` and `ilogb` runtime routines end to end: kernels calling
//! `___metal_frexp_<T>_pthread<I>` and `___metal_ilogb_<T>` on `float` and `half` scalars and
//! vectors compile on every profile below, link, pass L3, and on the CPU SPIR-V interpreter answer
//! every input of our probe as Metal did (an M4, macOS 27.0.1, libraries compiled offline;
//! every offline build, with and without fast math and `air.compile.denorms_disable`, answered
//! alike). The tables are those answers: the input's bits, `frexp`'s mantissa bits and exponent,
//! and `ilogb`. A module that records `PrecisionNote::NativeNanEncoding` answers a `half` NaN's
//! mantissa with a NaN whose bits are the device's (on ANV and RADV a `half` read from its bits is
//! the device's widening, which the interpreter answers with the positive quiet NaN).
//!
//! Each kernel loops over its input buffer (`[[buffer(0)]]`, one word or halfword an element) in
//! groups of its lanes, and writes three words an element to `[[buffer(1)]]`: the mantissa's
//! bits (zero-extended), the exponent, and `ilogb` (mode 0 in the scalar kernels, 3 in the vector
//! ones). `llvm-as` assembles the module; without it the tests skip visibly.

use crate::device::{DevicePolicy, DeviceProfile};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, profile, typed_module, AssembleError, Scratch,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run, Buffer, Config, Resources};

const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "amd-rdna3-radv",
];

/// Metal's answers (`nofastmath-offline`): `(x, frexp mantissa, frexp
/// exponent, ilogb)`.
const FLOATS: [(u32, u32, i32, i32); 76] = [
    (0x00000000, 0x00000000, 0, i32::MIN),
    (0x80000000, 0x00000000, 0, i32::MIN),
    (0x00000001, 0x00000000, 0, i32::MIN),
    (0x80000001, 0x00000000, 0, i32::MIN),
    (0x00000002, 0x00000000, 0, i32::MIN),
    (0x00000003, 0x00000000, 0, i32::MIN),
    (0x000000ff, 0x00000000, 0, i32::MIN),
    (0x00000100, 0x00000000, 0, i32::MIN),
    (0x00012345, 0x00000000, 0, i32::MIN),
    (0x00400000, 0x00000000, 0, i32::MIN),
    (0x80400000, 0x00000000, 0, i32::MIN),
    (0x007fffff, 0x00000000, 0, i32::MIN),
    (0x807fffff, 0x00000000, 0, i32::MIN),
    (0x00800000, 0x3f000000, -125, -126),
    (0x80800000, 0xbf000000, -125, -126),
    (0x00800001, 0x3f000001, -125, -126),
    (0x00ffffff, 0x3f7fffff, -125, -126),
    (0x01000000, 0x3f000000, -124, -125),
    (0x3f000000, 0x3f000000, 0, -1),
    (0x3f3fffff, 0x3f3fffff, 0, -1),
    (0x3f400000, 0x3f400000, 0, -1),
    (0x3f7fffff, 0x3f7fffff, 0, -1),
    (0x3f800000, 0x3f000000, 1, 0),
    (0xbf800000, 0xbf000000, 1, 0),
    (0x3f800001, 0x3f000001, 1, 0),
    (0x3fc00000, 0x3f400000, 1, 0),
    (0x40000000, 0x3f000000, 2, 1),
    (0x40400000, 0x3f400000, 2, 1),
    (0xc0400000, 0xbf400000, 2, 1),
    (0x40490fdb, 0x3f490fdb, 2, 1),
    (0x447a0000, 0x3f7a0000, 10, 9),
    (0x4b000000, 0x3f000000, 24, 23),
    (0x5f000000, 0x3f000000, 64, 63),
    (0x7e800000, 0x3f000000, 127, 126),
    (0x7f000000, 0x3f000000, 128, 127),
    (0x7f7fffff, 0x3f7fffff, 128, 127),
    (0xff7fffff, 0xbf7fffff, 128, 127),
    (0x7f800000, 0x7f800000, 0, i32::MAX),
    (0xff800000, 0xff800000, 0, i32::MAX),
    (0x7fc00000, 0x7fc00000, 0, i32::MIN),
    (0xffc00000, 0xffc00000, 0, i32::MIN),
    (0x7f800001, 0x7f800001, 0, i32::MIN),
    (0xff800001, 0xff800001, 0, i32::MIN),
    (0x7fa00000, 0x7fa00000, 0, i32::MIN),
    (0x7fbfffff, 0x7fbfffff, 0, i32::MIN),
    (0x7fc12345, 0x7fc12345, 0, i32::MIN),
    (0xffd2468a, 0xffd2468a, 0, i32::MIN),
    (0x7fffffff, 0x7fffffff, 0, i32::MIN),
    (0xffffffff, 0xffffffff, 0, i32::MIN),
    (0x33800000, 0x3f000000, -23, -24),
    (0x20000000, 0x3f000000, -62, -63),
    (0x1e3ce508, 0x3f3ce508, -66, -67),
    (0x0d800000, 0x3f000000, -99, -100),
    (0x00000001, 0x00000000, 0, i32::MIN),
    (0x00000002, 0x00000000, 0, i32::MIN),
    (0x00000004, 0x00000000, 0, i32::MIN),
    (0x00000008, 0x00000000, 0, i32::MIN),
    (0x00000010, 0x00000000, 0, i32::MIN),
    (0x00000020, 0x00000000, 0, i32::MIN),
    (0x00000040, 0x00000000, 0, i32::MIN),
    (0x00000080, 0x00000000, 0, i32::MIN),
    (0x00000100, 0x00000000, 0, i32::MIN),
    (0x00000200, 0x00000000, 0, i32::MIN),
    (0x00000400, 0x00000000, 0, i32::MIN),
    (0x00000800, 0x00000000, 0, i32::MIN),
    (0x00001000, 0x00000000, 0, i32::MIN),
    (0x00002000, 0x00000000, 0, i32::MIN),
    (0x00004000, 0x00000000, 0, i32::MIN),
    (0x00008000, 0x00000000, 0, i32::MIN),
    (0x00010000, 0x00000000, 0, i32::MIN),
    (0x00020000, 0x00000000, 0, i32::MIN),
    (0x00040000, 0x00000000, 0, i32::MIN),
    (0x00080000, 0x00000000, 0, i32::MIN),
    (0x00100000, 0x00000000, 0, i32::MIN),
    (0x00200000, 0x00000000, 0, i32::MIN),
    (0x00400000, 0x00000000, 0, i32::MIN),
];
const HALVES: [(u16, u16, i32, i32); 58] = [
    (0x0000, 0x0000, 0, i32::MIN),
    (0x8000, 0x0000, 0, i32::MIN),
    (0x0001, 0x3800, -23, -24),
    (0x8001, 0xb800, -23, -24),
    (0x0002, 0x3800, -22, -23),
    (0x0003, 0x3a00, -22, -23),
    (0x00ff, 0x3bf8, -16, -17),
    (0x0100, 0x3800, -15, -16),
    (0x0155, 0x3954, -15, -16),
    (0x0200, 0x3800, -14, -15),
    (0x8200, 0xb800, -14, -15),
    (0x03ff, 0x3bfe, -14, -15),
    (0x83ff, 0xbbfe, -14, -15),
    (0x0400, 0x3800, -13, -14),
    (0x8400, 0xb800, -13, -14),
    (0x0401, 0x3801, -13, -14),
    (0x07ff, 0x3bff, -13, -14),
    (0x0800, 0x3800, -12, -13),
    (0x3800, 0x3800, 0, -1),
    (0x39ff, 0x39ff, 0, -1),
    (0x3a00, 0x3a00, 0, -1),
    (0x3bff, 0x3bff, 0, -1),
    (0x3c00, 0x3800, 1, 0),
    (0xbc00, 0xb800, 1, 0),
    (0x3c01, 0x3801, 1, 0),
    (0x3e00, 0x3a00, 1, 0),
    (0x4000, 0x3800, 2, 1),
    (0x4200, 0x3a00, 2, 1),
    (0xc200, 0xba00, 2, 1),
    (0x4248, 0x3a48, 2, 1),
    (0x63d0, 0x3bd0, 10, 9),
    (0x7800, 0x3800, 16, 15),
    (0x7bff, 0x3bff, 16, 15),
    (0xfbff, 0xbbff, 16, 15),
    (0x7c00, 0x7c00, 0, i32::MAX),
    (0xfc00, 0xfc00, 0, i32::MAX),
    (0x7e00, 0x7e00, 0, i32::MIN),
    (0xfe00, 0xfe00, 0, i32::MIN),
    (0x7c01, 0x7c01, 0, i32::MIN),
    (0xfc01, 0xfc01, 0, i32::MIN),
    (0x7d00, 0x7d00, 0, i32::MIN),
    (0x7dff, 0x7dff, 0, i32::MIN),
    (0x7e55, 0x7e55, 0, i32::MIN),
    (0xfd34, 0xfd34, 0, i32::MIN),
    (0x7fff, 0x7fff, 0, i32::MIN),
    (0xffff, 0xffff, 0, i32::MIN),
    (0x1400, 0x3800, -9, -10),
    (0x0c00, 0x3800, -11, -12),
    (0x0001, 0x3800, -23, -24),
    (0x0002, 0x3800, -22, -23),
    (0x0004, 0x3800, -21, -22),
    (0x0008, 0x3800, -20, -21),
    (0x0010, 0x3800, -19, -20),
    (0x0020, 0x3800, -18, -19),
    (0x0040, 0x3800, -17, -18),
    (0x0080, 0x3800, -16, -17),
    (0x0100, 0x3800, -15, -16),
    (0x0200, 0x3800, -14, -15),
];

/// One kernel: the value type's element (`half` or `float`) and lanes.
struct Kernel {
    name: &'static str,
    half: bool,
    lanes: u32,
}

const KERNELS: [Kernel; 5] = [
    Kernel {
        name: "routines_float",
        half: false,
        lanes: 1,
    },
    Kernel {
        name: "routines_float3",
        half: false,
        lanes: 3,
    },
    Kernel {
        name: "routines_float4",
        half: false,
        lanes: 4,
    },
    Kernel {
        name: "routines_half",
        half: true,
        lanes: 1,
    },
    Kernel {
        name: "routines_half2",
        half: true,
        lanes: 2,
    },
];

impl Kernel {
    /// The inputs, padded with the first to a whole number of groups.
    fn inputs(&self) -> Vec<(u32, u32, i32, i32)> {
        let mut v: Vec<(u32, u32, i32, i32)> = if self.half {
            HALVES
                .iter()
                .map(|&(x, m, e, l)| (u32::from(x), u32::from(m), e, l))
                .collect()
        } else {
            FLOATS.to_vec()
        };
        while !v.len().is_multiple_of(self.lanes as usize) {
            v.push(v[0]);
        }
        v
    }

    /// The type tokens: the value (`T`, its routine spelling) and the exponent (`I`, its routine
    /// spelling), and the element's storage integer.
    fn types(&self) -> (String, String, String, String, &'static str) {
        let (elem, word) = if self.half {
            ("half", "i16")
        } else {
            ("float", "i32")
        };
        if self.lanes == 1 {
            (elem.into(), elem.into(), "i32".into(), "int32".into(), word)
        } else {
            let n = self.lanes;
            (
                format!("<{n} x {elem}>"),
                format!("v{n}{elem}"),
                format!("<{n} x i32>"),
                format!("v{n}int32"),
                word,
            )
        }
    }

    /// The kernel's definition and its two declarations.
    fn text(&self) -> (String, String) {
        let (t, tt, it, itt, word) = self.types();
        let (n, groups) = (self.lanes, self.inputs().len() as u32 / self.lanes);
        let frexp = format!("___metal_frexp_{tt}_pthread{itt}");
        let ilogb = format!("___metal_ilogb_{tt}");
        let mode = if n == 1 { 0 } else { 3 };
        let wt = if n == 1 {
            word.to_string()
        } else {
            format!("<{n} x {word}>")
        };
        let mut b = format!(
            "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr \
             addrspace(1) noundef \"air-buffer-no-alias\" %1) {{\nentry:\n  %e = alloca {it}, \
             align 16\n  br label %loop\nloop:\n  %g = phi i32 [ 0, %entry ], [ %g1, %loop ]\n  \
             %base = mul i32 %g, {n}\n",
            name = self.name
        );
        let mut w = if n == 1 {
            String::new()
        } else {
            "undef".to_string()
        };
        for j in 0..n {
            b.push_str(&format!(
                "  %i{j} = add i32 %base, {j}\n  %z{j} = zext i32 %i{j} to i64\n  %p{j} = \
                 getelementptr inbounds {word}, ptr addrspace(1) %0, i64 %z{j}\n  %w{j} = load \
                 {word}, ptr addrspace(1) %p{j}\n"
            ));
            if n == 1 {
                w = format!("%w{j}");
            } else {
                b.push_str(&format!(
                    "  %v{j} = insertelement {wt} {w}, {word} %w{j}, i32 {j}\n"
                ));
                w = format!("%v{j}");
            }
        }
        b.push_str(&format!(
            "  %x = bitcast {wt} {w} to {t}\n  %m = call {t} @{frexp}({t} %x, ptr %e)\n  %ev = \
             load {it}, ptr %e\n  %l = call {it} @{ilogb}({t} %x, i32 {mode})\n  %mb = bitcast \
             {t} %m to {wt}\n"
        ));
        for j in 0..n {
            let (mj, ej, lj) = if n == 1 {
                ("%mb".to_string(), "%ev".to_string(), "%l".to_string())
            } else {
                b.push_str(&format!(
                    "  %mx{j} = extractelement {wt} %mb, i32 {j}\n  %ex{j} = extractelement {it} \
                     %ev, i32 {j}\n  %lx{j} = extractelement {it} %l, i32 {j}\n"
                ));
                (format!("%mx{j}"), format!("%ex{j}"), format!("%lx{j}"))
            };
            let widen = if self.half {
                format!("  %mw{j} = zext i16 {mj} to i32\n")
            } else {
                format!("  %mw{j} = add i32 {mj}, 0\n")
            };
            b.push_str(&widen);
            for (k, v) in [(0, format!("%mw{j}")), (1, ej), (2, lj)] {
                b.push_str(&format!(
                    "  %o{j}_{k} = mul i32 %i{j}, 3\n  %q{j}_{k} = add i32 %o{j}_{k}, {k}\n  \
                     %r{j}_{k} = zext i32 %q{j}_{k} to i64\n  %s{j}_{k} = getelementptr inbounds \
                     i32, ptr addrspace(1) %1, i64 %r{j}_{k}\n  store i32 {v}, ptr addrspace(1) \
                     %s{j}_{k}, align 4\n"
                ));
            }
        }
        b.push_str(&format!(
            "  %g1 = add i32 %g, 1\n  %done = icmp eq i32 %g1, {groups}\n  br i1 %done, label \
             %exit, label %loop\nexit:\n  ret void\n}}\n\n"
        ));
        let decls = format!("declare {t} @{frexp}({t}, ptr)\ndeclare {it} @{ilogb}({t}, i32)\n");
        (b, decls)
    }
}

/// The module of every kernel.
fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    let mut decls = String::new();
    for k in &KERNELS {
        let (def, d) = k.text();
        s.push_str(&def);
        decls.push_str(&d);
    }
    s.push_str(&decls);
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str(
        "!10 = !{}\n!11 = !{!12, !13}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", \
         i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 \
         4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"x\"}\n!13 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 \
         1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", \
         i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"out\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, \
         i32 0, i32 0}\n",
    );
    for (i, k) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, k.name));
    }
    s
}

/// Assemble the module, or `None` (skipped visibly) without `llvm-as`.
fn assemble() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-routines");
    let dir = scratch.path();
    let ll = dir.join("routines.ll");
    std::fs::write(&ll, module()).expect("write");
    let assembled = llvm_assemble(&ll);
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the routines module: {e}"),
    }
}

/// The words kernel `k` writes on profile `p` (`what` names it), run on the interpreter.
fn interpret(bc: &[u8], k: &Kernel, p: &DeviceProfile, what: &str) -> Vec<u32> {
    let (plan, m) = typed_module(bc, k.name, p).unwrap_or_else(|e| panic!("{what}: {e}"));
    let inputs = k.inputs();
    let input: Vec<u8> = if k.half {
        inputs
            .iter()
            .flat_map(|i| (i.0 as u16).to_le_bytes())
            .collect()
    } else {
        inputs.iter().flat_map(|i| i.0.to_le_bytes()).collect()
    };
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: slot(0),
                address: None,
                bytes: input,
            },
            Buffer {
                slot: slot(1),
                address: None,
                bytes: vec![0xab; 12 * inputs.len()],
            },
        ],
        push: vec![0; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec: Default::default(),
    };
    let out =
        run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{what}: interpreter: {e}"));
    out.buffers[1]
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Whether `bits` (zero-extended) are a `half` NaN's.
fn half_nan(bits: u32) -> bool {
    bits <= 0xffff && bits & 0x7c00 == 0x7c00 && bits & 0x3ff != 0
}

/// Every kernel compiles, links and passes L3 on every profile, and answers
/// every measured input as Metal did on the interpreter: `float` denormals flushed (`(+0, 0)`,
/// `INT_MIN`), `half` denormals normalized, zeros `(+0, 0)` and `INT_MIN` whatever their sign,
/// infinities and NaNs returned with their bits and exponent 0, `ilogb` `INT_MAX` for an infinity
/// and `INT_MIN` for a NaN, lane by lane. The one exception: a module that records
/// `NativeNanEncoding` answers a `half` NaN's mantissa with a NaN of the device's bits; every
/// `half` kernel records it, on a profile whose moves of a NaN are measured to be Metal's
/// (`DriverDeviation::MetalNanMoves`) too, since the routines read the widened value's bits.
#[test]
fn frexp_and_ilogb_answer_as_metals_routines() {
    let Some(bc) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for k in &KERNELS {
            let what = format!("{} on {stem}", k.name);
            let a = compile(&bc, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let device_nans = a.precision.contains(&PrecisionNote::NativeNanEncoding);
            // The routines read a `half`'s widened bits, which the device's measured moves of a
            // NaN do not cover (`narrow::moves_unfolded`): noted on every profile.
            assert!(!k.half || device_nans, "{what}: {:?}", a.precision);
            let got = interpret(&bc, k, &p, &what);
            for (i, &(x, mantissa, exponent, ilogb)) in k.inputs().iter().enumerate() {
                let got = &got[3 * i..3 * i + 3];
                let want = [mantissa, exponent as u32, ilogb as u32];
                if k.half && device_nans && half_nan(x) {
                    assert!(
                        half_nan(got[0]) && got[1..] == want[1..],
                        "{what}: x = {x:#x}: got {got:#x?}"
                    );
                    continue;
                }
                assert_eq!(got, want, "{what}: x = {x:#x}: got {got:#x?}");
            }
        }
    }
}
