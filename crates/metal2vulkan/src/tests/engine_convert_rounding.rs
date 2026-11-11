//! `air.convert` with a rounding after its types and `.sat` after it, the forms Apple's OpenCL
//! compiler emits for every `convert_<type>[_sat][_<mode>]`.
//!
//! Outside OpenCL mode each compiles to exactly the module of the conversion its types name, which
//! is how Metal's compilers for the x86 Macs answer them there (the measurements are in
//! docs/API.md's accepted differences). Every rounding of each form, and each with `.sat` where it
//! converts to an integer, compiles on lavapipe (64-bit integers) and on moltenvk-m3 without
//! `shaderInt64` (the wide path) and passes L3, and its module's words and precision notes are the
//! plain conversion's.
//!
//! In a compute pipeline of OpenCL mode the suffix is honored where Metal honors it there: run on
//! the interpreter, each form answers the words a Radeon Pro 5300M and a UHD Graphics 630 (macOS
//! 26.7.1) answered alike in OpenCL mode, and the same kernel outside OpenCL mode answers the
//! conversion its types name.
//!
//! `llvm-as` assembles the modules; without it the test skips visibly.

use crate::air::{self, SpecRequest};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy, DeviceProfile};
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, compile_with, description, link_and_validate, llvm_assemble_text, meter, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// One conversion: its destination and source as `air` tokens (`f.f32`, `s.v4i32`) and LLVM
/// types.
struct Form {
    dst: (&'static str, &'static str),
    src: (&'static str, &'static str),
}

const fn form(dst: (&'static str, &'static str), src: (&'static str, &'static str)) -> Form {
    Form { dst, src }
}

/// Integer to float and `half` (exact and rounding), float to integer of each width, integer to
/// integer narrowing and widening across signedness, and float narrowing and widening, scalar and
/// vector.
const FORMS: [Form; 24] = [
    form(("f.f32", "float"), ("s.i32", "i32")),
    form(("f.f32", "float"), ("u.i32", "i32")),
    form(("f.f16", "half"), ("s.i32", "i32")),
    form(("f.f16", "half"), ("u.i16", "i16")),
    form(("f.f32", "float"), ("s.i64", "i64")),
    form(("f.f32", "float"), ("s.i16", "i16")),
    form(("f.f16", "half"), ("u.i8", "i8")),
    form(("f.f16", "half"), ("s.i8", "i8")),
    form(("f.v4f32", "<4 x float>"), ("u.v4i8", "<4 x i8>")),
    form(("f.v4f32", "<4 x float>"), ("s.v4i32", "<4 x i32>")),
    form(("s.i32", "i32"), ("f.f32", "float")),
    form(("u.i32", "i32"), ("f.f32", "float")),
    form(("u.i8", "i8"), ("f.f32", "float")),
    form(("s.i16", "i16"), ("f.f16", "half")),
    form(("s.i64", "i64"), ("f.f32", "float")),
    form(("s.v4i32", "<4 x i32>"), ("f.v4f32", "<4 x float>")),
    form(("u.i8", "i8"), ("s.i32", "i32")),
    form(("s.i16", "i16"), ("u.i32", "i32")),
    form(("s.i32", "i32"), ("s.i64", "i64")),
    form(("u.i64", "i64"), ("s.i32", "i32")),
    form(("u.v4i8", "<4 x i8>"), ("s.v4i32", "<4 x i32>")),
    form(("f.f16", "half"), ("f.f32", "float")),
    form(("f.f32", "float"), ("f.f16", "half")),
    form(("f.v4f16", "<4 x half>"), ("f.v4f32", "<4 x float>")),
];

/// A kernel storing `name`'s conversion of four loaded operands: buffer 0 the results, buffer 1
/// the operands.
fn kernel(f: &Form, name: &str) -> String {
    let (dst, src) = (f.dst.1, f.src.1);
    let mut body = String::new();
    for k in 0..4 {
        body.push_str(&format!(
            "  %p{k} = getelementptr inbounds {src}, ptr addrspace(1) %1, i64 {k}\n  %x{k} = load {src}, ptr addrspace(1) %p{k}, align 4\n  %r{k} = call {dst} @{name}({src} %x{k})\n  %q{k} = getelementptr inbounds {dst}, ptr addrspace(1) %0, i64 {k}\n  store {dst} %r{k}, ptr addrspace(1) %q{k}, align 4\n"
        ));
    }
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @convert(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1) {{\nentry:\n{body}  ret void\n}}\n\n\
declare {dst} @{name}({src})\n\n\
!llvm.module.flags = !{{!0}}\n!air.kernel = !{{!20}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n\
!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"in\"}}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n!20 = !{{ptr @convert, !10, !11}}\n"
    )
}

fn assemble(text: &str) -> Option<Vec<u8>> {
    match llvm_assemble_text(text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    }
}

/// The committed `stem` profile, with or without `shaderInt64`.
fn profile(stem: &str, int64: bool) -> DeviceProfile {
    let mut desc = description(stem).expect("description");
    if !int64 {
        assert!(desc.features.remove(&VkFeature::ShaderInt64), "{stem}");
    }
    let p = Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone();
    assert_eq!(p.fields().leaf.int64, int64, "{stem}");
    p
}

/// Every rounding of every form, and `.sat` after it to an integer, compiles to the plain
/// conversion's module and precision notes and passes L3, through native 64-bit integers and
/// through the wide path.
#[test]
fn a_rounded_or_saturated_convert_compiles_as_the_plain_conversion() {
    let profiles = [
        ("lavapipe", profile("lavapipe", true)),
        ("moltenvk-m3 words", profile("moltenvk-m3", false)),
    ];
    let mut compared = 0;
    for f in &FORMS {
        let plain = format!("air.convert.{}.{}", f.dst.0, f.src.0);
        let Some(plain_bc) = assemble(&kernel(f, &plain)) else {
            return;
        };
        let mut suffixes = vec!["rte", "rtz", "rtp", "rtn"];
        if !f.dst.0.starts_with("f.") {
            suffixes.extend(["rte.sat", "rtz.sat", "rtp.sat", "rtn.sat"]);
        }
        let suffixed: Vec<(String, Vec<u8>)> = suffixes
            .iter()
            .map(|suffix| {
                let name = format!("{plain}.{suffix}");
                let bc = assemble(&kernel(f, &name)).expect("llvm-as ran for the plain form");
                (name, bc)
            })
            .collect();
        for (stem, p) in &profiles {
            let want = compile(&plain_bc, "convert", p).unwrap_or_else(|e| panic!("{plain}: {e}"));
            for (name, bc) in &suffixed {
                let what = format!("{name} on {stem}");
                let a = compile(bc, "convert", p).unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                assert_eq!(a.precision, want.precision, "{what}: notes");
                assert!(
                    a.module.words == want.module.words,
                    "{what}: the module differs from {plain}'s"
                );
                compared += 1;
            }
        }
    }
    eprintln!(
        "compared {compared} suffixed conversions of {} forms with their plain modules",
        FORMS.len()
    );
}

/// One form in OpenCL mode: the conversion's name after `air.convert.`, its destination's and
/// source's LLVM types, and four operands, each with the word the 5300M and the 630 answered for
/// it in OpenCL mode and the conversion's answer without the suffix. Probe jobs m3-max-15651
/// (5300M) and m3-max-15652 (630), OpenCL C of our own, one work-item per operand; each operand
/// here is one the two answered alike, chosen where the suffix changes the answer where it can.
type Measured = (
    &'static str,
    &'static str,
    &'static str,
    [(u64, u64, u64); 4],
);

/// Every form the two honor: an integer of 32 or 64 bits to `float` toward zero, up and down; a
/// `float` to an integer of 8 to 64 bits to nearest even, up and down, saturating or not; and
/// `.sat` between integers (and one rounding between integers, which neither honors).
const MEASURED: [Measured; 58] = [
    (
        "f.f32.s.i32.rtz",
        "float",
        "i32",
        [
            (0x1000003, 0x4b800001, 0x4b800002),
            (0x1ffffff, 0x4bffffff, 0x4c000000),
            (0x2000003, 0x4c000000, 0x4c000001),
            (0xfefffffd, 0xcb800001, 0xcb800002),
        ],
    ),
    (
        "f.f32.s.i64.rtz",
        "float",
        "i64",
        [
            (0x7fffffffffffffff, 0x5effffff, 0x5f000000),
            (0x8000000000000001, 0xdeffffff, 0xdf000000),
            (0x7fffffc000000000, 0x5effffff, 0x5f000000),
            (0x10000010001, 0x53800000, 0x53800001),
        ],
    ),
    (
        "f.f32.s.i32.rtp",
        "float",
        "i32",
        [
            (0x1000001, 0x4b800001, 0x4b800000),
            (0x2000001, 0x4c000001, 0x4c000000),
            (0xfefffffd, 0xcb800001, 0xcb800002),
            (0xfe000001, 0xcbffffff, 0xcc000000),
        ],
    ),
    (
        "f.f32.s.i64.rtp",
        "float",
        "i64",
        [
            (0x1000001, 0x4b800001, 0x4b800000),
            (0x20000000000001, 0x5a000001, 0x5a000000),
            (0x8000000000000001, 0xdeffffff, 0xdf000000),
            (0x4000000000000001, 0x5e800001, 0x5e800000),
        ],
    ),
    (
        "f.f32.s.i32.rtn",
        "float",
        "i32",
        [
            (0x1000003, 0x4b800001, 0x4b800002),
            (0x1ffffff, 0x4bffffff, 0x4c000000),
            (0x2000003, 0x4c000000, 0x4c000001),
            (0xfeffffff, 0xcb800001, 0xcb800000),
        ],
    ),
    (
        "f.f32.s.i64.rtn",
        "float",
        "i64",
        [
            (0x7fffffffffffffff, 0x5effffff, 0x5f000000),
            (0x7fffffc000000000, 0x5effffff, 0x5f000000),
            (0x10000010001, 0x53800000, 0x53800001),
            (0x20000020000001, 0x5a000000, 0x5a000001),
        ],
    ),
    (
        "f.f32.u.i32.rtz",
        "float",
        "i32",
        [
            (0xffffffff, 0x4f7fffff, 0x4f800000),
            (0x1000003, 0x4b800001, 0x4b800002),
            (0x1ffffff, 0x4bffffff, 0x4c000000),
            (0x2000003, 0x4c000000, 0x4c000001),
        ],
    ),
    (
        "f.f32.u.i64.rtz",
        "float",
        "i64",
        [
            (0xffffffffffffffff, 0x5f7fffff, 0x5f800000),
            (0x7fffffffffffffff, 0x5effffff, 0x5f000000),
            (0x7fffffc000000000, 0x5effffff, 0x5f000000),
            (0xffffff8000000000, 0x5f7fffff, 0x5f800000),
        ],
    ),
    (
        "f.f32.u.i32.rtp",
        "float",
        "i32",
        [
            (0x1000001, 0x4b800001, 0x4b800000),
            (0x2000001, 0x4c000001, 0x4c000000),
            (0xfe000001, 0x4f7e0001, 0x4f7e0000),
            (0x80000001, 0x4f000001, 0x4f000000),
        ],
    ),
    (
        "f.f32.u.i64.rtp",
        "float",
        "i64",
        [
            (0x1000001, 0x4b800001, 0x4b800000),
            (0x20000000000001, 0x5a000001, 0x5a000000),
            (0x8000000000000001, 0x5f000001, 0x5f000000),
            (0x4000000000000001, 0x5e800001, 0x5e800000),
        ],
    ),
    (
        "f.f32.u.i32.rtn",
        "float",
        "i32",
        [
            (0xffffffff, 0x4f7fffff, 0x4f800000),
            (0x1000003, 0x4b800001, 0x4b800002),
            (0x1ffffff, 0x4bffffff, 0x4c000000),
            (0x2000003, 0x4c000000, 0x4c000001),
        ],
    ),
    (
        "f.f32.u.i64.rtn",
        "float",
        "i64",
        [
            (0xffffffffffffffff, 0x5f7fffff, 0x5f800000),
            (0x7fffffffffffffff, 0x5effffff, 0x5f000000),
            (0x7fffffc000000000, 0x5effffff, 0x5f000000),
            (0xffffff8000000000, 0x5f7fffff, 0x5f800000),
        ],
    ),
    (
        "s.i32.f.f32.rte",
        "i32",
        "float",
        [
            (0x3fc00000, 0x2, 0x1),
            (0xbfc00000, 0xfffffffe, 0xffffffff),
            (0x46ffff33, 0x8000, 0x7fff),
            (0xc7000099, 0xffff7fff, 0xffff8000),
        ],
    ),
    (
        "s.i32.f.f32.rte.sat",
        "i32",
        "float",
        [
            (0x3fc00000, 0x2, 0x1),
            (0xbfc00000, 0xfffffffe, 0xffffffff),
            (0x46ffff33, 0x8000, 0x7fff),
            (0xc7000099, 0xffff7fff, 0xffff8000),
        ],
    ),
    (
        "u.i32.f.f32.rte.sat",
        "i32",
        "float",
        [
            (0x3fc00000, 0x2, 0x1),
            (0x46ffff33, 0x8000, 0x7fff),
            (0x437f8000, 0x100, 0xff),
            (0x42ff0000, 0x80, 0x7f),
        ],
    ),
    (
        "s.i8.f.f32.rte.sat",
        "i8",
        "float",
        [
            (0x3fc00000, 0x2, 0x1),
            (0xbfc00000, 0xfe, 0xff),
            (0xbf400000, 0xff, 0x0),
            (0x406ccccd, 0x4, 0x3),
        ],
    ),
    (
        "u.i8.f.f32.rte.sat",
        "i8",
        "float",
        [
            (0x3f000000, 0x0, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x406ccccd, 0x4, 0x3),
            (0x3f400000, 0x1, 0x0),
        ],
    ),
    (
        "s.i16.f.f32.rte.sat",
        "i16",
        "float",
        [
            (0x3fc00000, 0x2, 0x1),
            (0xbfc00000, 0xfffe, 0xffff),
            (0x437f8000, 0x100, 0xff),
            (0x42ff0000, 0x80, 0x7f),
        ],
    ),
    (
        "u.i16.f.f32.rte.sat",
        "i16",
        "float",
        [
            (0x3fc00000, 0x2, 0x1),
            (0x46ffff33, 0x8000, 0x7fff),
            (0x437f8000, 0x100, 0xff),
            (0x42ff0000, 0x80, 0x7f),
        ],
    ),
    (
        "s.i64.f.f32.rte.sat",
        "i64",
        "float",
        [
            (0x3fc00000, 0x2, 0x1),
            (0xbfc00000, 0xfffffffffffffffe, 0xffffffffffffffff),
            (0x46ffff33, 0x8000, 0x7fff),
            (0xc7000099, 0xffffffffffff7fff, 0xffffffffffff8000),
        ],
    ),
    (
        "u.i64.f.f32.rte.sat",
        "i64",
        "float",
        [
            (0x3fc00000, 0x2, 0x1),
            (0x46ffff33, 0x8000, 0x7fff),
            (0x437f8000, 0x100, 0xff),
            (0x42ff0000, 0x80, 0x7f),
        ],
    ),
    (
        "s.i32.f.f32.rtp",
        "i32",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "s.i32.f.f32.rtp.sat",
        "i32",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "u.i32.f.f32.rtp.sat",
        "i32",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "s.i8.f.f32.rtp.sat",
        "i8",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "u.i8.f.f32.rtp.sat",
        "i8",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "s.i16.f.f32.rtp.sat",
        "i16",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "u.i16.f.f32.rtp.sat",
        "i16",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "s.i64.f.f32.rtp.sat",
        "i64",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "u.i64.f.f32.rtp.sat",
        "i64",
        "float",
        [
            (0x3f000000, 0x1, 0x0),
            (0x3fc00000, 0x2, 0x1),
            (0x40200000, 0x3, 0x2),
            (0x3effffff, 0x1, 0x0),
        ],
    ),
    (
        "s.i32.f.f32.rtn",
        "i32",
        "float",
        [
            (0xbf000000, 0xffffffff, 0x0),
            (0xbfc00000, 0xfffffffe, 0xffffffff),
            (0xc0200000, 0xfffffffd, 0xfffffffe),
            (0xbeffffff, 0xffffffff, 0x0),
        ],
    ),
    (
        "s.i32.f.f32.rtn.sat",
        "i32",
        "float",
        [
            (0xbf000000, 0xffffffff, 0x0),
            (0xbfc00000, 0xfffffffe, 0xffffffff),
            (0xc0200000, 0xfffffffd, 0xfffffffe),
            (0xbeffffff, 0xffffffff, 0x0),
        ],
    ),
    (
        "u.i32.f.f32.rtn.sat",
        "i32",
        "float",
        [
            (0x3f000000, 0x0, 0x0),
            (0x3fc00000, 0x1, 0x1),
            (0x40200000, 0x2, 0x2),
            (0xbf000000, 0x0, 0x0),
        ],
    ),
    (
        "s.i8.f.f32.rtn.sat",
        "i8",
        "float",
        [
            (0xbf000000, 0xff, 0x0),
            (0xbfc00000, 0xfe, 0xff),
            (0xc0200000, 0xfd, 0xfe),
            (0xbeffffff, 0xff, 0x0),
        ],
    ),
    (
        "u.i8.f.f32.rtn.sat",
        "i8",
        "float",
        [
            (0x3f000000, 0x0, 0x0),
            (0x3fc00000, 0x1, 0x1),
            (0x40200000, 0x2, 0x2),
            (0xbf000000, 0x0, 0x0),
        ],
    ),
    (
        "s.i16.f.f32.rtn.sat",
        "i16",
        "float",
        [
            (0xbf000000, 0xffff, 0x0),
            (0xbfc00000, 0xfffe, 0xffff),
            (0xc0200000, 0xfffd, 0xfffe),
            (0xbeffffff, 0xffff, 0x0),
        ],
    ),
    (
        "u.i16.f.f32.rtn.sat",
        "i16",
        "float",
        [
            (0x3f000000, 0x0, 0x0),
            (0x3fc00000, 0x1, 0x1),
            (0x40200000, 0x2, 0x2),
            (0xbf000000, 0x0, 0x0),
        ],
    ),
    (
        "s.i64.f.f32.rtn.sat",
        "i64",
        "float",
        [
            (0xbf000000, 0xffffffffffffffff, 0x0),
            (0xbfc00000, 0xfffffffffffffffe, 0xffffffffffffffff),
            (0xc0200000, 0xfffffffffffffffd, 0xfffffffffffffffe),
            (0xbeffffff, 0xffffffffffffffff, 0x0),
        ],
    ),
    (
        "u.i64.f.f32.rtn.sat",
        "i64",
        "float",
        [
            (0x3f000000, 0x0, 0x0),
            (0x3fc00000, 0x1, 0x1),
            (0x40200000, 0x2, 0x2),
            (0xbf000000, 0x0, 0x0),
        ],
    ),
    (
        "u.i8.s.i32.rtz.sat",
        "i8",
        "i32",
        [
            (0xffffffff, 0x0, 0xff),
            (0x1000000, 0xff, 0x0),
            (0x1000001, 0xff, 0x1),
            (0x1000002, 0xff, 0x2),
        ],
    ),
    (
        "s.i8.s.i32.rtz.sat",
        "i8",
        "i32",
        [
            (0x1000000, 0x7f, 0x0),
            (0x1000001, 0x7f, 0x1),
            (0x1000002, 0x7f, 0x2),
            (0x1000003, 0x7f, 0x3),
        ],
    ),
    (
        "u.i8.u.i32.rtz.sat",
        "i8",
        "i32",
        [
            (0x1000000, 0xff, 0x0),
            (0x1000001, 0xff, 0x1),
            (0x1000002, 0xff, 0x2),
            (0x1000003, 0xff, 0x3),
        ],
    ),
    (
        "s.i8.u.i32.rtz.sat",
        "i8",
        "i32",
        [
            (0xffffffff, 0x7f, 0xff),
            (0x1000000, 0x7f, 0x0),
            (0x1000001, 0x7f, 0x1),
            (0x1000002, 0x7f, 0x2),
        ],
    ),
    (
        "u.i16.s.i32.rtz.sat",
        "i16",
        "i32",
        [
            (0xffffffff, 0x0, 0xffff),
            (0x1000000, 0xffff, 0x0),
            (0x1000001, 0xffff, 0x1),
            (0x1000002, 0xffff, 0x2),
        ],
    ),
    (
        "s.i16.s.i32.rtz.sat",
        "i16",
        "i32",
        [
            (0x1000000, 0x7fff, 0x0),
            (0x1000001, 0x7fff, 0x1),
            (0x1000002, 0x7fff, 0x2),
            (0x1000003, 0x7fff, 0x3),
        ],
    ),
    (
        "u.i16.u.i32.rtz.sat",
        "i16",
        "i32",
        [
            (0x1000000, 0xffff, 0x0),
            (0x1000001, 0xffff, 0x1),
            (0x1000002, 0xffff, 0x2),
            (0x1000003, 0xffff, 0x3),
        ],
    ),
    (
        "s.i16.u.i32.rtz.sat",
        "i16",
        "i32",
        [
            (0xffffffff, 0x7fff, 0xffff),
            (0x1000000, 0x7fff, 0x0),
            (0x1000001, 0x7fff, 0x1),
            (0x1000002, 0x7fff, 0x2),
        ],
    ),
    (
        "u.i32.s.i32.rtz.sat",
        "i32",
        "i32",
        [
            (0xffffffff, 0x0, 0xffffffff),
            (0xfeffffff, 0x0, 0xfeffffff),
            (0xfefffffd, 0x0, 0xfefffffd),
            (0xfe000001, 0x0, 0xfe000001),
        ],
    ),
    (
        "s.i32.u.i32.rtz.sat",
        "i32",
        "i32",
        [
            (0xffffffff, 0x7fffffff, 0xffffffff),
            (0xfeffffff, 0x7fffffff, 0xfeffffff),
            (0xfefffffd, 0x7fffffff, 0xfefffffd),
            (0xfe000001, 0x7fffffff, 0xfe000001),
        ],
    ),
    (
        "u.i8.s.i32.rte.sat",
        "i8",
        "i32",
        [
            (0xffffffff, 0x0, 0xff),
            (0x1000000, 0xff, 0x0),
            (0x1000001, 0xff, 0x1),
            (0x1000002, 0xff, 0x2),
        ],
    ),
    (
        "s.i16.s.i32.rtp",
        "i16",
        "i32",
        [
            (0x0, 0x0, 0x0),
            (0x1, 0x1, 0x1),
            (0xffffffff, 0xffff, 0xffff),
            (0x1000000, 0x0, 0x0),
        ],
    ),
    (
        "u.i64.s.i32.rtz.sat",
        "i64",
        "i32",
        [
            (0xffffffff, 0x0, 0xffffffffffffffff),
            (0xfeffffff, 0x0, 0xfffffffffeffffff),
            (0xfefffffd, 0x0, 0xfffffffffefffffd),
            (0xfe000001, 0x0, 0xfffffffffe000001),
        ],
    ),
    (
        "s.i32.s.i64.rtz.sat",
        "i32",
        "i64",
        [
            (0x20000000000001, 0x7fffffff, 0x1),
            (0x7fffffffffffffff, 0x7fffffff, 0xffffffff),
            (0x8000000000000000, 0x80000000, 0x0),
            (0x8000000000000001, 0x80000000, 0x1),
        ],
    ),
    (
        "u.i32.s.i64.rtz.sat",
        "i32",
        "i64",
        [
            (0xffffffffffffffff, 0x0, 0xffffffff),
            (0x20000000000001, 0xffffffff, 0x1),
            (0x8000000000000001, 0x0, 0x1),
            (0x4000000000000001, 0xffffffff, 0x1),
        ],
    ),
    (
        "s.i32.u.i64.rtz.sat",
        "i32",
        "i64",
        [
            (0xffffffffffffffff, 0x7fffffff, 0xffffffff),
            (0x20000000000001, 0x7fffffff, 0x1),
            (0x7fffffffffffffff, 0x7fffffff, 0xffffffff),
            (0x8000000000000000, 0x7fffffff, 0x0),
        ],
    ),
    (
        "u.i32.u.i64.rtz.sat",
        "i32",
        "i64",
        [
            (0x20000000000001, 0xffffffff, 0x1),
            (0x8000000000000000, 0xffffffff, 0x0),
            (0x8000000000000001, 0xffffffff, 0x1),
            (0x4000000000000001, 0xffffffff, 0x1),
        ],
    ),
    (
        "s.i64.u.i64.rtz.sat",
        "i64",
        "i64",
        [
            (0xffffffffffffffff, 0x7fffffffffffffff, 0xffffffffffffffff),
            (0x8000000000000000, 0x7fffffffffffffff, 0x8000000000000000),
            (0x8000000000000001, 0x7fffffffffffffff, 0x8000000000000001),
            (0xffffff8000000000, 0x7fffffffffffffff, 0xffffff8000000000),
        ],
    ),
    (
        "u.i64.s.i64.rtz.sat",
        "i64",
        "i64",
        [
            (0xffffffffffffffff, 0x0, 0xffffffffffffffff),
            (0x8000000000000000, 0x0, 0x8000000000000000),
            (0x8000000000000001, 0x0, 0x8000000000000001),
            (0xffffff8000000000, 0x0, 0xffffff8000000000),
        ],
    ),
];

/// The bytes of an LLVM type's value.
fn size_of(ty: &str) -> usize {
    match ty {
        "i8" => 1,
        "i16" => 2,
        "i64" => 8,
        _ => 4,
    }
}

/// A kernel storing `name`'s conversion of four loaded operands of type `src` as `dst`: buffer 0
/// the results, buffer 1 the operands, each access aligned to its type.
fn aligned_kernel(dst: &str, src: &str, name: &str) -> String {
    let (da, sa) = (size_of(dst), size_of(src));
    let mut body = String::new();
    for k in 0..4 {
        body.push_str(&format!(
            "  %p{k} = getelementptr inbounds {src}, ptr addrspace(1) %1, i64 {k}\n  %x{k} = load {src}, ptr addrspace(1) %p{k}, align {sa}\n  %r{k} = call {dst} @{name}({src} %x{k})\n  %q{k} = getelementptr inbounds {dst}, ptr addrspace(1) %0, i64 {k}\n  store {dst} %r{k}, ptr addrspace(1) %q{k}, align {da}\n"
        ));
    }
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @convert(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1) {{\nentry:\n{body}  ret void\n}}\n\n\
declare {dst} @{name}({src})\n\n\
!llvm.module.flags = !{{!0}}\n!air.kernel = !{{!20}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n\
!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {da}, !\"air.arg_type_align_size\", i32 {da}, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {sa}, !\"air.arg_type_align_size\", i32 {sa}, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"in\"}}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n!20 = !{{ptr @convert, !10, !11}}\n"
    )
}

/// A kernel buffer at `index` holding `bytes`, padded to a multiple of 16.
fn kernel_buffer(index: u32, mut bytes: Vec<u8>) -> Buffer {
    while bytes.len() % 16 != 0 {
        bytes.push(0);
    }
    Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    }
}

/// The four results of entry `convert` of `bitcode` on `p`, in OpenCL mode where `open_cl_mode`,
/// over `operands` (each `src` bytes wide), each `dst` bytes wide: the typed module run on the
/// interpreter, in OpenCL mode with the three hidden buffers bound before the kernel's own.
fn interpret(
    bitcode: &[u8],
    p: &DeviceProfile,
    open_cl_mode: bool,
    (dst, src): (usize, usize),
    operands: [u64; 4],
) -> [u64; 4] {
    let mut m = meter();
    let decoded = air::decode(bitcode, &mut m).expect("decode");
    let req = SpecRequest {
        call_stack_depth: 1,
        entry: "convert".to_owned(),
        open_cl_mode,
        ..SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, &mut m).expect("specialize");
    let memory = analyze_memory(&spec, p, &mut m).expect("memory");
    let mut inputs = LowerInputs::new();
    inputs.open_cl_mode = open_cl_mode;
    let plan = select(&spec, &memory, p, &inputs, &mut m).expect("select");
    let module = emit_module(&spec, &plan, &mut m).expect("emit").module;
    let input: Vec<u8> = operands
        .iter()
        .flat_map(|v| v.to_le_bytes()[..src].to_vec())
        .collect();
    let first = if open_cl_mode { 3 } else { 0 };
    let mut buffers: Vec<Buffer> = (0..first).map(|i| kernel_buffer(i, vec![0; 16])).collect();
    buffers.push(kernel_buffer(first, vec![0xEE; 4 * dst]));
    buffers.push(kernel_buffer(first + 1, input));
    let res = Resources {
        buffers,
        push: vec![0; 128],
        spec: Default::default(),
    };
    let out = run(&module, &Config::default(), res).expect("interpreter");
    let words = &out.buffers[first as usize];
    std::array::from_fn(|k| {
        let mut b = [0u8; 8];
        b[..dst].copy_from_slice(&words[k * dst..(k + 1) * dst]);
        u64::from_le_bytes(b)
    })
}

/// In OpenCL mode every measured form answers the 5300M's and the 630's words, through native
/// 64-bit integers and through the wide path, and passes L3; outside OpenCL mode the same kernel
/// answers the conversion its types name.
#[test]
fn in_opencl_mode_a_suffixed_convert_answers_metal_s_words() {
    let profiles = [
        ("lavapipe", profile("lavapipe", true)),
        ("moltenvk-m3 words", profile("moltenvk-m3", false)),
    ];
    let mut honored = 0;
    for (form, dst, src, measured) in &MEASURED {
        let name = format!("air.convert.{form}");
        let Some(bc) = assemble(&aligned_kernel(dst, src, &name)) else {
            return;
        };
        let operands = measured.map(|(x, _, _)| x);
        for (stem, p) in &profiles {
            let mut inputs = LowerInputs::new();
            inputs.open_cl_mode = true;
            let a = compile_with(&bc, "convert", p, &inputs)
                .unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{name} on {stem}: L3: {e}"));
            let sizes = (size_of(dst), size_of(src));
            let got = interpret(&bc, p, true, sizes, operands);
            assert_eq!(
                got,
                measured.map(|(_, metal, _)| metal),
                "{name} on {stem} in OpenCL mode, operands {operands:x?}"
            );
            let plain = interpret(&bc, p, false, sizes, operands);
            assert_eq!(
                plain,
                measured.map(|(_, _, plain)| plain),
                "{name} on {stem} outside OpenCL mode, operands {operands:x?}"
            );
            honored += measured.iter().filter(|(_, m, q)| m != q).count();
        }
    }
    eprintln!(
        "{} forms answered Metal's OpenCL-mode words, {honored} of their operands where the suffix \
         changes the answer",
        MEASURED.len()
    );
}
