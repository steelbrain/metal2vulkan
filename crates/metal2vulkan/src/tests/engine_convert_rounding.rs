//! `air.convert` with a rounding after its types and `.sat` after it, the forms Apple's OpenCL
//! compiler emits for every `convert_<type>[_sat][_<mode>]`, compile to exactly the module of the
//! conversion their types name, which is how Metal's compilers for the x86 Macs answer them (the
//! measurements are in docs/API.md's accepted differences).
//!
//! Each kernel converts four loaded values by one intrinsic and stores the results. Every rounding
//! of each form, and each with `.sat` where it converts to an integer, compiles on lavapipe (64-bit
//! integers) and on moltenvk-m3 without `shaderInt64` (the wide path) and passes L3, and its
//! module's words and precision notes are the plain conversion's.
//!
//! `llvm-as` assembles the modules; without it the test skips visibly.

use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy, DeviceProfile};
use crate::pipeline::testing::{
    compile, description, link_and_validate, llvm_assemble_text, AssembleError,
};

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
