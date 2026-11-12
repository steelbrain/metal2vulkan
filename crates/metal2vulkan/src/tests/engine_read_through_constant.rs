//! A kernel buffer declared read-only in `device` space whose parameter is a `constant`-space
//! pointer, as Apple's compiler declares a by-value argument (`air.address_space 1`, a one-byte
//! `char`), compiles on every profile as the same kernel declaring the `constant` buffer it reads.
//!
//! The kernel is ours, `out[i] = p->rect + p->m[i & 3] * p->a + float4(p->n[i % 3], p->e)` over a
//! 160-byte `constant Params *p` (`float4 rect; float4x4 m; float3x3 n; float a, b, c, d, e;`),
//! written here as LLVM text with the buffer's node in either form. MSL has no spelling for the
//! rewritten node; Metal takes it (`abi::check`'s `read_through_constant`), and the runtime's
//! `read_through_constant` device test runs both forms against the M3 Max's answers.
//!
//! `llvm-as` assembles the modules; without it the test skips visibly. The owned fixture
//! `engine/kernel_read_through_constant` holds the same kernel in MSL with the rewritten node, and
//! its typed-pointer text (`metal-as`, macOS only) points the parameter at the 160-byte `Params`
//! while the node states one `char`.

use crate::air::abi::{Access, ParamRole};
use crate::air::ir::AddrSpace;
use crate::budget::{Budget, Meter};
use crate::device::DevicePolicy;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, llvm_assemble_text, metal_assemble,
    profile, AssembleError,
};

/// The node of the `Params` buffer as Metal's compiler writes it for `constant Params *p`.
const AS_CONSTANT: &str = "!\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 160, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"Params\"";
/// The node as Apple's compiler writes it for a by-value argument: `device`, one `char`.
const AS_DEVICE_CHAR: &str = "!\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"char\"";

/// The kernel `k`, its `Params` buffer's node holding `node`.
fn kernel(node: &str) -> String {
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @k(ptr addrspace(2) noundef readonly %0, ptr addrspace(1) noundef %1, i32 noundef %2) {{\n\
entry:\n\
  %rect = load <4 x float>, ptr addrspace(2) %0, align 16\n\
  %j = and i32 %2, 3\n\
  %j64 = zext i32 %j to i64\n\
  %jo = shl i64 %j64, 4\n\
  %mo = add i64 %jo, 16\n\
  %mp = getelementptr inbounds i8, ptr addrspace(2) %0, i64 %mo\n\
  %row = load <4 x float>, ptr addrspace(2) %mp, align 16\n\
  %ap = getelementptr inbounds i8, ptr addrspace(2) %0, i64 128\n\
  %a = load float, ptr addrspace(2) %ap, align 16\n\
  %r = urem i32 %2, 3\n\
  %r64 = zext i32 %r to i64\n\
  %ro = shl i64 %r64, 4\n\
  %no = add i64 %ro, 80\n\
  %np = getelementptr inbounds i8, ptr addrspace(2) %0, i64 %no\n\
  %n = load <3 x float>, ptr addrspace(2) %np, align 16\n\
  %ep = getelementptr inbounds i8, ptr addrspace(2) %0, i64 144\n\
  %e = load float, ptr addrspace(2) %ep, align 16\n\
  %a0 = insertelement <4 x float> poison, float %a, i64 0\n\
  %as = shufflevector <4 x float> %a0, <4 x float> poison, <4 x i32> zeroinitializer\n\
  %scaled = fmul <4 x float> %row, %as\n\
  %sum = fadd <4 x float> %rect, %scaled\n\
  %n4 = shufflevector <3 x float> %n, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>\n\
  %ne = insertelement <4 x float> %n4, float %e, i64 3\n\
  %v = fadd <4 x float> %sum, %ne\n\
  %i64 = zext i32 %2 to i64\n\
  %op = getelementptr inbounds <4 x float>, ptr addrspace(1) %1, i64 %i64\n\
  store <4 x float> %v, ptr addrspace(1) %op, align 16\n\
  ret void\n\
}}\n\n\
!llvm.module.flags = !{{!0}}\n!air.kernel = !{{!1}}\n!air.version = !{{!7}}\n!air.language_version = !{{!8}}\n\
!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n\
!1 = !{{ptr @k, !2, !3}}\n!2 = !{{}}\n!3 = !{{!4, !5, !6}}\n\
!4 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, {node}, !\"air.arg_name\", !\"p\"}}\n\
!5 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"out\"}}\n\
!6 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"i\"}}\n\
!7 = !{{i32 2, i32 8, i32 0}}\n!8 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// Both forms of the kernel compile and pass L3 on every profile here, the rewritten one keeping
/// its buffer's declared `device` space and read access.
#[test]
fn a_read_only_device_buffer_read_through_a_constant_pointer_compiles() {
    let assemble = |node| match llvm_assemble_text(&kernel(node)) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    };
    let (Some(constant), Some(device)) = (assemble(AS_CONSTANT), assemble(AS_DEVICE_CHAR)) else {
        return;
    };
    let mut compiled = 0;
    for stem in ["lavapipe", "moltenvk-m3", "amd-780m-radv", "vk12-floor"] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (form, bc) in [("constant", &constant), ("device char", &device)] {
            let what = format!("{form} on {stem}");
            let a = compile(bc, "k", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            compiled += 1;
        }
    }
    let abi = crate::air::bitcode::decode(&device, &mut Meter::new(&Budget::default()))
        .expect("the module decodes")
        .abi;
    let ParamRole::Buffer(b) = &abi.entries[0].params[0].role else {
        panic!("{:?}", abi.entries[0].params[0].role);
    };
    assert_eq!((b.space, b.access), (AddrSpace::Device, Access::Read));
    eprintln!("compiled {compiled} modules of the two forms");
}

/// The fixture's typed-pointer form, whose parameter points at the 160-byte `Params` while its node
/// states one `char`, compiles on every profile here to the module its opaque form compiles to: a
/// plain buffer's `air.arg_type_size` is its declared type's, which its pointee need not be.
#[test]
fn the_typed_fixture_compiles_as_its_opaque_form_does() {
    let dir = engine_dir();
    let typed = match metal_assemble(&dir.join("kernel_read_through_constant.air.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("metal-as rejected the fixture: {why}"),
    };
    let opaque = llvm_assemble(&dir.join("kernel_read_through_constant.ll"))
        .unwrap_or_else(|e| panic!("llvm-as: {e:?}"));
    for stem in ["lavapipe", "moltenvk-m3", "amd-780m-radv", "vk12-floor"] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let [typed, opaque] = [&typed, &opaque].map(|bc| {
            compile(bc, "read_through_constant", &p).unwrap_or_else(|e| panic!("{stem}: {e}"))
        });
        assert_eq!(typed.spirv, opaque.spirv, "{stem}");
    }
}
