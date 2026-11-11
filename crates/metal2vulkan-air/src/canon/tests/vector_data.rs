//! OpenCL C's vector data functions (`air.vload*`, `air.vstore*`) become their element accesses:
//! lane `i` at element `offset * stride + i` of the data, a `half` widened or narrowed for the
//! `half` forms. The module is LLVM text written here, assembled with `llvm-as`; a run without it
//! skips.

use super::*;

const HEAD: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v29-apple-macosx27.0.0\"\n";

/// `@vd(device const uint *g, device half *h, device uint *o, threadgroup char *t)` at constant
/// offsets: `vload3(5, g)` stored by `vstore3(_, 1, o)`; `vload_half3(2, h) + vloada_half3(2, h)`
/// stored by `vstorea_half3_rtz(_, 3, h)`; `vload_half(7, h) + 1` stored by `vstore_half_rtp(_, 20,
/// h)`; a `char2` through `vstore2(vload2(1, t), 4, t)`.
const KERNEL: &str = "define void @vd(ptr addrspace(1) %g, ptr addrspace(1) %h, ptr addrspace(1) %o, ptr addrspace(3) %t) #0 {\n\
  %a = tail call <3 x i32> @air.vload.global.v3i32(i64 5, ptr addrspace(1) %g) #1\n\
  tail call void @air.vstore.global.v3i32(<3 x i32> %a, i64 1, ptr addrspace(1) %o) #2\n\
  %b = tail call <3 x float> @air.vload_half.global.v3f32(i64 2, ptr addrspace(1) %h) #1\n\
  %c = tail call <3 x float> @air.vloada_half.global.v3f32(i64 2, ptr addrspace(1) %h) #1\n\
  %s = fadd <3 x float> %b, %c\n\
  tail call void @air.vstorea_half.global.rtz.v3f32(<3 x float> %s, i64 3, ptr addrspace(1) %h) #2\n\
  %x = tail call float @air.vload_half.global.f32(i64 7, ptr addrspace(1) %h) #1\n\
  %x1 = fadd float %x, 1.0\n\
  tail call void @air.vstore_half.global.rtp.f32(float %x1, i64 20, ptr addrspace(1) %h) #2\n\
  %y = tail call <2 x i8> @air.vload.local.v2i8(i64 1, ptr addrspace(3) %t) #1\n\
  tail call void @air.vstore.local.v2i8(<2 x i8> %y, i64 4, ptr addrspace(3) %t) #2\n\
  ret void\n}\n\
declare <3 x i32> @air.vload.global.v3i32(i64, ptr addrspace(1)) #1\n\
declare void @air.vstore.global.v3i32(<3 x i32>, i64, ptr addrspace(1)) #2\n\
declare <3 x float> @air.vload_half.global.v3f32(i64, ptr addrspace(1)) #1\n\
declare <3 x float> @air.vloada_half.global.v3f32(i64, ptr addrspace(1)) #1\n\
declare void @air.vstorea_half.global.rtz.v3f32(<3 x float>, i64, ptr addrspace(1)) #2\n\
declare float @air.vload_half.global.f32(i64, ptr addrspace(1)) #1\n\
declare void @air.vstore_half.global.rtp.f32(float, i64, ptr addrspace(1)) #2\n\
declare <2 x i8> @air.vload.local.v2i8(i64, ptr addrspace(3)) #1\n\
declare void @air.vstore.local.v2i8(<2 x i8>, i64, ptr addrspace(3)) #2\n\
attributes #0 = { nounwind }\n\
attributes #1 = { nounwind willreturn memory(argmem: read) }\n\
attributes #2 = { nounwind willreturn memory(argmem: write) }\n\
!air.kernel = !{!0}\n\
!air.version = !{!7}\n\
!air.language_version = !{!8}\n\
!0 = !{ptr @vd, !1, !2}\n\
!1 = !{}\n\
!2 = !{!3, !4, !5, !6}\n\
!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"g\"}\n\
!4 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2, !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"h\"}\n\
!5 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"o\"}\n\
!6 = !{i32 3, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"char\", !\"air.arg_name\", !\"t\"}\n\
!7 = !{i32 2, i32 9, i32 0}\n\
!8 = !{!\"Metal\", i32 1, i32 1, i32 0}\n";

/// The type of an access's element, as a token.
fn element(ty: Option<&Type>) -> &'static str {
    match ty {
        Some(Type::Int(8)) => "i8",
        Some(Type::Int(32)) => "i32",
        Some(Type::Half) => "half",
        other => panic!("{other:?}"),
    }
}

#[test]
fn opencl_vector_data_functions_become_their_element_accesses() {
    let test = "opencl_vector_data_functions_become_their_element_accesses";
    let bitcode = match metal2vulkan_devtools::llvm_assemble_text(&format!("{HEAD}{KERNEL}")) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP {test}: {e:?}");
            return;
        }
    };
    let d = crate::bitcode::decode(&bitcode, &mut meter()).expect("decoded");
    let req = SpecRequest {
        entry: "vd".into(),
        ..SpecRequest::default()
    };
    let e = specialize(d.module, &d.abi, &req, &mut meter()).expect("specialized");
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("the entry's body");
    let consts = e.module.consts();
    let types = e.module.types();
    // Each access as (store, parameter, element, index): its pointer a GEP of the element from
    // a parameter at a folded constant index.
    let at = |ptr: Value| -> (usize, &'static str, u64) {
        let Value::Inst(id) = ptr else {
            panic!("{ptr:?}");
        };
        match body.insts.get(id).map(|i| &i.kind) {
            Some(InstKind::Gep {
                src_ty,
                base: Value::Arg(a),
                indices,
                ..
            }) => match indices.as_ref() {
                [Value::Const(c)] => match consts.get(*c).map(|k| &k.value) {
                    Some(Const::Int(i)) => (a.index(), element(types.get(*src_ty)), *i),
                    other => panic!("{other:?}"),
                },
                other => panic!("{other:?}\n{}", print(&e.module)),
            },
            other => panic!("{other:?}\n{}", print(&e.module)),
        }
    };
    let mut accesses = Vec::new();
    let (mut widened, mut narrowed) = (0, 0);
    for inst in body.insts.values() {
        match &inst.kind {
            InstKind::Load { ptr, align, .. } => {
                accesses.push((false, at(*ptr), align.map(Align::bytes)));
            }
            InstKind::Store { ptr, align, .. } => {
                accesses.push((true, at(*ptr), align.map(Align::bytes)));
            }
            InstKind::Cast {
                op: CastOp::FPExt, ..
            } => widened += 1,
            InstKind::Cast {
                op: CastOp::FPTrunc,
                ..
            } => narrowed += 1,
            InstKind::Call(c) => assert!(
                !matches!(
                    c.callee,
                    Callee::Air {
                        op: AirIntrinsic::VectorData(_),
                        ..
                    }
                ),
                "{}",
                print(&e.module)
            ),
            _ => {}
        }
    }
    accesses.sort();
    let (load, store) = (false, true);
    #[rustfmt::skip]
    let expected = vec![
        // `vload3(5, g)`: `uint`s 15, 16 and 17, each at its own alignment.
        (load, (0, "i32", 15), Some(4)), (load, (0, "i32", 16), Some(4)), (load, (0, "i32", 17), Some(4)),
        // `vload_half3(2, h)`: halves 6, 7 and 8; `vloada_half3(2, h)`: 8, 9 and 10 (a stride of
        // 4); `vload_half(7, h)`: half 7.
        (load, (1, "half", 6), Some(2)), (load, (1, "half", 7), Some(2)), (load, (1, "half", 7), Some(2)),
        (load, (1, "half", 8), Some(2)), (load, (1, "half", 8), Some(2)), (load, (1, "half", 9), Some(2)),
        (load, (1, "half", 10), Some(2)),
        // `vload2(1, t)`: `char`s 2 and 3.
        (load, (3, "i8", 2), Some(1)), (load, (3, "i8", 3), Some(1)),
        // `vstorea_half3_rtz(_, 3, h)`: halves 12, 13 and 14; `vstore_half_rtp(_, 20, h)`: 20.
        (store, (1, "half", 12), Some(2)), (store, (1, "half", 13), Some(2)), (store, (1, "half", 14), Some(2)),
        (store, (1, "half", 20), Some(2)),
        // `vstore3(_, 1, o)`: `uint`s 3, 4 and 5.
        (store, (2, "i32", 3), Some(4)), (store, (2, "i32", 4), Some(4)), (store, (2, "i32", 5), Some(4)),
        // `vstore2(_, 4, t)`: `char`s 8 and 9.
        (store, (3, "i8", 8), Some(1)), (store, (3, "i8", 9), Some(1)),
    ];
    assert_eq!(accesses, expected, "{}", print(&e.module));
    // Seven halves read are widened, four stored narrowed: the device's conversion for every
    // rounding the names spell.
    assert_eq!((widened, narrowed), (7, 4), "{}", print(&e.module));
}
