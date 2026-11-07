//! OpenCL mode (`SpecRequest::open_cl_mode`): a kernel in the shape Apple's OpenCL compiler writes
//! takes the three hidden buffers after its own parameters, each buffer-indexed argument 3 past its
//! index, and each work-item read a read of its built-in or of the first buffer. Without the mode
//! the same kernel specializes as Metal AIR, its OpenCL-only reads left for lowering to refuse.
//! The modules are LLVM text written here, assembled with `llvm-as`; a run without it skips.

use super::*;
use crate::abi::Builtin;
use crate::intrinsics::WorkItem;

const HEAD: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v29-apple-macosx27.0.0\"\n";

/// `@k(device uint *x [[buffer(0)]], constant uint *c [[buffer(1)]], threadgroup uint *t
/// [[threadgroup(0)]])`: `x[0..6]` take `get_global_id(0)`, `get_global_offset(1)`,
/// `get_work_dim()`, `get_local_size(4)`, `get_global_id(c[0])` and `get_num_groups(c[0])`, and
/// `t[0]` takes 0; each read through an internal wrapper widening it, as the OpenCL compiler
/// writes them.
const KERNEL: &str = "define void @k(ptr addrspace(1) %x, ptr addrspace(2) %c, ptr addrspace(3) %t) #0 {\n\
  %d = load i32, ptr addrspace(2) %c, align 4\n\
  %g = tail call fastcc i64 @gid(i32 0)\n\
  %o = tail call i32 @air.get_global_offset.i32(i32 1)\n\
  %w = tail call i32 @air.get_work_dim.i32()\n\
  %s = tail call i32 @air.get_local_size.i32(i32 4)\n\
  %v = tail call fastcc i64 @gid(i32 %d)\n\
  %n = tail call i32 @air.get_num_groups.i32(i32 %d)\n\
  %g32 = trunc i64 %g to i32\n\
  store i32 %g32, ptr addrspace(1) %x, align 4\n\
  %x1 = getelementptr inbounds i32, ptr addrspace(1) %x, i64 1\n\
  store i32 %o, ptr addrspace(1) %x1, align 4\n\
  %x2 = getelementptr inbounds i32, ptr addrspace(1) %x, i64 2\n\
  store i32 %w, ptr addrspace(1) %x2, align 4\n\
  %x3 = getelementptr inbounds i32, ptr addrspace(1) %x, i64 3\n\
  store i32 %s, ptr addrspace(1) %x3, align 4\n\
  %v32 = trunc i64 %v to i32\n\
  %x4 = getelementptr inbounds i32, ptr addrspace(1) %x, i64 4\n\
  store i32 %v32, ptr addrspace(1) %x4, align 4\n\
  %x5 = getelementptr inbounds i32, ptr addrspace(1) %x, i64 5\n\
  store i32 %n, ptr addrspace(1) %x5, align 4\n\
  store i32 0, ptr addrspace(3) %t, align 4\n\
  ret void\n}\n\
define internal fastcc i64 @gid(i32 %dim) #1 {\n\
  %r = tail call i32 @air.get_global_id.i32(i32 %dim) #2\n\
  %z = zext i32 %r to i64\n\
  ret i64 %z\n}\n\
declare i32 @air.get_global_id.i32(i32) #2\n\
declare i32 @air.get_global_offset.i32(i32) #2\n\
declare i32 @air.get_work_dim.i32() #2\n\
declare i32 @air.get_local_size.i32(i32) #2\n\
declare i32 @air.get_num_groups.i32(i32) #2\n\
attributes #0 = { nounwind }\n\
attributes #1 = { nounwind optsize }\n\
attributes #2 = { nounwind readnone willreturn }\n\
!air.kernel = !{!0}\n\
!air.version = !{!6}\n\
!air.language_version = !{!7}\n\
!0 = !{ptr @k, !1, !2}\n\
!1 = !{}\n\
!2 = !{!3, !4, !5}\n\
!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"x\"}\n\
!4 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"c\"}\n\
!5 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}\n\
!6 = !{i32 2, i32 9, i32 0}\n\
!7 = !{!\"Metal\", i32 1, i32 1, i32 0}\n";

/// [`KERNEL`] specialized, in OpenCL mode where `open_cl_mode`; `None` without `llvm-as`.
fn specialized(test: &str, open_cl_mode: bool) -> Option<Entry> {
    let bitcode = match metal2vulkan_devtools::llvm_assemble_text(&format!("{HEAD}{KERNEL}")) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP {test}: {e:?}");
            return None;
        }
    };
    let d = crate::bitcode::decode(&bitcode, &mut meter()).expect("decoded");
    let req = SpecRequest {
        entry: "k".into(),
        open_cl_mode,
        ..SpecRequest::default()
    };
    Some(specialize(d.module, &d.abi, &req, &mut meter()).expect("specialized"))
}

/// The slot index of each of the entry's resource parameters, by parameter and role.
fn slots(e: &Entry) -> Vec<(u32, &'static str, u32)> {
    e.ep.params
        .iter()
        .filter_map(|p| {
            let (kind, b) = match &p.role {
                ParamRole::Buffer(b) => ("buffer", b),
                ParamRole::Threadgroup(b) => ("threadgroup", b),
                _ => return None,
            };
            let AbiInt::Const(index) = b.location.index else {
                panic!("{p:?}");
            };
            Some((p.param, kind, index))
        })
        .collect()
}

/// The entry's body.
fn body(e: &Entry) -> &crate::ir::Body {
    e.module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("the entry's body")
}

/// The `air.*` calls the entry's body still makes that `keep` selects.
fn calls(e: &Entry, keep: impl Fn(&AirIntrinsic) -> bool) -> usize {
    body(e)
        .insts
        .values()
        .filter(|i| matches!(&i.kind, InstKind::Call(c) if matches!(&c.callee, Callee::Air { op, .. } if keep(op))))
        .count()
}

#[test]
fn opencl_mode_moves_buffer_indices_past_the_three_hidden_buffers() {
    let test = "opencl_mode_moves_buffer_indices_past_the_three_hidden_buffers";
    let (Some(plain), Some(cl)) = (specialized(test, false), specialized(test, true)) else {
        return;
    };
    assert_eq!(
        slots(&plain),
        [(0, "buffer", 0), (1, "buffer", 1), (2, "threadgroup", 0)]
    );
    // The kernel's buffers 3 on, its threadgroup memory where it was, the hidden buffers after
    // its own parameters at 0 to 2.
    assert_eq!(
        slots(&cl),
        [
            (0, "buffer", 3),
            (1, "buffer", 4),
            (2, "threadgroup", 0),
            (3, "buffer", 0),
            (4, "buffer", 1),
            (5, "buffer", 2)
        ]
    );
    let hidden: Vec<_> = cl.ep.params[3..6]
        .iter()
        .map(|p| {
            let ParamRole::Buffer(b) = &p.role else {
                panic!("{p:?}");
            };
            (
                p.name.as_ref().map(ToString::to_string),
                p.ty.name.as_ref().map(ToString::to_string),
                p.ty.size,
                b.access,
                b.space,
            )
        })
        .collect();
    let named = |name: &str, ty: &str, size, access, space| {
        (
            Some(name.to_owned()),
            Some(ty.to_owned()),
            Some(size),
            access,
            space,
        )
    };
    assert_eq!(
        hidden,
        [
            named(
                "__global_offset_and_num_dims",
                "uint",
                4,
                Access::Read,
                AddrSpace::Constant
            ),
            named(
                "__printf_buffer",
                "char",
                1,
                Access::ReadWrite,
                AddrSpace::Device
            ),
            named(
                "__image_order_data_type_table",
                "uint",
                4,
                Access::Read,
                AddrSpace::Constant
            ),
        ]
    );
}

#[test]
fn opencl_work_item_reads_become_built_ins_and_the_first_hidden_buffer() {
    let test = "opencl_work_item_reads_become_built_ins_and_the_first_hidden_buffer";
    let (Some(plain), Some(cl)) = (specialized(test, false), specialized(test, true)) else {
        return;
    };
    let work_item = |op: &AirIntrinsic| matches!(op, AirIntrinsic::WorkItem(_));
    let builtin = |op: &AirIntrinsic| matches!(op, AirIntrinsic::GlobalBuiltin(_));
    // As Metal AIR: the offset and the dimension count stay calls, as do the reads at dimension
    // 4 and at the loaded dimension, which lowering refuses.
    assert_eq!(calls(&plain, work_item), 2);
    assert_eq!(calls(&plain, builtin), 3);
    // In OpenCL mode no work-item call is left.
    assert_eq!(calls(&cl, work_item), 0);
    assert_eq!(calls(&cl, builtin), 0);
    // The words of the first hidden buffer (parameter 3) read: the offset `get_global_id(0)`
    // adds, `get_global_offset(1)`, the dimension count, and one offset for each lane the loaded
    // dimension may name.
    let words = body(&cl)
        .insts
        .values()
        .filter(|i| matches!(i.kind, InstKind::Gep { base: Value::Arg(a), .. } if a.index() == 3))
        .count();
    assert_eq!(words, 6);
    // `get_local_size(4)` is OpenCL's 1, stored as a constant.
    let stored_one = body(&cl).insts.values().any(|i| match i.kind {
        InstKind::Store {
            val: Value::Const(c),
            ..
        } => matches!(
            cl.module.consts().get(c).map(|k| &k.value),
            Some(Const::Int(1))
        ),
        _ => false,
    });
    assert!(stored_one, "{}", print(&cl.module));
    // The built-ins the lanes read: the grid position and the threadgroup count.
    let builtins: Vec<_> = cl
        .ep
        .params
        .iter()
        .filter_map(|p| match p.role {
            ParamRole::Builtin(b) => Some(b),
            _ => None,
        })
        .collect();
    assert_eq!(
        builtins,
        [Builtin::ThreadPositionInGrid, Builtin::ThreadgroupsPerGrid]
    );
    // The intrinsic the plain entry leaves is the one OpenCL mode reads from the buffer.
    let left: Vec<_> = body(&plain)
        .insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::Call(c) => match c.callee {
                Callee::Air {
                    op: AirIntrinsic::WorkItem(w),
                    ..
                } => Some(w),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(left, [WorkItem::GlobalOffset, WorkItem::WorkDim]);
}
