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

/// `@img(texture2d a [[texture(0)]], texture1d b [[texture(1)]], texture1d_array c
/// [[texture(2)]], texture1d<write> d [[texture(3)]], sampler s [[sampler(0)]], device int *o
/// [[buffer(0)]], device float4 *f [[buffer(1)]])` as Apple's OpenCL compiler writes OpenCL C's
/// image functions (macOS 27.0.1, through internal wrappers where it uses one): `o[0..6]` take
/// `get_image_channel_order(b)`, `get_image_channel_data_type(c)`, the order of `a`, the data
/// type of `d`, `get_image_width(b)` and `get_image_array_size(c)`; `f[0]` reads `b` at `x`, `f[1]`
/// samples layer 1 of `c` at `x`, and `d` takes `f[1]`'s texel at `x`.
const IMAGES: &str = "define void @img(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(2) %s, ptr addrspace(1) %o, ptr addrspace(1) %f) #0 {\n\
  %x = tail call i32 @air.get_global_id.i32(i32 0)\n\
  %ob = tail call fastcc i32 @_Z23get_image_channel_order14ocl_image1d_ro(ptr addrspace(1) %b)\n\
  store i32 %ob, ptr addrspace(1) %o, align 4\n\
  %dc = tail call i32 @air.get_channel_data_type_texture_1d_array(ptr addrspace(1) %c)\n\
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 1\n\
  store i32 %dc, ptr addrspace(1) %o1, align 4\n\
  %oa = tail call i32 @air.get_channel_order_texture_2d(ptr addrspace(1) %a)\n\
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 2\n\
  store i32 %oa, ptr addrspace(1) %o2, align 4\n\
  %da = tail call i32 @air.get_channel_data_type_texture_1d(ptr addrspace(1) %d)\n\
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 3\n\
  store i32 %da, ptr addrspace(1) %o3, align 4\n\
  %wb = tail call i32 @air.get_width_texture_1d(ptr addrspace(1) %b, i32 0)\n\
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 4\n\
  store i32 %wb, ptr addrspace(1) %o4, align 4\n\
  %nc = tail call i32 @air.get_array_size_texture_1d_array(ptr addrspace(1) %c)\n\
  %o5 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 5\n\
  store i32 %nc, ptr addrspace(1) %o5, align 4\n\
  %rs = tail call ptr addrspace(2) @air.get_read_sampler()\n\
  %rb = tail call { <4 x float>, i8 } @air.read_texture_1d.v4f32(ptr addrspace(1) %b, ptr addrspace(2) %rs, i32 %x, i32 0, i32 1)\n\
  %tb = extractvalue { <4 x float>, i8 } %rb, 0\n\
  store <4 x float> %tb, ptr addrspace(1) %f, align 16\n\
  %fx = sitofp i32 %x to float\n\
  %sc = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) %c, ptr addrspace(2) %s, float %fx, i32 1, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
  %tc = extractvalue { <4 x float>, i8 } %sc, 0\n\
  %f1 = getelementptr inbounds <4 x float>, ptr addrspace(1) %f, i64 1\n\
  store <4 x float> %tc, ptr addrspace(1) %f1, align 16\n\
  tail call void @air.write_texture_1d.v4f32(ptr addrspace(1) %d, i32 %x, <4 x float> %tc, i32 0, i32 2)\n\
  ret void\n}\n\
define internal fastcc i32 @_Z23get_image_channel_order14ocl_image1d_ro(ptr addrspace(1) %0) #1 {\n\
  %2 = tail call i32 @air.get_channel_order_texture_1d(ptr addrspace(1) %0) #2\n\
  ret i32 %2\n}\n\
declare i32 @air.get_global_id.i32(i32) #2\n\
declare i32 @air.get_channel_order_texture_1d(ptr addrspace(1)) #2\n\
declare i32 @air.get_channel_data_type_texture_1d_array(ptr addrspace(1)) #2\n\
declare i32 @air.get_channel_order_texture_2d(ptr addrspace(1)) #2\n\
declare i32 @air.get_channel_data_type_texture_1d(ptr addrspace(1)) #2\n\
declare i32 @air.get_width_texture_1d(ptr addrspace(1), i32) #2\n\
declare i32 @air.get_array_size_texture_1d_array(ptr addrspace(1)) #2\n\
declare ptr addrspace(2) @air.get_read_sampler() #2\n\
declare { <4 x float>, i8 } @air.read_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32, i32) #2\n\
declare { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1), ptr addrspace(2), float, i32, i1, i32, i1, float, float, i32) #2\n\
declare void @air.write_texture_1d.v4f32(ptr addrspace(1), i32, <4 x float>, i32, i32) #3\n\
attributes #0 = { nounwind }\n\
attributes #1 = { nounwind optsize }\n\
attributes #2 = { nounwind readonly willreturn }\n\
attributes #3 = { nounwind willreturn }\n\
!air.kernel = !{!0}\n\
!air.version = !{!10}\n\
!air.language_version = !{!11}\n\
!0 = !{ptr @img, !1, !2}\n\
!1 = !{}\n\
!2 = !{!3, !4, !5, !6, !7, !8, !9}\n\
!3 = !{i32 0, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<unknown, read>\", !\"air.arg_name\", !\"a\"}\n\
!4 = !{i32 1, !\"air.texture\", !\"air.location_index\", i32 1, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture1d<unknown, read>\", !\"air.arg_name\", !\"b\"}\n\
!5 = !{i32 2, !\"air.texture\", !\"air.location_index\", i32 2, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture1d_array<unknown, read>\", !\"air.arg_name\", !\"c\"}\n\
!6 = !{i32 3, !\"air.texture\", !\"air.location_index\", i32 3, i32 1, !\"air.write\", !\"air.arg_type_name\", !\"texture1d<unknown, write>\", !\"air.arg_name\", !\"d\"}\n\
!7 = !{i32 4, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n\
!8 = !{i32 5, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"int\", !\"air.arg_name\", !\"o\"}\n\
!9 = !{i32 6, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"f\"}\n\
!10 = !{i32 2, i32 9, i32 0}\n\
!11 = !{!\"Metal\", i32 1, i32 1, i32 0}\n";

/// [`IMAGES`] specialized, in OpenCL mode where `open_cl_mode`; `None` without `llvm-as`.
fn images(test: &str, open_cl_mode: bool) -> Option<Entry> {
    let bitcode = match metal2vulkan_devtools::llvm_assemble_text(&format!("{HEAD}{IMAGES}")) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP {test}: {e:?}");
            return None;
        }
    };
    let d = crate::bitcode::decode(&bitcode, &mut meter()).expect("decoded");
    let req = SpecRequest {
        entry: "img".into(),
        open_cl_mode,
        ..SpecRequest::default()
    };
    Some(specialize(d.module, &d.abi, &req, &mut meter()).expect("specialized"))
}

/// The texture operations the entry's body calls: each one's operation and texture shape.
fn texture_ops(e: &Entry) -> Vec<(TexOp, crate::intrinsics::ImageDim, bool)> {
    body(e)
        .insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::Call(c) => match c.callee {
                Callee::Air {
                    op: AirIntrinsic::Texture(t),
                    ..
                } => Some((t.op, t.image.dim, t.image.array)),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

#[test]
fn opencl_image_formats_read_their_words_of_the_third_hidden_buffer() {
    use crate::intrinsics::{ImageDim, Query};
    let test = "opencl_image_formats_read_their_words_of_the_third_hidden_buffer";
    let (Some(plain), Some(cl)) = (images(test, false), images(test, true)) else {
        return;
    };
    let format = |op: &(TexOp, ImageDim, bool)| {
        matches!(
            op.0,
            TexOp::Query(Query::ChannelOrder | Query::ChannelDataType)
        )
    };
    // As Metal AIR the four queries stay, for lowering to refuse.
    assert_eq!(texture_ops(&plain).iter().filter(|o| format(o)).count(), 4);
    // In OpenCL mode none is left: each is a load of a word of the third hidden buffer, the
    // seven parameters' ninth, at `2t + 1` for an order and `2t` for a data type, `t` the image's
    // texture index: the order of texture 1, the data type of texture 2, the order of texture 0
    // and the data type of texture 3.
    assert!(
        !texture_ops(&cl).iter().any(format),
        "{}",
        print(&cl.module)
    );
    let consts = cl.module.consts();
    let words: Vec<u64> = body(&cl)
        .insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::Gep {
                base: Value::Arg(a),
                indices,
                ..
            } if a.index() == 9 => match indices.as_ref() {
                [Value::Const(c)] => match consts.get(*c).map(|k| &k.value) {
                    Some(Const::Int(w)) => Some(*w),
                    other => panic!("{other:?}"),
                },
                other => panic!("{other:?}"),
            },
            _ => None,
        })
        .collect();
    assert_eq!(words, [3, 4, 1, 6]);
    let table = cl
        .ep
        .params
        .iter()
        .find(|p| p.param == 9)
        .expect("the table");
    assert_eq!(
        table.name.as_ref().map(ToString::to_string).as_deref(),
        Some("__image_order_data_type_table")
    );
}

#[test]
fn opencl_1d_images_are_read_written_and_queried_as_2d_textures() {
    use crate::intrinsics::{ImageDim, Query};
    let test = "opencl_1d_images_are_read_written_and_queried_as_2d_textures";
    let (Some(plain), Some(cl)) = (images(test, false), images(test, true)) else {
        return;
    };
    let ops = |e: &Entry| {
        let mut v: Vec<_> = texture_ops(e)
            .into_iter()
            .filter(|(op, ..)| {
                !matches!(
                    op,
                    TexOp::Query(Query::ChannelOrder | Query::ChannelDataType)
                )
            })
            .map(|(op, dim, array)| {
                let op = match op {
                    TexOp::Read => "read",
                    TexOp::Sample { .. } => "sample",
                    TexOp::Write => "write",
                    TexOp::Query(Query::Width) => "width",
                    TexOp::Query(Query::ArraySize) => "array_size",
                    other => panic!("{other:?}"),
                };
                (op, dim, array)
            })
            .collect();
        v.sort();
        v
    };
    // As Metal AIR the operations stay those of 1D textures.
    assert_eq!(
        ops(&plain),
        [
            ("array_size", ImageDim::D1, true),
            ("read", ImageDim::D1, false),
            ("sample", ImageDim::D1, true),
            ("width", ImageDim::D1, false),
            ("write", ImageDim::D1, false),
        ]
    );
    // In OpenCL mode each is the 2D texture's, or the 2D array texture's, Apple's OpenCL binds.
    assert_eq!(
        ops(&cl),
        [
            ("array_size", ImageDim::D2, true),
            ("read", ImageDim::D2, false),
            ("sample", ImageDim::D2, true),
            ("width", ImageDim::D2, false),
            ("write", ImageDim::D2, false),
        ],
        "{}",
        print(&cl.module)
    );
    // The sampled coordinate is `(x, 0.5)`, the integer ones `(x, 0)`: each built by inserting
    // `x` into lane 0 of a constant pair.
    let pairs: Vec<Const> = body(&cl)
        .insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::InsertElement {
                vec: Value::Const(c),
                ..
            } => cl.module.consts().get(*c).map(|k| k.value.clone()),
            _ => None,
        })
        .collect();
    let half = Const::Float(u64::from(0.5f32.to_bits()));
    let sampled = pairs.iter().filter(|c| match c {
        Const::Aggregate(lanes) => {
            lanes.len() == 2 && cl.module.consts().get(lanes[1]).map(|k| &k.value) == Some(&half)
        }
        _ => false,
    });
    assert_eq!(sampled.count(), 1, "{}", print(&cl.module));
    // The read's and the write's coordinates from zero.
    assert!(
        pairs.iter().filter(|c| matches!(c, Const::Null)).count() >= 2,
        "{}",
        print(&cl.module)
    );
}

#[test]
fn opencl_image_formats_of_buffer_and_depth_images_stay_undecoded() {
    // Metal makes no pipeline of OpenCL mode that queries their formats (M3 Max, macOS 27.0.1;
    // Radeon Pro 5300M, macOS 26.7.1): the names stay unknown, and are refused as such.
    for name in [
        "air.get_channel_order_texture_buffer_1d",
        "air.get_channel_data_type_texture_buffer_1d",
        "air.get_channel_order_depth_2d",
        "air.get_channel_data_type_depth_2d_array",
    ] {
        assert_eq!(AirIntrinsic::parse(name), None, "{name}");
    }
    for name in [
        "air.get_channel_order_texture_1d",
        "air.get_channel_data_type_texture_1d_array",
        "air.get_channel_order_texture_2d",
        "air.get_channel_data_type_texture_2d_array",
        "air.get_channel_order_texture_3d",
    ] {
        assert!(AirIntrinsic::parse(name).is_some(), "{name}");
    }
}
