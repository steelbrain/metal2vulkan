//! Argument-buffer layouts: where `MTLArgumentEncoder` writes each argument, from a function's
//! `air.struct_type_info`. Every expected offset, length and
//! alignment here is what Metal's encoder answered on an M3 Max and an M4 (probes, CPU only: the
//! bytes each setter changed in a buffer).

use crate::bind::{ArgumentBufferLayout, ArgumentLayoutError, ArgumentType as T};
use crate::budget::Budget;
use crate::library::{FunctionConstantValues, Library};
use crate::pipeline::testing::AssembleError;

/// Argument `index`'s offset and type.
fn at(layout: &ArgumentBufferLayout, index: u32) -> (u64, T) {
    let place = layout
        .argument(index)
        .unwrap_or_else(|| panic!("argument {index}"));
    (place.offset, place.ty)
}

/// `(encodedLength, alignment)`.
fn sized(layout: &ArgumentBufferLayout) -> (u64, u64) {
    (layout.encoded_length, layout.alignment)
}

/// Synthetic AIR (opaque pointers, empty bodies). Kernels `kA` and `kH` state the
/// `air.struct_type_info` that Metal's compiler gives the probe's `A` and probe 3's `H` (MSL of our
/// own; nodes !9 to !92). `kAA` takes an array of arrays, and `kP` a plain buffer.
const MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @kA(ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %i) {
  ret void
}
define void @kH(ptr addrspace(1) %h, ptr addrspace(1) %out) {
  ret void
}
define void @kAA(ptr addrspace(2) %k, ptr addrspace(1) %out) {
  ret void
}
define void @kP(ptr addrspace(1) %p) {
  ret void
}
!air.kernel = !{!9, !76, !200, !210}
!air.version = !{!113}
!air.language_version = !{!114}
!9 = !{ptr @kA, !10, !11}
!10 = !{}
!11 = !{!12, !34, !35}
!12 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 256, !"air.location_index", i32 0,
  i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !13,
  !"air.arg_type_size", i32 256, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"A",
  !"air.arg_name", !"a"}
!13 = !{i32 0, i32 8, i32 0, !"float", !"buf", !"air.indirect_argument", !14, i32 8, i32 8,
  i32 0, !"texture2d<float, sample>", !"tex", !"air.indirect_argument", !15, i32 16, i32 8,
  i32 0, !"sampler", !"smp", !"air.indirect_argument", !16, i32 24, i32 4, i32 0, !"float",
  !"c1", !"air.indirect_argument", !17, i32 32, i32 16, i32 0, !"float4", !"c4",
  !"air.indirect_argument", !18, i32 48, i32 24, i32 0, !"array<texture2d<float, sample>, 3>",
  !"tarr", !"air.indirect_argument", !19, i32 72, i32 8, i32 2, !"texture2d<float, sample>",
  !"carr", !"air.indirect_argument", !20, i32 88, i32 4, i32 3, !"int", !"ci",
  !"air.indirect_argument", !21, i32 104, i32 8, i32 0, !"Inner", !"nested",
  !"air.indirect_argument", !22, !"air.struct_type_info", !23, i32 112, i32 24, i32 0, !"Inner",
  !"byval", !"air.indirect_argument", i32 30, i32 136, i32 8, i32 0, !"command_buffer", !"icb",
  !"air.indirect_argument", !27, i32 144, i32 8, i32 0, !"compute_pipeline_state", !"cps",
  !"air.indirect_argument", !28, i32 152, i32 8, i32 0, !"render_pipeline_state", !"rps",
  !"air.indirect_argument", !29, i32 160, i32 1, i32 0, !"uchar", !"u8",
  !"air.indirect_argument", !30, i32 164, i32 4, i32 0, !"half2", !"h2",
  !"air.indirect_argument", !31, !"air.struct_type_info", !23, i32 168, i32 24, i32 2, !"Inner",
  !"sarr", !"air.indirect_argument", i32 50, i32 216, i32 8, i32 2, !"float", !"bufs",
  !"air.indirect_argument", !32, i32 232, i32 8, i32 2, !"sampler", !"smps",
  !"air.indirect_argument", !33}
!14 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4,
  !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
!15 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample",
  !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!16 = !{i32 2, !"air.sampler", !"air.location_index", i32 2, i32 1, !"air.arg_type_name",
  !"sampler", !"air.arg_name", !"smp"}
!17 = !{i32 3, !"air.indirect_constant", !"air.location_index", i32 3, i32 1,
  !"air.arg_type_name", !"float", !"air.arg_name", !"c1"}
!18 = !{i32 4, !"air.indirect_constant", !"air.location_index", i32 4, i32 1,
  !"air.arg_type_name", !"float4", !"air.arg_name", !"c4"}
!19 = !{i32 5, !"air.texture", !"air.location_index", i32 5, i32 3, !"air.sample",
  !"air.arg_type_name", !"array<texture2d<float, sample>, 3>", !"air.arg_name", !"tarr"}
!20 = !{i32 6, !"air.texture", !"air.location_index", i32 8, i32 1, !"air.sample",
  !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"carr"}
!21 = !{i32 7, !"air.indirect_constant", !"air.location_index", i32 10, i32 1,
  !"air.arg_type_name", !"int", !"air.arg_name", !"ci"}
!22 = !{i32 8, !"air.indirect_buffer", !"air.location_index", i32 20, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.struct_type_info", !23, !"air.arg_type_size", i32 24,
  !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Inner", !"air.arg_name", !"nested"}
!23 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !24,
  i32 8, i32 8, i32 0, !"float", !"b", !"air.indirect_argument", !25, i32 16, i32 4, i32 0,
  !"float", !"c", !"air.indirect_argument", !26}
!24 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample",
  !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!25 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4,
  !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!26 = !{i32 2, !"air.indirect_constant", !"air.location_index", i32 2, i32 1,
  !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
!27 = !{i32 10, !"air.command_buffer", !"air.location_index", i32 40, i32 1,
  !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"icb"}
!28 = !{i32 11, !"air.compute_pipeline_state", !"air.location_index", i32 41, i32 1,
  !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"cps"}
!29 = !{i32 12, !"air.render_pipeline_state", !"air.location_index", i32 42, i32 1,
  !"air.arg_type_name", !"render_pipeline_state", !"air.arg_name", !"rps"}
!30 = !{i32 13, !"air.indirect_constant", !"air.location_index", i32 44, i32 1,
  !"air.arg_type_name", !"uchar", !"air.arg_name", !"u8"}
!31 = !{i32 14, !"air.indirect_constant", !"air.location_index", i32 45, i32 1,
  !"air.arg_type_name", !"half2", !"air.arg_name", !"h2"}
!32 = !{i32 16, !"air.buffer", !"air.location_index", i32 60, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4,
  !"air.arg_type_name", !"float", !"air.arg_name", !"bufs"}
!33 = !{i32 17, !"air.sampler", !"air.location_index", i32 62, i32 1, !"air.arg_type_name",
  !"sampler", !"air.arg_name", !"smps"}
!34 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4,
  !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!35 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name",
  !"i"}
!42 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4,
  !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!76 = !{ptr @kH, !10, !77}
!77 = !{!78, !42}
!78 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 80, !"air.location_index", i32 2,
  i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !79,
  !"air.arg_type_size", i32 80, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"H",
  !"air.arg_name", !"h"}
!79 = !{!"air.struct_type_info", !80, i32 0, i32 16, i32 2, !"Gap", !"garr",
  !"air.indirect_argument", i32 70, i32 32, i32 8, i32 0, !"Plain", !"pp",
  !"air.indirect_argument", !82, !"air.struct_type_info", !84, i32 40, i32 8, i32 0, !"Plain",
  !"pl", !"air.indirect_argument", i32 91, i32 48, i32 8, i32 0, !"Inner2", !"n2",
  !"air.indirect_argument", !87, i32 56, i32 8, i32 2, !"Inner", !"np", !"air.indirect_argument",
  !91, i32 72, i32 4, i32 0, !"float", !"tail", !"air.indirect_argument", !92}
!80 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !24,
  i32 8, i32 8, i32 0, !"float", !"b", !"air.indirect_argument", !81}
!81 = !{i32 1, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4,
  !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!82 = !{i32 1, !"air.buffer", !"air.location_index", i32 90, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.struct_type_info", !83, !"air.arg_type_size", i32 8,
  !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Plain", !"air.arg_name", !"pp"}
!83 = !{i32 0, i32 4, i32 0, !"float", !"a", i32 4, i32 4, i32 0, !"int", !"b"}
!84 = !{i32 0, i32 4, i32 0, !"float", !"a", !"air.indirect_argument", !85, i32 4, i32 4, i32 0,
  !"int", !"b", !"air.indirect_argument", !86}
!85 = !{i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1,
  !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!86 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1,
  !"air.arg_type_name", !"int", !"air.arg_name", !"b"}
!87 = !{i32 3, !"air.indirect_buffer", !"air.location_index", i32 94, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.struct_type_info", !88, !"air.arg_type_size", i32 16,
  !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Inner2", !"air.arg_name", !"n2"}
!88 = !{i32 0, i32 8, i32 0, !"Inner", !"in", !"air.indirect_argument", !89, i32 8, i32 4, i32 0,
  !"float", !"k", !"air.indirect_argument", !90}
!89 = !{i32 0, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.struct_type_info", !23, !"air.arg_type_size", i32 24,
  !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Inner", !"air.arg_name", !"in"}
!90 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1,
  !"air.arg_type_name", !"float", !"air.arg_name", !"k"}
!91 = !{i32 4, !"air.indirect_buffer", !"air.location_index", i32 100, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.struct_type_info", !23, !"air.arg_type_size", i32 24,
  !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Inner", !"air.arg_name", !"np"}
!92 = !{i32 5, !"air.indirect_constant", !"air.location_index", i32 110, i32 1,
  !"air.arg_type_name", !"float", !"air.arg_name", !"tail"}
!113 = !{i32 2, i32 8, i32 0}
!114 = !{!"Metal", i32 4, i32 0, i32 0}
!200 = !{ptr @kAA, !10, !201}
!201 = !{!202, !206}
!202 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 56, !"air.location_index", i32 0,
  i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !203,
  !"air.arg_type_size", i32 56, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"K",
  !"air.arg_name", !"k"}
!203 = !{i32 0, i32 16, i32 3, !"array<texture2d<float, sample>, 2>", !"t2",
  !"air.indirect_argument", !204, i32 48, i32 4, i32 0, !"float", !"tail",
  !"air.indirect_argument", !205}
!204 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 2, !"air.sample",
  !"air.arg_type_name", !"array<texture2d<float, sample>, 2>", !"air.arg_name", !"t2"}
!205 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 6, i32 1,
  !"air.arg_type_name", !"float", !"air.arg_name", !"tail"}
!206 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4,
  !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!210 = !{ptr @kP, !10, !211}
!211 = !{!212}
!212 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write",
  !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4,
  !"air.arg_type_name", !"float", !"air.arg_name", !"p"}
"#;

/// The layout of the argument buffer `kernel` of [`MODULE`] takes at `index`; `None` where
/// `llvm-as` is missing.
fn function_layout(
    kernel: &str,
    index: u32,
) -> Option<Result<ArgumentBufferLayout, ArgumentLayoutError>> {
    let bitcode = match metal2vulkan_devtools::llvm_assemble_text(MODULE) {
        Ok(bitcode) => bitcode,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return None;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    };
    let library = Library::from_air(&bitcode).expect("the module decodes");
    let function = library.function(kernel).expect("the kernel");
    Some(function.argument_buffer_layout(
        index,
        &FunctionConstantValues::defaults(),
        &Budget::default(),
    ))
}

#[test]
fn a_functions_argument_buffer_is_laid_out_as_its_air_states() {
    let Some(a) = function_layout("kA", 0) else {
        return;
    };
    let a = a.expect("kA's argument buffer");
    // Metal's encoder: encodedLength 256, alignment 16, and each setter's bytes.
    assert_eq!(sized(&a), (256, 16));
    let expected = [
        (0, 0, T::Buffer),
        (1, 8, T::Texture),
        (2, 16, T::Sampler),
        (3, 24, T::Constant),
        (4, 32, T::Constant),
        (5, 48, T::Texture),
        (6, 56, T::Texture),
        (7, 64, T::Texture),
        (8, 72, T::Texture),
        (9, 80, T::Texture),
        (10, 88, T::Constant),
        (11, 92, T::Constant),
        (12, 96, T::Constant),
        (20, 104, T::Buffer),
        (30, 112, T::Texture),
        (31, 120, T::Buffer),
        (32, 128, T::Constant),
        (40, 136, T::IndirectCommandBuffer),
        (41, 144, T::ComputePipelineState),
        (42, 152, T::RenderPipelineState),
        (44, 160, T::Constant),
        (45, 164, T::Constant),
        (50, 168, T::Texture),
        (51, 176, T::Buffer),
        (52, 184, T::Constant),
        (53, 192, T::Texture),
        (54, 200, T::Buffer),
        (55, 208, T::Constant),
        (60, 216, T::Buffer),
        (61, 224, T::Buffer),
        (62, 232, T::Sampler),
        (63, 240, T::Sampler),
    ];
    for (index, offset, ty) in expected {
        assert_eq!(at(&a, index), (offset, ty), "argument {index}");
    }
    for index in [13, 15, 21, 33, 43, 56, 64, 1000] {
        assert!(a.argument(index).is_none(), "argument {index}");
    }
    assert_eq!(a.argument(44).map(|p| p.size), Some(1));
    assert_eq!(a.argument(4).map(|p| p.size), Some(16));
    // A pointer to `Inner` is a nested argument buffer (24 bytes, aligned to 8); a pointer to a
    // float is not.
    let nested = a
        .argument(20)
        .and_then(|p| p.nested)
        .expect("a nested layout");
    assert_eq!(sized(&nested), (24, 8));
    assert_eq!(
        [at(&nested, 0), at(&nested, 1), at(&nested, 2)],
        [(0, T::Texture), (8, T::Buffer), (16, T::Constant)]
    );
    assert!(a.argument(0).is_some_and(|p| p.nested.is_none()));
    assert!(a.argument(31).is_some_and(|p| p.nested.is_none()));
}

#[test]
fn struct_arrays_take_their_span_of_indices_per_element() {
    let Some(h) = function_layout("kH", 2) else {
        return;
    };
    let h = h.expect("kH's argument buffer");
    assert_eq!(sized(&h), (80, 8));
    // `Gap garr[2]` from 70, `Gap` holding ids 0 and 5: element 1 starts at 76.
    let expected = [
        (70, 0, T::Texture),
        (75, 8, T::Buffer),
        (76, 16, T::Texture),
        (81, 24, T::Buffer),
        (90, 32, T::Buffer),
        (91, 40, T::Constant),
        (92, 44, T::Constant),
        (94, 48, T::Buffer),
        (100, 56, T::Buffer),
        (101, 64, T::Buffer),
        (110, 72, T::Constant),
    ];
    for (index, offset, ty) in expected {
        assert_eq!(at(&h, index), (offset, ty), "argument {index}");
    }
    for index in [71, 74, 77, 82, 93] {
        assert!(h.argument(index).is_none(), "argument {index}");
    }
    // A pointer to a struct of plain members is a plain buffer (Metal answers no encoder).
    assert!(h.argument(90).is_some_and(|p| p.nested.is_none()));
    // Nested twice: `Inner2` (16, 8) holds a pointer to `Inner` (24, 8).
    let n2 = h.argument(94).and_then(|p| p.nested).expect("Inner2");
    assert_eq!(sized(&n2), (16, 8));
    assert_eq!(at(&n2, 1), (8, T::Constant));
    let inner = n2.argument(0).and_then(|p| p.nested).expect("Inner");
    assert_eq!(sized(&inner), (24, 8));
    for index in [100, 101] {
        let np = h.argument(index).and_then(|p| p.nested).expect("Inner");
        assert_eq!(sized(&np), (24, 8));
    }
}

#[test]
fn what_metal_makes_no_function_encoder_for_is_refused() {
    let Some(aa) = function_layout("kAA", 0) else {
        return;
    };
    assert_eq!(aa, Err(ArgumentLayoutError::ArrayOfArrays { id: 0 }));
    for (kernel, index) in [("kP", 0), ("kA", 1), ("kA", 2), ("kH", 0)] {
        assert_eq!(
            function_layout(kernel, index).expect("assembled"),
            Err(ArgumentLayoutError::NoArgumentBuffer { index }),
            "{kernel} {index}"
        );
    }
}
