//! `[[front_facing]]` in an older front end's ABI: Apple's compiler at an iOS 13 simulator target
//! (metalfe-32023.884) passes a fragment's `bool [[front_facing]]` argument as `i8`, read through
//! `trunc i8 %0 to i1`, where today's passes `i1 zeroext`. The argument's `air.front_facing`
//! metadata names the builtin and its type says how it is held, so either form reads the
//! fragment's face: the `i8` one as 1 or 0. Today's compiler emits no such module and
//! `scripts/regen-air-fixtures` cannot make one, so the module is written here: both forms side
//! by side. `llvm-as` assembles; a missing tool skips visibly. No GPU is used.

use crate::pipeline::testing::{compile, link_and_validate, llvm_assemble_text, AssembleError};
use crate::spirv::testing::committed_profiles;

/// Two fragment functions of one module, each answering `float4(1)` for a front face and
/// `float4(0)` for a back one: `ff_bool` takes the face as `i1 zeroext`, `ff_byte` as `i8`.
const MODULE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-ios13.0.0-simulator"

define <4 x float> @ff_bool(i1 zeroext %0) {
  %r = select i1 %0, <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float> zeroinitializer
  ret <4 x float> %r
}

define <4 x float> @ff_byte(i8 %0) {
  %b = trunc i8 %0 to i1
  %r = select i1 %b, <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float> zeroinitializer
  ret <4 x float> %r
}

!air.fragment = !{!0, !5}
!air.compile_options = !{!6, !7}
!air.version = !{!8}
!air.language_version = !{!9}

!0 = !{ptr @ff_bool, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.front_facing", !"air.arg_type_name", !"bool", !"air.arg_name", !"f"}
!5 = !{ptr @ff_byte, !1, !3}
!6 = !{!"air.compile.denorms_disable"}
!7 = !{!"air.compile.fast_math_enable"}
!8 = !{i32 2, i32 2, i32 0}
!9 = !{!"Metal", i32 2, i32 2, i32 0}
"#;

/// Each `OpSelect` of `words` whose condition is a load of the `FrontFacing` builtin and whose
/// operands are integer constants, as those constants' values.
fn front_facing_reads(words: &[u32]) -> Vec<(u64, u64)> {
    use crate::spirv_binary::{parse_bytes, Operand};
    use crate::spv::{BuiltIn, Decoration, Op};
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = parse_bytes(&bytes).expect("parse");
    let face: Vec<u32> = (parsed.instructions.iter())
        .filter(|i| i.opcode == Op::Decorate)
        .filter_map(|i| match i.operands.as_slice() {
            [Operand::IdRef(id), Operand::Word(d), Operand::Word(b)]
                if *d == Decoration::BuiltIn as u32 && *b == BuiltIn::FrontFacing as u32 =>
            {
                Some(*id)
            }
            _ => None,
        })
        .collect();
    assert_eq!(face.len(), 1, "one FrontFacing variable");
    let loads: Vec<u32> = (parsed.instructions.iter())
        .filter(|i| i.opcode == Op::Load && i.operands.first() == Some(&Operand::IdRef(face[0])))
        .filter_map(|i| i.result_id)
        .collect();
    let constant = |id: u32| -> Option<u64> {
        let i = parsed
            .instructions
            .iter()
            .find(|i| i.result_id == Some(id))?;
        match (i.opcode, i.operands.as_slice()) {
            (Op::Constant, [Operand::Word(v)]) => Some(u64::from(*v)),
            (Op::Constant, [Operand::Wide(v)]) => Some(*v),
            (Op::ConstantNull, []) => Some(0),
            _ => None,
        }
    };
    // Each select on a load of the face, as its two constant operands.
    (parsed.instructions.iter())
        .filter(|i| i.opcode == Op::Select)
        .filter_map(|i| match i.operands.as_slice() {
            [Operand::IdRef(c), Operand::IdRef(a), Operand::IdRef(b)] if loads.contains(c) => {
                Some((constant(*a)?, constant(*b)?))
            }
            _ => None,
        })
        .collect()
}

/// An `i8` `[[front_facing]]` argument translates, links and validates on every committed
/// profile, as the `i1 zeroext` one does: the module reads the `FrontFacing` builtin and holds
/// the argument as 1 for a front face and 0 for a back one, which `trunc` reads as the face.
#[test]
fn an_i8_front_facing_argument_reads_the_face_as_one_or_zero() {
    let bc = match llvm_assemble_text(MODULE) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    };
    let profiles = committed_profiles().expect("profiles");
    for (stem, p) in &profiles {
        for entry in ["ff_bool", "ff_byte"] {
            let a = compile(&bc, entry, p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e:?}"));
            assert_eq!(a.precision, [], "{entry} on {stem}");
            let (_, words) =
                link_and_validate(&a, p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
            let selects = front_facing_reads(&words);
            if entry == "ff_byte" {
                assert_eq!(selects, [(1, 0)], "{entry} on {stem}");
            } else {
                assert_eq!(selects, [], "{entry} on {stem}");
            }
        }
    }
}
