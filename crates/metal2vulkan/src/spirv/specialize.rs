//! `specialize`: a module's specialization constants made the fixed constants of their values.
//!
//! Vulkan states a stage's code valid "after applying the specializations provided in
//! `pSpecializationInfo`, if any, and then converting all specialization constants into fixed
//! constants" (VUID-VkPipelineShaderStageCreateInfo-pSpecializationInfo-06849). [`specialize`] does
//! that conversion before a driver sees the words, so the module a driver compiles holds no
//! specialization constant: each is the plain constant of its value, each `OpSpecConstantOp` the
//! constant it computes, and a `WorkgroupSize` composite's sizes are the literals of the entry
//! point's `LocalSize` execution mode.

use std::collections::BTreeMap;

use crate::error::{Error, InternalError, InternalKind};
use crate::spirv::ir::SpecOp;
use crate::spv::{BuiltIn, Decoration, ExecutionMode, Op};

/// The scalar type of a constant.
#[derive(Clone, Copy)]
enum Scalar {
    Bool,
    /// An integer or float of this width; `signed` for a signed integer.
    Bits {
        width: u32,
        signed: bool,
    },
}

/// The instructions the pass reads or rewrites; it copies every other as it is.
const READ: [Op; 17] = [
    Op::Decorate,
    Op::DecorationGroup,
    Op::EntryPoint,
    Op::ExecutionMode,
    Op::ExecutionModeId,
    Op::TypeBool,
    Op::TypeInt,
    Op::TypeFloat,
    Op::ConstantTrue,
    Op::ConstantFalse,
    Op::Constant,
    Op::ConstantComposite,
    Op::SpecConstantTrue,
    Op::SpecConstantFalse,
    Op::SpecConstant,
    Op::SpecConstantComposite,
    Op::SpecConstantOperation,
];

/// `words`, a module this crate wrote, with every specialization constant made the fixed constant
/// of its value: `values` gives `(SpecId, bits)` as a `VkSpecializationInfo` would (the low bits of
/// the constant's width; for a `bool`, true where nonzero), and a constant `values` does not name
/// keeps its default. Each `OpSpecConstantOp` becomes the constant it computes and each
/// `OpSpecConstantComposite` an `OpConstantComposite`; the `SpecId` decorations go. Where the
/// module declares a `WorkgroupSize` composite, its `BuiltIn` decoration goes too and the entry
/// point takes the composite's values as its `LocalSize` execution mode, as a compute shader of a
/// literal workgroup size states it. The result is valid where `words` is, under the same
/// requirements, and runs as `words` specialized by `values` does.
///
/// Err: [`Error::Internal`], naming what it met, where the words are not such a module: a
/// malformed instruction; an instruction the translator does not write, whose constants the pass
/// would leave half specialized (an `OpExecutionModeId`, a `LocalSizeId` mode among them; an
/// `OpDecorationGroup`; an `OpSpecConstantOp` of another opcode); or a `WorkgroupSize` beside a
/// `LocalSize` mode or beside other than one entry point.
pub fn specialize(words: &[u32], values: &[(u32, u64)]) -> Result<Vec<u32>, Error> {
    let header = words
        .get(..5)
        .filter(|h| h.first() == Some(&crate::spv::MAGIC_NUMBER))
        .ok_or_else(|| malformed("no SPIR-V header"))?;
    let values: BTreeMap<u32, u64> = values.iter().copied().collect();
    let mut out = Vec::with_capacity(words.len() + 6);
    out.extend_from_slice(header);
    let mut spec_ids = BTreeMap::new();
    let mut workgroup = None;
    let mut entries = Vec::new();
    let mut local_size = false;
    let mut types = BTreeMap::new();
    // Each scalar constant's value and type, by result id.
    let mut known: BTreeMap<u32, (u64, Scalar)> = BTreeMap::new();
    let mut size = None;
    let mut rest = words.get(5..).unwrap_or_default();
    while let Some(&first) = rest.first() {
        let count = (first >> 16) as usize;
        let (inst, after) = match rest.split_at_checked(count) {
            Some(split) if count > 0 => split,
            _ => return Err(malformed("an instruction past the module's end")),
        };
        rest = after;
        let opcode = first & 0xFFFF;
        let operand = |i: usize| {
            (inst.get(i).copied()).ok_or_else(|| malformed("an instruction short of its operands"))
        };
        let tail = |i: usize| inst.get(i..).unwrap_or_default();
        let op = READ.into_iter().find(|&op| op as u32 == opcode);
        match op {
            Some(Op::Decorate) if operand(2)? == Decoration::SpecId as u32 => {
                spec_ids.insert(operand(1)?, operand(3)?);
                continue;
            }
            Some(Op::Decorate)
                if operand(2)? == Decoration::BuiltIn as u32
                    && operand(3)? == BuiltIn::WorkgroupSize as u32 =>
            {
                workgroup = Some(operand(1)?);
                continue;
            }
            // A mode of id operands (`LocalSizeId` among them) could name a size the pass would
            // set a second time, and a group's decorations a `SpecId` it would miss.
            Some(unknown @ (Op::ExecutionModeId | Op::DecorationGroup)) => {
                return Err(malformed(format!(
                    "an Op{unknown:?}, which the translator does not write"
                )));
            }
            Some(Op::EntryPoint) => {
                out.extend_from_slice(inst);
                entries.push((operand(2)?, out.len()));
                continue;
            }
            Some(Op::ExecutionMode) => {
                local_size |= operand(2)? == ExecutionMode::LocalSize as u32;
            }
            Some(Op::TypeBool) => {
                types.insert(operand(1)?, Scalar::Bool);
            }
            Some(Op::TypeInt) => {
                let (width, signed) = (operand(2)?, operand(3)? != 0);
                types.insert(operand(1)?, Scalar::Bits { width, signed });
            }
            Some(Op::TypeFloat) => {
                let width = operand(2)?;
                types.insert(
                    operand(1)?,
                    Scalar::Bits {
                        width,
                        signed: false,
                    },
                );
            }
            Some(Op::ConstantTrue | Op::ConstantFalse) => {
                let value = u64::from(opcode == Op::ConstantTrue as u32);
                known.insert(operand(2)?, (value, Scalar::Bool));
            }
            Some(Op::Constant) => {
                if let Some(&scalar) = types.get(&operand(1)?) {
                    known.insert(operand(2)?, (literal(tail(3)), scalar));
                }
            }
            Some(Op::SpecConstantTrue | Op::SpecConstantFalse | Op::SpecConstant) => {
                let (ty, id) = (operand(1)?, operand(2)?);
                let scalar = scalar(&types, ty)?;
                let default = match op {
                    Some(Op::SpecConstant) => literal(tail(3)),
                    _ => u64::from(opcode == Op::SpecConstantTrue as u32),
                };
                let given = spec_ids.get(&id).and_then(|spec| values.get(spec));
                let value = masked(scalar, given.copied().unwrap_or(default));
                known.insert(id, (value, scalar));
                fixed(&mut out, ty, id, scalar, value);
                continue;
            }
            Some(Op::SpecConstantOperation) => {
                let (ty, id) = (operand(1)?, operand(2)?);
                let code = operand(3)?;
                let unknown = || malformed(format!("an OpSpecConstantOp of opcode {code}"));
                let op = SpecOp::of_opcode(code).ok_or_else(unknown)?;
                let operands: Vec<(u64, Scalar)> = (tail(4).iter())
                    .map(|o| known.get(o).copied())
                    .collect::<Option<_>>()
                    .ok_or_else(|| malformed("an operand that is no scalar constant"))?;
                // The integer operands' width: a select's arms', else the first operand's.
                let integer = operands.get(usize::from(op == SpecOp::Select));
                let width = match integer {
                    Some((_, Scalar::Bits { width, .. })) => u8::try_from(*width).unwrap_or(64),
                    _ => 1,
                };
                let bits: Vec<u64> = operands.iter().map(|&(v, _)| v).collect();
                let value = op
                    .eval(width, &bits)
                    .ok_or_else(|| malformed("a specialization operation of another arity"))?;
                let scalar = scalar(&types, ty)?;
                let value = masked(scalar, value);
                known.insert(id, (value, scalar));
                fixed(&mut out, ty, id, scalar, value);
                continue;
            }
            Some(Op::SpecConstantComposite | Op::ConstantComposite) => {
                if workgroup == Some(operand(2)?) {
                    let sizes: Vec<u32> = (tail(3).iter())
                        .map(|p| known.get(p).and_then(|&(v, _)| u32::try_from(v).ok()))
                        .collect::<Option<_>>()
                        .ok_or_else(|| malformed("a workgroup size of other parts"))?;
                    size = Some(sizes);
                }
                out.push((first & !0xFFFF) | Op::ConstantComposite as u32);
                out.extend_from_slice(tail(1));
                continue;
            }
            _ => {}
        }
        out.extend_from_slice(inst);
    }
    let Some(sizes) = size else {
        return match workgroup {
            Some(_) => Err(malformed("a WorkgroupSize decoration of no composite")),
            None => Ok(out),
        };
    };
    let ([(entry, at)], false, [x, y, z]) = (entries.as_slice(), local_size, sizes.as_slice())
    else {
        return Err(malformed(
            "a WorkgroupSize of other than three parts, beside a local-size mode, or beside other \
             than one entry point",
        ));
    };
    let mode = [
        (6 << 16) | Op::ExecutionMode as u32,
        *entry,
        ExecutionMode::LocalSize as u32,
        *x,
        *y,
        *z,
    ];
    out.splice(*at..*at, mode);
    Ok(out)
}

fn malformed(detail: impl std::fmt::Display) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        format!("specialize: {detail}"),
    ))
}

/// The scalar type `ty`.
fn scalar(types: &BTreeMap<u32, Scalar>, ty: u32) -> Result<Scalar, Error> {
    (types.get(&ty).copied()).ok_or_else(|| malformed("a constant of no scalar type"))
}

/// A literal of one or two words, low word first, as an `OpConstant` holds it.
fn literal(words: &[u32]) -> u64 {
    match words {
        [lo] => u64::from(*lo),
        [lo, hi, ..] => u64::from(*lo) | (u64::from(*hi) << 32),
        [] => 0,
    }
}

/// `v` cut to `scalar`'s width (a `bool` is 0 or 1).
fn masked(scalar: Scalar, v: u64) -> u64 {
    match scalar {
        Scalar::Bool => u64::from(v != 0),
        Scalar::Bits { width, .. } if width >= 64 => v,
        Scalar::Bits { width, .. } => v & ((1u64 << width) - 1),
    }
}

/// Write the fixed constant `id` of `ty` with `value`: `OpConstantTrue` or `OpConstantFalse` for a
/// `bool`, else an `OpConstant` of one word, or of two past 32 bits, where a signed integer under
/// 32 bits is sign-extended to its word, as SPIR-V's literal rule asks.
fn fixed(out: &mut Vec<u32>, ty: u32, id: u32, scalar: Scalar, value: u64) {
    match scalar {
        Scalar::Bool => {
            let op = match value != 0 {
                true => Op::ConstantTrue,
                false => Op::ConstantFalse,
            };
            out.extend([(3 << 16) | op as u32, ty, id]);
        }
        Scalar::Bits { width, .. } if width > 32 => {
            let (lo, hi) = (value as u32, (value >> 32) as u32);
            out.extend([(5 << 16) | Op::Constant as u32, ty, id, lo, hi]);
        }
        Scalar::Bits { width, signed } => {
            let negative = signed && (1..32).contains(&width) && (value >> (width - 1)) & 1 == 1;
            let word = match negative {
                true => (value | (u64::MAX << width)) as u32,
                false => value as u32,
            };
            out.extend([(4 << 16) | Op::Constant as u32, ty, id, word]);
        }
    }
}
