//! MetalPerformancePrimitives' `matmul2d` ABI.
//!
//! `mpp::tensor_ops::matmul2d` (MetalPerformancePrimitives.framework, `MPPTensorOpsMatMul2d.h`)
//! reaches AIR as calls to functions its implementation header (`__impl/MPPTensorOpsMatMul2dImpl.h`)
//! declares `extern "C"` in section `air.externally_defined`, every one named
//! `__tensorops_impl_matmul2d_op_*`. No module defines them: Metal supplies their bodies when it
//! creates the pipeline. They are declarations of Apple's public SDK, which the Metal front end
//! emits as it emits `air.*` intrinsics, so this module reads them only as that ABI: the
//! declaration's section, the symbol's grammar, and the argument positions the header gives each
//! form. Nothing here keys on a kernel's or a library's own names.
//!
//! The forms (the header's macOS 26.2 spelling; for a deployment target of 27.0 the header spells
//! every `run` in its `_v2` form and keeps the helpers'):
//!
//! - **Cooperative-tensor helpers**, each taking the operand index
//!   (`__matmul2d_cooperative_operand_index`: left 0, right 1, destination 2) and the descriptor
//!   by value, flattened into its seven fields ([`Descriptor`]):
//!   `cooperative_tensor_data_size` (bytes of one thread's storage), `_num_elements` (one
//!   thread's element capacity), `_get_element_pointer` (element `i` of a thread's storage; the
//!   reserved index `0xFFFF` names the storage itself, which the header passes back to `run`),
//!   `_get_element_index`, `_get_coordinate` (an element's two coordinates in the operand's
//!   tensor, written as the given integer type), `_init`, and `_is_valid_element`.
//! - **`run`**, `run_[cooperative_|single_thread_]<left>_<right>_<destination>[_v2]`: the
//!   descriptor by reference, then each operand. An operand spelled `<space>_<type>` (`dv`
//!   device, `tg` threadgroup, `th` thread) is a tensor, passed as a pointer to its descriptor
//!   object followed by its `__tensor_ops_tensor_descriptor_type`; one spelled `<type>` alone is a
//!   cooperative tensor, passed as its storage. `cooperative_` names the forms with a cooperative
//!   operand, `single_thread_` the `execution_thread` scope. `_v2` (deployment target 27.0) adds each
//!   tensor operand's scale type, scale block sizes, and static extents. The last argument of a
//!   simdgroup-scope `run` is the scope's thread count.
//!
//! The cooperative tensors' layout across threads is the implementation's (the header: "the
//! layout of data among lanes is implementation defined"): a guest reaches an element only
//! through these helpers, so a translation chooses its own.

use std::fmt;

/// The section every `matmul2d` ABI declaration carries.
pub const SECTION: &[u8] = b"air.externally_defined";

/// The symbol prefix of the `matmul2d` ABI.
pub const PREFIX: &[u8] = b"__tensorops_impl_matmul2d_op_";

/// `__tensor_ops_datatype_invalid`: the scale type a `_v2` run passes for a tensor operand without
/// scale factors (`tensor_plane_scales`).
pub(crate) const NO_SCALE: u64 = 0;

/// `dynamic_extent` as a `_v2` run passes a tensor's static extent that is dynamic: the `int` -1,
/// zero-extended.
pub(crate) const DYNAMIC_EXTENT: u64 = 0xFFFF_FFFF;

/// The reserved element index (`__tensor_ops_reserved_index`, `USHRT_MAX`): the header passes
/// `get_element_pointer` of it to `run` and to the copy helpers as the storage itself.
pub const RESERVED_INDEX: u16 = u16::MAX;

/// `__matmul2d_cooperative_operand_index`: which operand of the product a cooperative tensor
/// holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperandIndex {
    /// The left operand (0).
    Left,
    /// The right operand (1).
    Right,
    /// The destination (2).
    Destination,
}

impl OperandIndex {
    /// The enum's value as the header numbers it.
    pub fn from_raw(raw: u64) -> Option<OperandIndex> {
        match raw {
            0 => Some(OperandIndex::Left),
            1 => Some(OperandIndex::Right),
            2 => Some(OperandIndex::Destination),
            _ => None,
        }
    }
}

/// `__tensor_ops_datatype`: an element type, as the helpers and the symbols name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    /// `float`.
    Float32,
    /// `half`.
    Float16,
    /// `bfloat`.
    BFloat16,
    /// A signed 2-bit integer.
    Int2,
    /// A signed 4-bit integer.
    Int4,
    /// `char`, a signed 8-bit integer.
    Int8,
    /// `short`.
    Int16,
    /// `int`.
    Int32,
    /// An unsigned 2-bit integer.
    UInt2,
    /// An unsigned 4-bit integer.
    UInt4,
    /// `uchar`.
    UInt8,
    /// `ushort`.
    UInt16,
    /// `uint`.
    UInt32,
    /// A 4-bit float: a sign, 2 exponent bits, and 1 significand bit.
    Fp4E2M1,
    /// An 8-bit float: a sign, 4 exponent bits, and 3 significand bits.
    Fp8E4M3,
    /// An 8-bit float: a sign, 5 exponent bits, and 2 significand bits.
    Fp8E5M2,
    /// An unsigned 8-bit exponent with no significand bits.
    Fp8UE8M0,
}

impl DataType {
    /// Every element type, for exhaustive tests.
    pub const ALL: [DataType; 17] = [
        DataType::Float32,
        DataType::Float16,
        DataType::BFloat16,
        DataType::Int2,
        DataType::Int4,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::UInt2,
        DataType::UInt4,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::Fp4E2M1,
        DataType::Fp8E4M3,
        DataType::Fp8E5M2,
        DataType::Fp8UE8M0,
    ];

    /// The enum's value in `MPPTensorOpsTypes.h`: a bit width in the low 16 bits, `0x10000000`
    /// for a float, `0x20000000` for a signed integer, and `0x80000000` / `0x40000000` for the
    /// alternate encodings (`bfloat`, `fp8_e5m2`, `fp8_ue8m0`).
    pub const fn bits(self) -> u32 {
        const FLOAT: u32 = 0x1000_0000;
        const SIGNED: u32 = 0x2000_0000;
        const ALT: u32 = 0x8000_0000;
        const ALT2: u32 = 0x4000_0000;
        match self {
            DataType::Float32 => FLOAT | 32,
            DataType::Float16 => FLOAT | 16,
            DataType::BFloat16 => ALT | FLOAT | 16,
            DataType::Int2 => SIGNED | 2,
            DataType::Int4 => SIGNED | 4,
            DataType::Int8 => SIGNED | 8,
            DataType::Int16 => SIGNED | 16,
            DataType::Int32 => SIGNED | 32,
            DataType::UInt2 => 2,
            DataType::UInt4 => 4,
            DataType::UInt8 => 8,
            DataType::UInt16 => 16,
            DataType::UInt32 => 32,
            DataType::Fp4E2M1 => FLOAT | 4,
            DataType::Fp8E4M3 => FLOAT | 8,
            DataType::Fp8E5M2 => ALT | FLOAT | 8,
            DataType::Fp8UE8M0 => ALT2 | FLOAT | 8,
        }
    }

    /// The element type whose enum value is `raw`.
    pub fn from_raw(raw: u64) -> Option<DataType> {
        DataType::ALL
            .into_iter()
            .find(|d| u64::from(d.bits()) == raw)
    }

    /// The element type a `run` symbol spells `token` (`f32`, `f16`, `b16`, `i8`, `ui8`, `i32`,
    /// `i4`, `ui4`, `i2`, `ui2`, `fp4e2m1`, `fp8e4m3`, `fp8e5m2`).
    pub fn from_token(token: &str) -> Option<DataType> {
        Some(match token {
            "f32" => DataType::Float32,
            "f16" => DataType::Float16,
            "b16" => DataType::BFloat16,
            "i8" => DataType::Int8,
            "ui8" => DataType::UInt8,
            "i32" => DataType::Int32,
            "i4" => DataType::Int4,
            "ui4" => DataType::UInt4,
            "i2" => DataType::Int2,
            "ui2" => DataType::UInt2,
            "fp4e2m1" => DataType::Fp4E2M1,
            "fp8e4m3" => DataType::Fp8E4M3,
            "fp8e5m2" => DataType::Fp8E5M2,
            _ => return None,
        })
    }

    /// Whether the type is an integer type.
    pub const fn is_integer(self) -> bool {
        self.bits() & 0x1000_0000 == 0
    }
}

/// The address space of a tensor operand (`dv`, `tg`, `th` in a `run` symbol).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorSpace {
    /// `dv`: device memory.
    Device,
    /// `tg`: threadgroup memory.
    Threadgroup,
    /// `th`: thread memory.
    Thread,
}

/// `__tensor_ops_tensor_descriptor_type`: how a tensor operand's descriptor object is formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DescriptorType {
    /// `tensor_handle`: a bound `MTLTensor`.
    Handle,
    /// A `tensor_handle` with an origin (`tensor_offset`).
    Offset,
    /// `tensor_inline`: a descriptor the shader built (`air.init_strided_private_tensor`).
    Inline,
    /// A raw thread pointer.
    None,
}

impl DescriptorType {
    /// The enum's value as the header numbers it.
    pub fn from_raw(raw: u64) -> Option<DescriptorType> {
        match raw {
            0 => Some(DescriptorType::Handle),
            1 => Some(DescriptorType::Offset),
            2 => Some(DescriptorType::Inline),
            3 => Some(DescriptorType::None),
            _ => None,
        }
    }
}

/// One operand of a `run` form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperandForm {
    /// A tensor in `space`, passed as its descriptor object and descriptor type.
    Tensor {
        /// The tensor's address space.
        space: TensorSpace,
        /// Its element type.
        elem: DataType,
    },
    /// A cooperative tensor, passed as its storage.
    Cooperative {
        /// Its element type.
        elem: DataType,
    },
}

impl OperandForm {
    /// The element type.
    pub const fn elem(self) -> DataType {
        match self {
            OperandForm::Tensor { elem, .. } | OperandForm::Cooperative { elem } => elem,
        }
    }

    /// Whether the operand is a cooperative tensor.
    pub const fn is_cooperative(self) -> bool {
        matches!(self, OperandForm::Cooperative { .. })
    }
}

/// A `run` form: its scope, its three operands, and whether it is the `_v2` spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunForm {
    /// `run_single_thread_*`: the `execution_thread` scope, which passes no thread count.
    pub single_thread: bool,
    /// The left operand's form.
    pub left: OperandForm,
    /// The right operand's form.
    pub right: OperandForm,
    /// The destination's form.
    pub destination: OperandForm,
    /// `_v2` (deployment target 27.0): each tensor operand's scale and static extents follow it.
    pub v2: bool,
}

impl RunForm {
    /// The number of arguments the header's declaration of this form takes.
    pub fn arity(&self) -> usize {
        let tensor = if self.v2 { 7 } else { 2 };
        let operand = |o: OperandForm, last: bool| match o {
            OperandForm::Cooperative { .. } => 1,
            // A `_v2` destination tensor passes its descriptor type and its two static extents,
            // no scale.
            OperandForm::Tensor { .. } if last && self.v2 => 4,
            OperandForm::Tensor { .. } => tensor,
        };
        1 + operand(self.left, false)
            + operand(self.right, false)
            + operand(self.destination, true)
            + usize::from(!self.single_thread)
    }

    /// The argument index of each operand's pointer: left, right, destination.
    pub fn operand_args(&self) -> [usize; 3] {
        let tensor = if self.v2 { 7 } else { 2 };
        let width = |o: OperandForm| if o.is_cooperative() { 1 } else { tensor };
        let left = 1;
        let right = left + width(self.left);
        let destination = right + width(self.right);
        [left, right, destination]
    }

    /// The argument indices of a `_v2` input tensor's scale type, its scale factors' two block
    /// sizes and its two static extents (`get_static_extent`), which follow its descriptor type
    /// in that order, for the tensor whose pointer is argument `at` ([`RunForm::operand_args`]).
    /// `None` for a form other than `_v2`.
    pub(crate) fn v2_tensor_args(&self, at: usize) -> Option<[usize; 5]> {
        self.v2.then_some([at + 2, at + 3, at + 4, at + 5, at + 6])
    }
}

/// One function of the `matmul2d` ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MatMul2dFn {
    /// `cooperative_tensor_data_size`: `i64 (operand, desc..., left, right, destination types,
    /// threads)`.
    TensorDataSize,
    /// `cooperative_tensor_num_elements`: `i16 (operand, desc..., storage, left, right types,
    /// threads)`.
    TensorNumElements,
    /// `cooperative_tensor_get_element_pointer`: `ptr (operand, desc..., storage, i16 index,
    /// left, right, destination types)`.
    TensorGetElementPointer,
    /// `cooperative_tensor_get_element_index`: `i16 (operand, desc..., storage, element, left,
    /// right, destination types)`.
    TensorGetElementIndex,
    /// `cooperative_tensor_get_coordinate`: `void (operand, desc..., storage, i16 index, ptr
    /// coordinates, coordinate type, threads, left, right, destination types)`.
    TensorGetCoordinate,
    /// `cooperative_tensor_init`: `void (operand, desc..., storage, left, right, destination
    /// types, threads)`.
    TensorInit,
    /// `cooperative_tensor_is_valid_element`: `i1 (operand, desc..., storage, i16 index, left,
    /// right, destination types, threads)`.
    TensorIsValidElement,
    /// A `run` form.
    Run(RunForm),
    /// A function of the family this contract does not model: the copies, conversions,
    /// reductions, loads and stores, the reduction-destination helpers, and the pre-26.2
    /// destination-only spellings. A translation refuses it; nothing is assumed of it.
    Unmodeled,
}

impl MatMul2dFn {
    /// The function a declaration named `symbol` in `section` is, or `None` when it is not a
    /// `matmul2d` ABI declaration (another name, or not in [`SECTION`]).
    pub fn of_declaration(symbol: &[u8], section: Option<&[u8]>) -> Option<MatMul2dFn> {
        if section != Some(SECTION) {
            return None;
        }
        let rest = symbol.strip_prefix(PREFIX)?;
        Some(std::str::from_utf8(rest).map_or(MatMul2dFn::Unmodeled, MatMul2dFn::parse))
    }

    /// The function the symbol's name after [`PREFIX`] spells.
    pub fn parse(rest: &str) -> MatMul2dFn {
        match rest {
            "cooperative_tensor_data_size" => MatMul2dFn::TensorDataSize,
            "cooperative_tensor_num_elements" => MatMul2dFn::TensorNumElements,
            "cooperative_tensor_get_element_pointer" => MatMul2dFn::TensorGetElementPointer,
            "cooperative_tensor_get_element_index" => MatMul2dFn::TensorGetElementIndex,
            "cooperative_tensor_get_coordinate" => MatMul2dFn::TensorGetCoordinate,
            "cooperative_tensor_init" => MatMul2dFn::TensorInit,
            "cooperative_tensor_is_valid_element" => MatMul2dFn::TensorIsValidElement,
            _ => rest
                .strip_prefix("run_")
                .and_then(parse_run)
                .map_or(MatMul2dFn::Unmodeled, MatMul2dFn::Run),
        }
    }

    /// The number of arguments the header's declaration takes, `None` for
    /// [`MatMul2dFn::Unmodeled`].
    pub fn arity(&self) -> Option<usize> {
        Some(match self {
            MatMul2dFn::TensorDataSize | MatMul2dFn::TensorNumElements => 12,
            MatMul2dFn::TensorGetElementPointer
            | MatMul2dFn::TensorGetElementIndex
            | MatMul2dFn::TensorInit => 13,
            MatMul2dFn::TensorIsValidElement => 14,
            MatMul2dFn::TensorGetCoordinate => 16,
            MatMul2dFn::Run(run) => run.arity(),
            MatMul2dFn::Unmodeled => return None,
        })
    }

    /// Whether the form takes the descriptor by value, as its seven fields at arguments 1 to 7
    /// (the cooperative-tensor helpers), rather than by reference at argument 0 (`run`).
    pub fn descriptor_by_value(&self) -> bool {
        matches!(
            self,
            MatMul2dFn::TensorDataSize
                | MatMul2dFn::TensorNumElements
                | MatMul2dFn::TensorGetElementPointer
                | MatMul2dFn::TensorGetElementIndex
                | MatMul2dFn::TensorGetCoordinate
                | MatMul2dFn::TensorInit
                | MatMul2dFn::TensorIsValidElement
        )
    }
}

/// `run_<...>` after `run_`.
fn parse_run(rest: &str) -> Option<RunForm> {
    let (single_thread, cooperative, rest) = if let Some(r) = rest.strip_prefix("single_thread_") {
        (true, false, r)
    } else if let Some(r) = rest.strip_prefix("cooperative_") {
        (false, true, r)
    } else {
        (false, false, rest)
    };
    let (v2, rest) = match rest.strip_suffix("_v2") {
        Some(r) => (true, r),
        None => (false, rest),
    };
    let mut tokens = rest.split('_').peekable();
    let mut operands = [None; 3];
    for slot in &mut operands {
        let first = tokens.next()?;
        let space = match first {
            "dv" => Some(TensorSpace::Device),
            "tg" => Some(TensorSpace::Threadgroup),
            "th" => Some(TensorSpace::Thread),
            _ => None,
        };
        *slot = Some(match space {
            Some(space) => OperandForm::Tensor {
                space,
                elem: DataType::from_token(tokens.next()?)?,
            },
            None => OperandForm::Cooperative {
                elem: DataType::from_token(first)?,
            },
        });
    }
    if tokens.next().is_some() {
        return None;
    }
    let [left, right, destination] = operands;
    let (left, right, destination) = (left?, right?, destination?);
    // `cooperative_` names exactly the forms with a cooperative operand, which only a simdgroup
    // scope has.
    let any_cooperative =
        left.is_cooperative() || right.is_cooperative() || destination.is_cooperative();
    if any_cooperative != cooperative || (single_thread && any_cooperative) {
        return None;
    }
    Some(RunForm {
        single_thread,
        left,
        right,
        destination,
        v2,
    })
}

/// A descriptor's inner dimension (`k`): a constant, or `dynamic_extent` (`-1` as an `int`) or
/// `dynamic_length_v<int>` (`INT_MAX`), which read it from the operands' extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Inner {
    /// A constant inner dimension.
    Static(u32),
    /// `dynamic_extent` or `dynamic_length_v<int>`: read from the operands' extents.
    Dynamic,
}

/// `matmul2d_descriptor::mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mode {
    /// `multiply`: the destination is the product.
    Multiply,
    /// `multiply_accumulate`: the product is added to the destination.
    MultiplyAccumulate,
}

/// `mpp::tensor_ops::matmul2d_descriptor`: `{ int m, n, k; bool transpose_left,
/// transpose_right, relaxed_precision; mode matmul_mode; }`, 20 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Descriptor {
    /// `m`: the destination's rows, the left operand's.
    pub m: u32,
    /// `n`: the destination's columns, the right operand's.
    pub n: u32,
    /// `k`: the inner dimension, the left operand's columns and the right operand's rows.
    pub k: Inner,
    /// `transpose_left`: the left operand is read transposed.
    pub transpose_left: bool,
    /// `transpose_right`: the right operand is read transposed.
    pub transpose_right: bool,
    /// `relaxed_precision`: the implementation may compute at a lower precision.
    pub relaxed_precision: bool,
    /// `matmul_mode`: whether the product replaces or accumulates into the destination.
    pub mode: Mode,
}

/// The descriptor's size in bytes.
pub const DESCRIPTOR_BYTES: usize = 20;

impl Descriptor {
    /// The descriptor of seven field values, each zero-extended from its width (`i32`, `i32`,
    /// `i32`, `i8`, `i8`, `i8`, `i32`): `None` unless `m` and `n` are positive, `k` is positive
    /// or dynamic, each flag is 0 or 1, and the mode is 0 or 1.
    pub fn from_fields(fields: [u64; 7]) -> Option<Descriptor> {
        let [m, n, k, tl, tr, relaxed, mode] = fields;
        let positive = |v: u64| {
            u32::try_from(v)
                .ok()
                .filter(|&v| v > 0 && v <= i32::MAX as u32)
        };
        let flag = |v: u64| match v {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        };
        let k = match k {
            0xFFFF_FFFF | 0x7FFF_FFFF => Inner::Dynamic,
            k => Inner::Static(positive(k)?),
        };
        Some(Descriptor {
            m: positive(m)?,
            n: positive(n)?,
            k,
            transpose_left: flag(tl)?,
            transpose_right: flag(tr)?,
            relaxed_precision: flag(relaxed)?,
            mode: match mode {
                0 => Mode::Multiply,
                1 => Mode::MultiplyAccumulate,
                _ => return None,
            },
        })
    }

    /// The descriptor stored as `bytes` (little-endian, [`DESCRIPTOR_BYTES`] long; the padding
    /// byte at offset 15 is ignored).
    #[cfg(test)]
    pub fn from_bytes(bytes: &[u8]) -> Option<Descriptor> {
        let word = |at: usize| -> Option<u64> {
            let b: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
            Some(u64::from(u32::from_le_bytes(b)))
        };
        let byte = |at: usize| bytes.get(at).map(|&b| u64::from(b));
        if bytes.len() != DESCRIPTOR_BYTES {
            return None;
        }
        Descriptor::from_fields([
            word(0)?,
            word(4)?,
            word(8)?,
            byte(12)?,
            byte(13)?,
            byte(14)?,
            word(16)?,
        ])
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let k = match self.k {
            Inner::Static(k) => k.to_string(),
            Inner::Dynamic => "dynamic".to_string(),
        };
        write!(f, "{}x{}x{k}", self.m, self.n)?;
        if self.transpose_left {
            f.write_str(" transpose_left")?;
        }
        if self.transpose_right {
            f.write_str(" transpose_right")?;
        }
        if self.relaxed_precision {
            f.write_str(" relaxed_precision")?;
        }
        if self.mode == Mode::MultiplyAccumulate {
            f.write_str(" multiply_accumulate")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
