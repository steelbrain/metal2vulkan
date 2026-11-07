//! The cases of the probe `linked_function_signatures.m` (the conformance crate's `probes/`): a
//! visible function called through a declaration of another type than its definition's. The owned
//! fixture `engine/linked_signature_caller` has a kernel `k_<case>` for each, which calls its
//! function once on `in = {2, 3, 4, 5}` (buffer 1) and stores the bits of the result to word 0 of
//! `out` (buffer 0), then 0x5e47 to word 1 (`k_void_for_value` stores 7 to word 0 after its call).
//! The functions are those of `engine/linked_signature_functions`, and the stitched
//! [`STITCHED_PAIR`].
//!
//! Each case's [`Answer`] is word 0: the one Metal stored on an M3 Max under macOS 27.0.1 (the
//! probe compiling these fixtures), where the consumer's Radeon Pro 5300M and UHD Graphics 630
//! under macOS 26.7.1 stored the same where they ran the shape; the translator's own where Metal's
//! read is undefined (docs/API.md, section 10, "Linked functions"); or a refusal where the
//! translator has no adapter for the shape, which Metal makes.

use crate::stitched::{Graph, Node, Operand};

/// The stitched function the case `stitched_extra` links: `stitched_pair(a, b) = stitch_add(a,
/// b)`, of two inputs, which the caller declares with three.
pub const STITCHED_PAIR: Graph<'static> = Graph {
    name: "stitched_pair",
    inputs: 2,
    nodes: &[Node {
        function: "stitch_add",
        op: "fadd",
        args: [Operand::Input(0), Operand::Input(1)],
    }],
};

/// Word 0 a case's kernel stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    /// The word Metal stored.
    Metal(u32),
    /// Metal's read is undefined (the M3 Max stored nothing to word 0): the word the translator
    /// stores.
    Undefined(u32),
    /// Metal makes the pipeline; the translator refuses it (`Unsupported`, `Feature::Linking`).
    Refused,
}

/// One case: its kernel's name less `k_`, the function it links (of the functions fixture, or
/// [`STITCHED_PAIR`]'s), and word 0.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub name: &'static str,
    pub function: &'static str,
    pub stitched: bool,
    pub answer: Answer,
}

const fn case(name: &'static str, answer: Answer) -> Case {
    Case {
        name,
        function: name,
        stitched: false,
        answer,
    }
}

/// Every case of the probe but `double_param` (Metal compiles no `double`).
pub const CASES: [Case; 28] = [
    Case {
        name: "stitched_extra",
        function: "stitched_pair",
        stitched: true,
        answer: Answer::Metal(0x40a0_0000),
    },
    case("extra", Answer::Metal(0x40a0_0000)),
    case("extra_two", Answer::Metal(0x4000_0000)),
    case("extra_pointer", Answer::Metal(0x40a0_0000)),
    // `a + b + c` with `c` read as zero.
    case("missing", Answer::Metal(0x40a0_0000)),
    case("missing_unused", Answer::Metal(0x40a0_0000)),
    case("missing_pointer_unused", Answer::Metal(0x40a0_0000)),
    // 2 + the bits of 3.0 as an `int`, 1077936128.
    case("int_param", Answer::Metal(0x4e80_8000)),
    // 2 + the halves of 3.0's bits, 0.0 and 2.125.
    case("half2_param", Answer::Metal(0x4084_0000)),
    // The bits of 2.0 and 3.0 as `uint`s, summed.
    case("uint2_param", Answer::Metal(0x4f00_4000)),
    // 3.0's low 16 bits are the `half` 0.0: 2 + 0.
    case("half_param", Answer::Undefined(0x4000_0000)),
    // The `half` 3.0's bits, zero above.
    case("uint_for_half", Answer::Undefined(0x0000_4200)),
    // 2 + 1077936128, the bits of 3.0 zero above, rounds to 1077936128.
    case("long_param", Answer::Undefined(0x4e80_8000)),
    // `a` is the vector's low lane, 2.0, and `b` the next argument, 4.0: 2 + 10 * 4.
    case("float2_for_two", Answer::Undefined(0x4228_0000)),
    // 2.0's low bit is 0, false: -3.
    case("bool_param", Answer::Undefined(0xc040_0000)),
    // 5.0's bits as an `int`.
    case("int_return", Answer::Metal(0x40a0_0000)),
    // The `half` 5.0's bits, zero above.
    case("float_for_half_return", Answer::Undefined(0x0000_4500)),
    // 5.0's low 16 bits are the `half` 0.0.
    case("half_for_float_return", Answer::Undefined(0)),
    case("value_for_void", Answer::Undefined(0)),
    case("void_for_value", Answer::Metal(7)),
    case("value_for_pointer", Answer::Refused),
    case("pointer_for_value", Answer::Refused),
    case("constant_for_device", Answer::Refused),
    case("device_for_constant", Answer::Refused),
    // A struct of two floats is its two floats to the compiler.
    case("pair_for_floats", Answer::Metal(0x40a0_0000)),
    case("uints_for_floats", Answer::Metal(0x4f00_4000)),
    // MPS's `write_fn([8 x i8], float, float)` shape: 2 + the first byte, 1.
    case("bytes_extra", Answer::Metal(0x4040_0000)),
    case("uint_array_for_float_array", Answer::Refused),
];
