//! The depth-stencil state both executors set for a draw: the Metal oracle's
//! `MTLDepthStencilState`, stencil references, and front-facing winding, and the engine
//! candidate's Vulkan state of the same encoder state, derived once, so the two agree. A case's
//! authored state ([`crate::case::DepthStencilStateLiteral`]) is the state; without one, the
//! state is derived from the fragment function's AIR-layer interface.

use super::DepthCompare;
use crate::case::{
    CompareFunctionLiteral, DepthStencilStateLiteral, StencilFaceLiteral, StencilOperationLiteral,
    WindingLiteral,
};
use crate::check::interface::AirInterface;
use metal2vulkan::metal::state::{
    depth_stencil_state, DepthStencilDescriptor, DepthStencilState, StencilDescriptor,
};
use metal2vulkan::metal::{CompareFunction, StencilOperation, Winding};
use metal2vulkan::vk;
use metal2vulkan_air::abi::{DepthQualifier, OutputRole};

/// The render encoder's depth/stencil state for one draw: the `MTLDepthStencilDescriptor`, the
/// front and back stencil references, and the front-facing winding. Both executors set exactly
/// this; the Vulkan candidate maps the descriptor with the product's
/// `metal::state::depth_stencil_state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawDepthStencil {
    pub descriptor: DepthStencilDescriptor,
    pub front_reference: u32,
    pub back_reference: u32,
    pub winding: Winding,
}

impl Default for DrawDepthStencil {
    /// Metal's render encoder defaults: the default descriptor (no depth test, no stencil
    /// test), references 0, and clockwise front faces.
    fn default() -> DrawDepthStencil {
        DrawDepthStencil {
            descriptor: DepthStencilDescriptor::default(),
            front_reference: 0,
            back_reference: 0,
            winding: Winding::Clockwise,
        }
    }
}

impl DrawDepthStencil {
    /// The state of `test` under the encoder's other defaults: a depth test with its comparison
    /// that writes depth, and on both faces a stencil test that always passes and replaces every
    /// bit with the reference (the fragment function's `[[stencil]]` output).
    pub fn derived(test: DepthStencilTest) -> DrawDepthStencil {
        let mut descriptor = DepthStencilDescriptor::default();
        if let Some(compare) = test.depth {
            descriptor.depth_write_enabled = true;
            descriptor.depth_compare_function = match compare {
                DepthCompare::Always => CompareFunction::Always,
                DepthCompare::Less => CompareFunction::Less,
                DepthCompare::Greater => CompareFunction::Greater,
            };
        }
        if test.stencil {
            let mut face = StencilDescriptor::default();
            face.depth_stencil_pass_operation = StencilOperation::Replace;
            descriptor.front_face_stencil = face;
            descriptor.back_face_stencil = face;
        }
        DrawDepthStencil {
            descriptor,
            ..DrawDepthStencil::default()
        }
    }

    /// The authored state `state`.
    pub fn authored(state: &DepthStencilStateLiteral) -> DrawDepthStencil {
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = compare(state.depth_compare);
        descriptor.depth_write_enabled = state.depth_write;
        descriptor.front_face_stencil = face(&state.front);
        descriptor.back_face_stencil = face(&state.back);
        DrawDepthStencil {
            descriptor,
            front_reference: state.front_reference,
            back_reference: state.back_reference,
            winding: match state.front_facing_winding {
                WindingLiteral::Clockwise => Winding::Clockwise,
                WindingLiteral::CounterClockwise => Winding::CounterClockwise,
            },
        }
    }

    /// The state of a fragment case's draw: its authored state when it has one, else the state
    /// its fragment function's live outputs imply.
    pub fn of_case(
        authored: Option<&DepthStencilStateLiteral>,
        interface: &AirInterface,
    ) -> DrawDepthStencil {
        match authored {
            Some(state) => DrawDepthStencil::authored(state),
            None => DrawDepthStencil::derived(depth_stencil_test(interface)),
        }
    }

    /// The Vulkan depth/stencil state the product maps the descriptor to
    /// (`metal::state::depth_stencil_state`), the one mapping both executors' tests read.
    pub fn vulkan(&self) -> DepthStencilState {
        depth_stencil_state(&self.descriptor)
    }

    /// Whether the draw compares or writes depth (Metal's depth test is the identity under
    /// `Always` without writes; Vulkan's is disabled then).
    pub fn tests_depth(&self) -> bool {
        self.vulkan().depth_test_enable
    }

    /// Whether either face's stencil test can fail or change the stencil value (Metal's stencil
    /// test is the identity otherwise; Vulkan's is disabled then).
    pub fn tests_stencil(&self) -> bool {
        self.vulkan().stencil_test_enable
    }

    /// The first face, with its compare function, whose stencil test a fragment function's
    /// `[[stencil]]` output decides under this state: the output is the test's reference on both
    /// faces, so a face whose test compares by a function other than `Always` and `Never` takes its
    /// answer from it. `None` where the stencil test is off ([`Self::tests_stencil`]) or both faces
    /// compare by `Always` or `Never`.
    ///
    /// It is the runtime's rule for a device without `VK_EXT_shader_stencil_export`, which draws
    /// the output by bits, every draw but its first under a reference of its own
    /// (`metal2vulkan_runtime`'s `depth_stencil::BitDraws::of`, read on the same Vulkan state,
    /// `metal::state::depth_stencil_state`): it refuses a draw whose test the output decides
    /// (`Gap::Draw` of `DrawKind::StencilOutput`). It refuses one more draw: where the state
    /// replaces a stencil bit, one inside a `Counting` visibility result's segment, whose count
    /// each of its draws would add to. That clause has nothing to read: the case schema expresses
    /// no visibility result mode (no field of `AuthoredCase`, its `Draw` or its
    /// `DepthStencilResource` names one, and each refuses an unknown field), and neither executor
    /// sets one or gives its pass a visibility result buffer, so no case's draw is in a segment.
    pub fn stencil_reference_face(&self) -> Option<(&'static str, CompareFunction)> {
        let state = self.vulkan();
        if !state.stencil_test_enable {
            return None;
        }
        let fixed = |op: vk::CompareOp| matches!(op, vk::CompareOp::ALWAYS | vk::CompareOp::NEVER);
        let faces = [
            ("front", state.front, self.descriptor.front_face_stencil),
            ("back", state.back, self.descriptor.back_face_stencil),
        ];
        faces
            .into_iter()
            .find(|(_, face, _)| !fixed(face.compare_op))
            .map(|(name, _, metal)| (name, metal.stencil_compare_function))
    }
}

fn compare(f: CompareFunctionLiteral) -> CompareFunction {
    match f {
        CompareFunctionLiteral::Never => CompareFunction::Never,
        CompareFunctionLiteral::Less => CompareFunction::Less,
        CompareFunctionLiteral::Equal => CompareFunction::Equal,
        CompareFunctionLiteral::LessEqual => CompareFunction::LessEqual,
        CompareFunctionLiteral::Greater => CompareFunction::Greater,
        CompareFunctionLiteral::NotEqual => CompareFunction::NotEqual,
        CompareFunctionLiteral::GreaterEqual => CompareFunction::GreaterEqual,
        CompareFunctionLiteral::Always => CompareFunction::Always,
    }
}

fn operation(op: StencilOperationLiteral) -> StencilOperation {
    match op {
        StencilOperationLiteral::Keep => StencilOperation::Keep,
        StencilOperationLiteral::Zero => StencilOperation::Zero,
        StencilOperationLiteral::Replace => StencilOperation::Replace,
        StencilOperationLiteral::IncrementClamp => StencilOperation::IncrementClamp,
        StencilOperationLiteral::DecrementClamp => StencilOperation::DecrementClamp,
        StencilOperationLiteral::Invert => StencilOperation::Invert,
        StencilOperationLiteral::IncrementWrap => StencilOperation::IncrementWrap,
        StencilOperationLiteral::DecrementWrap => StencilOperation::DecrementWrap,
    }
}

fn face(s: &StencilFaceLiteral) -> StencilDescriptor {
    let mut out = StencilDescriptor::default();
    out.stencil_compare_function = compare(s.compare);
    out.stencil_failure_operation = operation(s.stencil_failure);
    out.depth_failure_operation = operation(s.depth_failure);
    out.depth_stencil_pass_operation = operation(s.depth_stencil_pass);
    out.read_mask = s.read_mask;
    out.write_mask = s.write_mask;
    out
}

/// The depth-stencil state both executors set for a draw: the Metal oracle's
/// `MTLDepthStencilState` and the engine candidate's Vulkan depth-stencil state. A fragment
/// function that writes `[[depth(q)]]` tests with the comparison `q` names (`any` always passes)
/// and writes depth; one that writes `[[stencil]]` always passes the stencil test and replaces the
/// stencil value with its output, all bits read and written. Otherwise neither test runs, so an
/// attached aspect keeps its contents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilTest {
    pub depth: Option<DepthCompare>,
    pub stencil: bool,
}

/// The [`DepthStencilTest`] of the fragment function whose AIR-layer interface is `interface`:
/// read from its live outputs' AIR ABI roles (`OutputRole::Depth(qualifier)`,
/// `OutputRole::Stencil`), so an output the case's function constants gate off (which the
/// pipeline drops) sets no test. The one derivation of the state; neither executor derives it
/// another way.
pub fn depth_stencil_test(interface: &AirInterface) -> DepthStencilTest {
    let mut test = DepthStencilTest::default();
    for output in interface.outputs.iter().filter(|output| output.live) {
        match &output.binding.role {
            OutputRole::Depth(qualifier) => {
                test.depth = Some(match qualifier {
                    DepthQualifier::Any => DepthCompare::Always,
                    DepthQualifier::Less => DepthCompare::Less,
                    DepthQualifier::Greater => DepthCompare::Greater,
                })
            }
            OutputRole::Stencil => test.stencil = true,
            _ => {}
        }
    }
    test
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The fragment module whose function returns a depth output with `qualifier` (none when
    /// `None`) and a stencil output when `stencil`; the depth output is gated by function constant
    /// 0 when `gated`.
    pub(crate) fn depth_stencil_module(
        qualifier: Option<&str>,
        stencil: bool,
        gated: bool,
    ) -> String {
        let mut outputs = Vec::new();
        let mut returned = Vec::new();
        if let Some(qualifier) = qualifier {
            let predicate = if gated {
                r#"!"air.function_constant", !9, "#
            } else {
                ""
            };
            outputs.push(format!(
                r#"!{{{predicate}!"air.depth", !"air.depth_qualifier", !"air.{qualifier}", !"air.arg_type_name", !"float"}}"#
            ));
            returned.push(("float", "5.000000e-01"));
        }
        if stencil {
            outputs.push(r#"!{!"air.stencil", !"air.arg_type_name", !"uint"}"#.into());
            returned.push(("i32", "7"));
        }
        let ty = format!(
            "<{{ {} }}>",
            returned
                .iter()
                .map(|(ty, _)| *ty)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut body = String::new();
        let mut previous = "undef".to_string();
        for (index, (field, value)) in returned.iter().enumerate() {
            body.push_str(&format!(
                "  %v{index} = insertvalue {ty} {previous}, {field} {value}, {index}\n"
            ));
            previous = format!("%v{index}");
        }
        let nodes = (0..outputs.len())
            .map(|index| format!("!{}", 3 + index))
            .collect::<Vec<_>>()
            .join(", ");
        let defined = outputs
            .iter()
            .enumerate()
            .map(|(index, output)| format!("!{} = {output}\n", 3 + index))
            .collect::<String>();
        format!(
            r#"target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"
@flag.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@pred = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x {{ i32, ptr, ptr }}] [{{ i32, ptr, ptr }} {{ i32 65535, ptr @init, ptr null }}]
define internal void @init() section "air.static_init" {{
  %v = load i8, ptr addrspace(2) @flag.MTL_FC_INIT_0_b, align 1
  %p = call i8 @air.normalize_function_constant_predicate.i8(i8 %v)
  store i8 %p, ptr addrspace(2) @pred, align 1
  ret void
}}
declare i8 @air.normalize_function_constant_predicate.i8(i8)
define {ty} @frag() {{
{body}  ret {ty} {previous}
}}
!air.fragment = !{{!0}}
!air.function_constants = !{{!8}}
!0 = !{{ptr @frag, !1, !2}}
!1 = !{{{nodes}}}
!2 = !{{}}
{defined}!8 = !{{ptr addrspace(2) @flag.MTL_FC_INIT_0_b, !"bool", !"flag", i32 0}}
!9 = !{{ptr addrspace(2) @pred, !"bool", !"flag"}}
"#
        )
    }

    /// The one depth-stencil state both executors set: each depth qualifier's comparison, the
    /// stencil test, and no test for an output the case's function constants gate off (the pipeline
    /// drops it), though its declaration remains.
    #[test]
    fn the_depth_stencil_state_follows_the_live_outputs() {
        use crate::check::interface::tests::of_text;
        use metal2vulkan_air::canon::ConstValue;
        let flag = |value: u8| {
            BTreeMap::from([(
                0,
                ConstValue {
                    ty: metal2vulkan::metal::DataType::Bool,
                    bytes: Box::new([value]),
                },
            )])
        };
        let test = |qualifier, stencil, gated, constants: &BTreeMap<u32, ConstValue>| {
            let ll = depth_stencil_module(qualifier, stencil, gated);
            match of_text(&ll, "frag", constants) {
                Ok(interface) => Some(depth_stencil_test(&interface)),
                Err(error) if error.contains("llvm-as is unavailable") => {
                    eprintln!("SKIPPED: {error}");
                    None
                }
                Err(error) => panic!("{error}\n{ll}"),
            }
        };
        let none = BTreeMap::new();
        let Some(less) = test(Some("less"), false, false, &none) else {
            return;
        };
        let state = |depth, stencil| Some(DepthStencilTest { depth, stencil });
        assert_eq!(Some(less), state(Some(DepthCompare::Less), false));
        assert_eq!(
            test(Some("greater"), false, false, &none),
            state(Some(DepthCompare::Greater), false)
        );
        assert_eq!(
            test(Some("any"), true, false, &none),
            state(Some(DepthCompare::Always), true)
        );
        assert_eq!(test(None, true, false, &none), state(None, true));
        // Gated: live when the constant is true, and no test when it is false or unsupplied, while
        // the stencil output beside it still sets its test.
        assert_eq!(
            test(Some("less"), true, true, &flag(1)),
            state(Some(DepthCompare::Less), true)
        );
        assert_eq!(test(Some("less"), true, true, &flag(0)), state(None, true));
        assert_eq!(test(Some("less"), false, true, &none), state(None, false));
    }
}
