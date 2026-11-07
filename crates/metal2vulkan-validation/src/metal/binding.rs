//! Whether a module the oracle compiles binds as the row the checker proved. The oracle binds by
//! the checked row's AIR-layer interface but runs the module it compiles, which for a public
//! fixture is this host's compile of the fixture's source, made apart from the committed row. The
//! two must agree on every fact the oracle binds by.

use crate::check::interface::{AirInterface, Member, Param};
use metal2vulkan_air::abi::{Access, ParamRole};
use std::mem::discriminant;

/// The first fact the oracle binds by on which `compiled`, the interface of a module it
/// compiles, differs from `checked`, the interface of the row the checker proved, or `None`.
///
/// Compared: the stage; each parameter's LLVM index and presence in the pipeline for the case's
/// constants; and for a parameter present in either, its resource class, its argument-table
/// slots, a texture's declared access (it sets the texture's usage), and an argument buffer's
/// members (offset, argument id, class, a texture's declared access). Not compared, because the
/// oracle binds by none of them: the rest of a role (a buffer's declared access, address space,
/// size, and layout; an imageblock's layouts; which builtin a value is), argument names and type
/// information, outputs, what the code does through its arguments, and the slots of a parameter
/// the pipeline leaves out for the case's constants (Metal binds no such argument). A
/// hand-written fixture and this host's compile of its source differ in exactly those:
/// `kernel_vector_function_constant` declares its output `air.write` with no name or type size
/// where the compiled module declares it `air.read_write`, named, and sized;
/// `fragment_custom_imageblock` names no master layout for its imageblock data; and
/// `kernel_absent_texture_reads_zero` computes the slot of a texture its case's constants leave
/// out as 0 where the compiled module declares 1.
pub(super) fn binding_difference(
    checked: &AirInterface,
    compiled: &AirInterface,
) -> Option<String> {
    if checked.stage != compiled.stage {
        return Some(format!(
            "its stage is {:?}, where the checked row's is {:?}",
            compiled.stage, checked.stage
        ));
    }
    if checked.params.len() != compiled.params.len() {
        return Some(format!(
            "it has {} parameters, where the checked row has {}",
            compiled.params.len(),
            checked.params.len()
        ));
    }
    checked
        .params
        .iter()
        .zip(&compiled.params)
        .find_map(|(ours, theirs)| param_difference(checked, ours, compiled, theirs))
}

fn param_difference(
    checked: &AirInterface,
    ours: &Param,
    compiled: &AirInterface,
    theirs: &Param,
) -> Option<String> {
    let present = ours.live || theirs.live;
    let alike = ours.binding.param == theirs.binding.param
        && ours.live == theirs.live
        && (!present
            || (same_class(&ours.binding.role, &theirs.binding.role)
                && ours.slots() == theirs.slots()));
    if !alike {
        return Some(format!(
            "parameter {} is {:?} (in the pipeline: {}), where the checked row's parameter {} is \
             {:?} (in the pipeline: {})",
            theirs.binding.param,
            theirs.binding.role,
            theirs.live,
            ours.binding.param,
            ours.binding.role,
            ours.live
        ));
    }
    if !present {
        return None;
    }
    let (ours_members, theirs_members) = (checked.members(ours), compiled.members(theirs));
    let member = |a: &Member, b: &Member| {
        a.offset == b.offset && a.id == b.id && same_class(&a.role, &b.role)
    };
    let alike = ours_members.len() == theirs_members.len()
        && ours_members
            .iter()
            .zip(&theirs_members)
            .all(|(a, b)| member(a, b));
    (!alike).then(|| {
        format!(
            "the members of argument buffer parameter {} are {theirs_members:?}, where the \
             checked row's are {ours_members:?}",
            ours.binding.param
        )
    })
}

/// Whether two roles are of one resource class and, for a texture, declare one access.
fn same_class(checked: &ParamRole, compiled: &ParamRole) -> bool {
    discriminant(checked) == discriminant(compiled)
        && texture_access(checked) == texture_access(compiled)
}

fn texture_access(role: &ParamRole) -> Option<Access> {
    match role {
        ParamRole::Texture(texture) => Some(texture.access),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::interface::tests::param;
    use metal2vulkan_air::abi::{
        AbiInt, BufferArg, ImageblockArg, ImageblockKind, Slot, Stage, StructRef, TextureArg,
    };
    use metal2vulkan_air::ir::AddrSpace;

    fn slot(index: u32) -> Slot {
        Slot {
            index: AbiInt::Const(index),
            count: AbiInt::Const(1),
        }
    }

    fn buffer(index: u32, access: Access) -> ParamRole {
        ParamRole::Buffer(BufferArg {
            location: slot(index),
            access,
            space: AddrSpace::Device,
            buffer_size: None,
            layout: None,
            raster_order_group: None,
        })
    }

    fn texture(index: u32, access: Access) -> ParamRole {
        ParamRole::Texture(TextureArg {
            location: slot(index),
            access,
            raster_order_group: None,
        })
    }

    fn kernel(params: Vec<Param>) -> AirInterface {
        AirInterface::synthetic(
            Stage::Kernel {
                max_total_threads: None,
                work_group_size: None,
            },
            params,
            vec![],
            vec![],
        )
    }

    fn differ(checked: Vec<Param>, compiled: Vec<Param>) -> bool {
        binding_difference(&kernel(checked), &kernel(compiled)).is_some()
    }

    /// A slot, a class, a texture's access, or presence in the pipeline binds otherwise.
    #[test]
    fn what_the_oracle_binds_by_must_agree() {
        let base = || {
            vec![
                param(0, buffer(0, Access::Read)),
                param(1, texture(1, Access::Read)),
            ]
        };
        assert!(!differ(base(), base()));
        let mut moved = base();
        moved[0] = param(0, buffer(2, Access::Read));
        assert!(differ(base(), moved));
        let mut written = base();
        written[1] = param(1, texture(1, Access::Write));
        assert!(differ(base(), written));
        let mut sampler = base();
        sampler[1] = param(1, ParamRole::Sampler { location: slot(1) });
        assert!(differ(base(), sampler));
        let mut absent = base();
        absent[1].live = false;
        assert!(differ(base(), absent));
        assert!(differ(base(), base()[..1].to_vec()));
        let fragment = AirInterface::synthetic(
            Stage::Fragment {
                early_fragment_tests: false,
            },
            base(),
            vec![],
            vec![],
        );
        assert!(binding_difference(&kernel(base()), &fragment).is_some());
    }

    /// A buffer's declared access, an imageblock's master layout, and the slot of an argument
    /// the pipeline leaves out bind nothing.
    #[test]
    fn what_the_oracle_does_not_bind_by_may_differ() {
        assert!(!differ(
            vec![param(0, buffer(0, Access::Write))],
            vec![param(0, buffer(0, Access::ReadWrite))],
        ));
        let imageblock = |master| {
            param(
                0,
                ParamRole::Imageblock(ImageblockArg {
                    kind: ImageblockKind::Data,
                    data_size: Some(8),
                    layout: Some(StructRef(0)),
                    master,
                    alias_implicit: false,
                }),
            )
        };
        assert!(!differ(
            vec![imageblock(Some(StructRef(0)))],
            vec![imageblock(None)],
        ));
        let absent = |index| Param {
            live: false,
            ..param(0, texture(index, Access::Write))
        };
        assert!(!differ(vec![absent(0)], vec![absent(1)]));
    }
}
