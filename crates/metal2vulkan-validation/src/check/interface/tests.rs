//! The AIR-layer interface facts over public fixtures and synthetic text modules.

use super::*;
use crate::check::loop_free::{assemble, DecodedModule};
use metal2vulkan::metal::DataType;
use metal2vulkan_air::abi::{ArgTypeInfo, ObjectArg};

/// The interface of `entry` in the text module `ll` for `constants`.
pub(crate) fn of_text(
    ll: &str,
    entry: &str,
    constants: &BTreeMap<u32, ConstValue>,
) -> Result<AirInterface, String> {
    let bitcode = assemble(ll)?;
    let decoded = DecodedModule::from_bitcode("test", &bitcode).map_err(|e| e.to_string())?;
    let abi = decoded.abi().map_err(str::to_string)?;
    AirInterface::new(decoded.module(), abi, entry, constants)
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/fixtures/public/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("public fixture")
}

/// A parameter with `role` and nothing else, present in the pipeline.
pub(crate) fn param(param: u32, role: ParamRole) -> Param {
    Param {
        binding: ParamBinding {
            param,
            role,
            unused: false,
            predicate: None,
            ty: ArgTypeInfo::default(),
            name: None,
        },
        live: true,
        table: false,
        value: None,
    }
}

/// A vertex entry whose only arguments are read-only device buffers at `slots`.
pub(crate) fn vertex_buffers_at(slots: &[u32]) -> AirInterface {
    use metal2vulkan_air::abi::BufferArg;
    let params = slots
        .iter()
        .enumerate()
        .map(|(position, &slot)| {
            param(
                position as u32,
                ParamRole::Buffer(BufferArg {
                    location: Slot {
                        index: AbiInt::Const(slot),
                        count: AbiInt::Const(1),
                    },
                    access: Access::Read,
                    space: AddrSpace::Device,
                    buffer_size: None,
                    layout: None,
                    raster_order_group: None,
                }),
            )
        })
        .collect();
    AirInterface::synthetic(
        Stage::Vertex { patch: None },
        params,
        Vec::new(),
        Vec::new(),
    )
}

/// An argument buffer at `slot` whose layout (index 0) holds `members` as `(offset, role)`; each
/// member's argument id is its role's location.
pub(crate) fn argument_buffer(slot: u32, members: &[(u32, ParamRole)]) -> AirInterface {
    use metal2vulkan::error::DiagName;
    use metal2vulkan_air::abi::{
        BufferArg, IndirectArg, Member as AbiMember, StructInfo as AbiStruct,
    };
    let role = ParamRole::ArgumentBuffer(BufferArg {
        location: Slot {
            index: AbiInt::Const(slot),
            count: AbiInt::Const(1),
        },
        access: Access::Read,
        space: AddrSpace::Constant,
        buffer_size: None,
        layout: Some(StructRef(0)),
        raster_order_group: None,
    });
    let layout = AbiStruct {
        members: members
            .iter()
            .enumerate()
            .map(|(ordinal, (offset, role))| AbiMember {
                offset: *offset,
                size: 8,
                array_len: 0,
                layout: None,
                kind: MemberKind::Indirect(IndirectArg {
                    ordinal: ordinal as u32,
                    role: role.clone(),
                    ty: ArgTypeInfo::default(),
                    name: None,
                }),
                render_target: None,
                raster_order_group: None,
                type_name: DiagName::new("t"),
                name: DiagName::new("m"),
                key: metal2vulkan_air::abi::UserKey(b"m".as_slice().into()),
            })
            .collect(),
    };
    AirInterface::synthetic(
        Stage::Kernel {
            max_total_threads: None,
            work_group_size: None,
        },
        vec![param(0, role)],
        Vec::new(),
        vec![layout],
    )
}

/// An object argument at `slot` (a pipeline state, command buffer, or table member).
pub(crate) fn object(slot: u32) -> ObjectArg {
    ObjectArg {
        location: Slot {
            index: AbiInt::Const(slot),
            count: AbiInt::Const(1),
        },
        access: None,
    }
}

#[test]
fn slots_and_declared_access_come_from_the_metadata_and_writes_from_the_ir() {
    // Both buffers are declared `air.read`; the kernel stores through the second.
    let interface = of_text(
        &fixture("kernel_copy_word_declared_read.ll"),
        "copy_word_declared_read",
        &BTreeMap::new(),
    )
    .unwrap();
    let buffers = interface
        .params_of(|role| match role {
            ParamRole::Buffer(buffer) => Some(buffer.access),
            _ => None,
        })
        .map(|(param, access)| (param.slots().unwrap(), access, param.required()))
        .collect::<Vec<_>>();
    assert_eq!(
        buffers,
        [((0, 1), Access::Read, true), ((1, 1), Access::Read, true)]
    );
    assert!(!interface.writes(Root::Param(0)));
    assert!(interface.writes(Root::Param(1)));
    assert!(interface.writes_through(1));
}

#[test]
fn predicates_and_computed_slots_resolve_for_the_case_constants() {
    // Textures 0 and 1 are gated by `provide_absent` and bound at slots the static initializer
    // computes; 2 and 3 are unconditional.
    let ll = fixture("kernel_absent_texture_reads_zero.ll");
    let live = |constants: &BTreeMap<u32, ConstValue>| {
        of_text(&ll, "absent_texture_reads_zero", constants)
            .unwrap()
            .params_of(texture)
            .map(|(param, _)| (param.binding.param, param.slots().unwrap().0, param.live))
            .collect::<Vec<_>>()
    };
    let flag = |value: u8| {
        BTreeMap::from([(
            0,
            ConstValue {
                ty: DataType::Bool,
                bytes: Box::new([value]),
            },
        )])
    };
    assert_eq!(
        live(&flag(1)),
        [(0, 0, true), (1, 0, true), (2, 2, true), (3, 3, true)]
    );
    assert_eq!(
        live(&flag(0)),
        [(0, 0, false), (1, 0, false), (2, 2, true), (3, 3, true)]
    );
    // An unsupplied constant's predicate is false.
    assert_eq!(
        live(&BTreeMap::new()),
        [(0, 0, false), (1, 0, false), (2, 2, true), (3, 3, true)]
    );
    // A value of the wrong type is the case's error.
    let wrong = BTreeMap::from([(
        0,
        ConstValue {
            ty: DataType::Float,
            bytes: 1f32.to_le_bytes().into(),
        },
    )]);
    let error = of_text(&ll, "absent_texture_reads_zero", &wrong).unwrap_err();
    assert!(error.contains("FunctionConstantType"), "{error}");
}

fn texture(role: &ParamRole) -> Option<&TextureArg> {
    match role {
        ParamRole::Texture(texture) => Some(texture),
        _ => None,
    }
}

use metal2vulkan_air::abi::TextureArg;

#[test]
fn a_texture_tables_element_uses_spell_its_shape() {
    // `array<texture2d<float, sample>, 2>`: a table; the kernel queries element 1.
    let interface = of_text(
        &fixture("kernel_texture_array_width.ll"),
        "texture_array_width",
        &BTreeMap::new(),
    )
    .unwrap();
    let (table, _) = interface.params_of(texture).next().unwrap();
    assert!(table.table);
    assert_eq!(table.slots(), Some((0, 2)));
    assert_eq!(interface.loaded_uses(0, 0).count(), 0);
    let uses = interface.loaded_uses(0, 8).collect::<Vec<_>>();
    assert_eq!(uses.len(), 1);
    assert_eq!(uses[0].texture_type(), Some(TextureType::D2));
    // A query spells no texel, so any format's class is accepted.
    assert!(uses[0].accepts_format(TextureFormat::R32Uint));
}

#[test]
fn member_ids_add_down_the_nesting_path_and_are_not_ordinals() {
    // `tex` is the third member (ordinal 2), but `bounds` spends ids 1 and 2 on its own members,
    // so Metal's argument encoder takes `tex` at id 3: its `air.location_index`.
    let ll = r#"
define void @nested(ptr addrspace(2) %args) {
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @nested, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"head", !"air.indirect_argument", !5, !"air.struct_type_info", !6, i32 4, i32 8, i32 0, !"Bounds", !"bounds", !"air.indirect_argument", i32 1, i32 16, i32 8, i32 0, !"texture2d<half, write>", !"tex", !"air.indirect_argument", !9}
!5 = !{i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"head"}
!6 = !{i32 0, i32 4, i32 0, !"ushort2", !"min", !"air.indirect_argument", !7, i32 4, i32 4, i32 0, !"ushort2", !"max", !"air.indirect_argument", !8}
!7 = !{i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"ushort2", !"air.arg_name", !"min"}
!8 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"ushort2", !"air.arg_name", !"max"}
!9 = !{i32 2, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"tex"}
"#;
    let interface = of_text(ll, "nested", &BTreeMap::new()).unwrap();
    let ids = interface
        .members(&interface.params[0])
        .into_iter()
        .map(|member| (member.offset, member.id))
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        [(0, Some(0)), (4, Some(1)), (8, Some(2)), (16, Some(3))]
    );
    let (_, tex) = interface.member_at(0, 16).unwrap();
    assert!(matches!(tex.role, ParamRole::Texture(_)));
}

#[test]
fn a_metal_array_member_binds_every_element_its_location_count_states() {
    // `array<texture2d<half, sample>, 2> tex` states array length 0, the whole array's 16 bytes,
    // and its two elements as the location count; `texture2d<half> c[2]` states length 2 and
    // 8 bytes per element. Both bind one argument per element.
    let ll = r#"
define void @arrays(ptr addrspace(2) %args) {
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @arrays, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 40, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 40, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{i32 0, i32 8, i32 0, !"uint", !"head", !"air.indirect_argument", !5, i32 8, i32 16, i32 0, !"array<texture2d<half, sample>, 2>", !"tex", !"air.indirect_argument", !6, i32 24, i32 8, i32 2, !"texture2d<half, sample>", !"c", !"air.indirect_argument", !7}
!5 = !{i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"head"}
!6 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 2, !"air.sample", !"air.arg_type_name", !"array<texture2d<half, sample>, 2>", !"air.arg_name", !"tex"}
!7 = !{i32 2, !"air.texture", !"air.location_index", i32 3, i32 2, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"c"}
"#;
    let interface = of_text(ll, "arrays", &BTreeMap::new()).unwrap();
    let ids = interface
        .members(&interface.params[0])
        .into_iter()
        .map(|member| (member.offset, member.id))
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        [
            (0, Some(0)),
            (8, Some(1)),
            (16, Some(2)),
            (24, Some(3)),
            (32, Some(4))
        ]
    );
}

#[test]
fn argument_buffer_members_keep_their_offsets_and_ids() {
    let interface = of_text(
        &fixture("kernel_device_pointer_chain_phi.ll"),
        "kernel_device_pointer_chain_phi",
        &BTreeMap::new(),
    )
    .unwrap();
    let (buffer, _) = interface
        .params_of(|role| match role {
            ParamRole::ArgumentBuffer(buffer) => Some(buffer),
            _ => None,
        })
        .next()
        .unwrap();
    let members = interface
        .members(buffer)
        .into_iter()
        .map(|member| {
            (
                member.offset,
                member.id,
                matches!(member.role, ParamRole::Buffer(_)),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(members, [(0, Some(0), true), (8, Some(1), false)]);
    assert!(interface.member_at(0, 0).is_some());
    assert!(interface.member_at(0, 4).is_none());
    assert!(interface.member_at(1, 0).is_none());
}

#[test]
fn texel_classes_admit_the_formats_of_their_component_class() {
    let usage = |texel| TextureUse {
        dim: ImageDimKey::D2,
        array: false,
        ms: false,
        depth: false,
        texel,
    };
    // A half texture is float-classed: every float format backs it, no integer one does.
    let float = usage(Some(TexelClass::Float));
    assert!(float.accepts_format(TextureFormat::R16Float));
    assert!(float.accepts_format(TextureFormat::Rg16Float));
    assert!(float.accepts_format(TextureFormat::Depth32Float));
    assert!(!float.accepts_format(TextureFormat::R32Uint));
    let unsigned = usage(Some(TexelClass::Uint));
    assert!(unsigned.accepts_format(TextureFormat::Rgba8Uint));
    assert!(!unsigned.accepts_format(TextureFormat::R16Float));
    assert!(!unsigned.accepts_format(TextureFormat::R32Sint));
    // An operation without a sign token admits either integer class, never a float one.
    let integer = usage(Some(TexelClass::Int));
    assert!(integer.accepts_format(TextureFormat::R32Sint));
    assert!(integer.accepts_format(TextureFormat::R32Uint));
    assert!(!integer.accepts_format(TextureFormat::R32Float));
    // A `ulong` texture is backed by `Rg32Uint` alone, and `Rg32Uint` backs nothing else.
    let ulong = usage(Some(TexelClass::Ulong));
    assert!(ulong.accepts_format(TextureFormat::Rg32Uint));
    for other in [
        TextureFormat::Rgba32Uint,
        TextureFormat::R32Uint,
        TextureFormat::Rg32Float,
    ] {
        assert!(!ulong.accepts_format(other), "{other:?}");
    }
    for other in [unsigned, integer, float] {
        assert!(!other.accepts_format(TextureFormat::Rg32Uint), "{other:?}");
    }
    assert!(usage(None).accepts_format(TextureFormat::Rg32Uint));
    assert_eq!(
        TextureUse {
            array: true,
            ..float
        }
        .texture_type(),
        Some(TextureType::D2Array)
    );
}

/// A kernel whose static initializer computes buffer 1's predicate from function constant 0 and
/// buffer 2's through float arithmetic the AIR layer does not evaluate (`frem`).
const FLOAT_GATED: &str = r#"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"
@flag.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@pred = internal addrspace(2) global i8 0, align 1
@scaled = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @init, ptr null }]
define internal void @init() section "air.static_init" {
  %v = load i8, ptr addrspace(2) @flag.MTL_FC_INIT_0_b, align 1
  %p = call i8 @air.normalize_function_constant_predicate.i8(i8 %v)
  store i8 %p, ptr addrspace(2) @pred, align 1
  %f = frem float 1.0, 2.0
  %c = fcmp ogt float %f, 0.0
  %z = zext i1 %c to i8
  store i8 %z, ptr addrspace(2) @scaled, align 1
  ret void
}
declare i8 @air.normalize_function_constant_predicate.i8(i8)
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
  store i32 1, ptr addrspace(1) %a, align 4
  ret void
}
!air.kernel = !{!0}
!air.function_constants = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.function_constant", !5, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
!5 = !{ptr addrspace(2) @pred, !"bool", !"flag"}
!6 = !{ptr addrspace(2) @flag.MTL_FC_INIT_0_b, !"bool", !"flag", i32 0}
!7 = !{i32 2, !"air.function_constant", !8, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"c"}
!8 = !{ptr addrspace(2) @scaled, !"bool", !"scaled"}
"#;

fn flag(value: u8) -> BTreeMap<u32, ConstValue> {
    BTreeMap::from([(
        0,
        ConstValue {
            ty: DataType::Bool,
            bytes: Box::new([value]),
        },
    )])
}

#[test]
fn only_what_float_arithmetic_computes_is_left_unknown_and_it_is_named() {
    // Buffer 1's predicate is a pure function of the constant, so it resolves and gates the
    // argument; buffer 2's goes through `frem`, so it alone is unknown: optional, and recorded.
    let buffers = |interface: &AirInterface| {
        interface
            .params_of(|role| match role {
                ParamRole::Buffer(buffer) => Some(buffer),
                _ => None,
            })
            .map(|(param, _)| (param.slots().unwrap().0, param.live))
            .collect::<Vec<_>>()
    };
    let on = of_text(FLOAT_GATED, "k", &flag(1)).unwrap();
    assert_eq!(buffers(&on), [(0, true), (1, true), (2, false)]);
    let off = of_text(FLOAT_GATED, "k", &flag(0)).unwrap();
    assert_eq!(buffers(&off), [(0, true), (1, false), (2, false)]);
    let [unresolved] = &on.unresolved[..] else {
        panic!("{:?}", on.unresolved);
    };
    assert_eq!(unresolved.value, Unresolved::Predicate { param: 2 });
    assert!(unresolved
        .to_string()
        .starts_with("the predicate of entry parameter 2 does not evaluate ("));
    assert!(unresolved
        .to_string()
        .ends_with("); the argument is optional"));
}

#[test]
fn an_air_layer_limit_is_a_checker_error_not_an_unknown_value() {
    // Under a work cap the evaluation cannot meet, the refusal is the checker's error: a limit,
    // a deadline, or cancellation is no fact about the case.
    let bitcode = assemble(FLOAT_GATED).unwrap();
    let decoded = DecodedModule::from_bitcode("test", &bitcode).unwrap();
    let abi = decoded.abi().unwrap();
    let mut tiny = Meter::with_caps(&Budget::default(), 1, u64::MAX);
    let error = AirInterface::metered(decoded.module(), abi, "k", &flag(1), &mut tiny).unwrap_err();
    assert!(
        error.starts_with("the AIR layer does not evaluate the metadata globals: "),
        "{error}"
    );
}

#[test]
fn object_roles_bind_their_slots() {
    let interface = AirInterface::synthetic(
        Stage::Kernel {
            max_total_threads: None,
            work_group_size: None,
        },
        vec![param(0, ParamRole::FunctionTable(object(3)))],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(interface.params[0].slots(), Some((3, 1)));
}
