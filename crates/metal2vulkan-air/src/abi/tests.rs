//! ABI decode tests over modules built with the IR test builder.

use super::check::{core_image_pointee_check, pointee_check};
use super::keys::{Key, BUFFER_ACCESS, TEXTURE_ACCESS};
use super::md::Site;
use super::*;
use crate::bitcode::PointeeTable;
use crate::budget::{Budget, Meter};
use crate::container::{ContainerConstant, FunctionPublicMetadata, FunctionStage, LibraryFunction};
use crate::error::{AbiError, DiagName, ErrorClass, InputErrorKind, Location};
use crate::error::{Error, Feature, Limit};
use crate::ir::build::ModuleBuilder;
use crate::ir::{
    AddrSpace, Const, ConstId, FuncId, Global, GlobalId, Linkage, Md, MdId, Module, Section, Type,
    TypeId, TypeLayouts,
};
use crate::metal::DataType;

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

// ---- A small metadata builder ----

/// One metadata operand.
#[derive(Clone, Copy)]
enum O<'a> {
    S(&'a str),
    /// An `i32`.
    I(u64),
    /// An `i1`.
    B(bool),
    G(GlobalId),
    F(FuncId),
    N(MdId),
    /// A `null` operand.
    Z,
}

use O::{B, F, G, I, N, S, Z};

fn value(b: &mut ModuleBuilder, bits: u8, v: u64) -> Option<MdId> {
    let ty = b.int(bits);
    let c = b.constant(ty, Const::Int(v));
    Some(b.md(Md::Value(c)))
}

fn node(b: &mut ModuleBuilder, ops: &[O<'_>]) -> MdId {
    let ops: Vec<Option<MdId>> = ops
        .iter()
        .map(|o| match *o {
            O::S(s) => Some(b.md_string(s)),
            O::I(v) => value(b, 32, v),
            O::B(v) => value(b, 1, u64::from(v)),
            O::G(g) => match b.global_addr(g) {
                crate::ir::Value::Const(c) => Some(b.md(Md::Value(c))),
                _ => None,
            },
            O::F(f) => {
                let ptr = b.ptr(AddrSpace::Thread);
                let c = b.constant(ptr, Const::FuncAddr(f));
                Some(b.md(Md::Value(c)))
            }
            O::N(n) => Some(n),
            O::Z => None,
        })
        .collect();
    b.md_node(&ops)
}

/// Define `name` with parameter types `params` returning `ret`; its body is `ret void` or
/// `unreachable`.
fn entry_fn(b: &mut ModuleBuilder, name: &str, ret: TypeId, params: &[TypeId]) -> FuncId {
    let ty = b.fn_ty(ret, params);
    let mut f = b.define(name, ty);
    if f.module().ty(Type::Void) == ret {
        f.ret(None);
    } else {
        f.push_void(crate::ir::InstKind::Unreachable);
    }
    f.finish()
}

/// The standard kernel: `k(device float* buf, texture2d tex, sampler smp, uint3 gid)`.
struct Kernel {
    b: ModuleBuilder,
    f: FuncId,
}

fn kernel_params(b: &mut ModuleBuilder) -> Vec<TypeId> {
    let dev = b.ptr(AddrSpace::Device);
    let cst = b.ptr(AddrSpace::Constant);
    let i32t = b.int(32);
    let v3 = b.vector(i32t, 3);
    vec![dev, dev, cst, v3]
}

fn buffer_arg<'a>(i: u64) -> Vec<O<'a>> {
    vec![
        I(i),
        S("air.buffer"),
        S("air.location_index"),
        I(0),
        I(1),
        S("air.read_write"),
        S("air.address_space"),
        I(1),
        S("air.arg_type_size"),
        I(4),
        S("air.arg_type_align_size"),
        I(4),
        S("air.arg_type_name"),
        S("float"),
        S("air.arg_name"),
        S("buf"),
    ]
}

fn texture_arg<'a>(i: u64) -> Vec<O<'a>> {
    vec![
        I(i),
        S("air.texture"),
        S("air.location_index"),
        I(2),
        I(1),
        S("air.sample"),
        S("air.arg_type_name"),
        S("texture2d<float, sample>"),
    ]
}

fn sampler_arg<'a>(i: u64) -> Vec<O<'a>> {
    vec![I(i), S("air.sampler"), S("air.location_index"), I(3), I(1)]
}

fn gid_arg<'a>(i: u64) -> Vec<O<'a>> {
    vec![I(i), S("air.thread_position_in_grid"), S("air.arg_unused")]
}

impl Kernel {
    /// The kernel with argument nodes `args` (replacing the standard four).
    fn with(args: &[Vec<O<'_>>]) -> Kernel {
        let mut b = ModuleBuilder::new();
        let params = kernel_params(&mut b);
        let void = b.void();
        let f = entry_fn(&mut b, "k", void, &params);
        let arg_nodes: Vec<O<'_>> = args.iter().map(|a| N(node(&mut b, a))).collect();
        let inputs = node(&mut b, &arg_nodes);
        let outputs = node(&mut b, &[]);
        let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
        b.named_md("air.kernel", &[e]);
        Kernel { b, f }
    }

    fn standard() -> Kernel {
        Kernel::with(&[buffer_arg(0), texture_arg(1), sampler_arg(2), gid_arg(3)])
    }

    fn decode(self) -> Result<AirAbi, Error> {
        let m = self.b.finish().expect("verified module");
        decode(&m, &PointeeTable::default(), &mut meter())
    }
}

fn abi_kind(e: Error) -> AbiError {
    match e {
        Error::InvalidInput(i) => match i.kind {
            InputErrorKind::Abi(a) => a,
            other => panic!("not an ABI error: {other:?}"),
        },
        other => panic!("not an input error: {other:?}"),
    }
}

fn expect_abi<T: std::fmt::Debug>(r: Result<T, Error>, want: AbiError) {
    assert_eq!(abi_kind(r.expect_err("decode must fail")), want);
}

// ---- Keys ----

#[test]
fn key_spellings_round_trip_and_are_unique() {
    // `Key::parse` and `Key::name` are matches, not scans of the table: this is what holds the
    // three in step. Every entry of the table must parse to its key and every key must name its
    // entry, so a key added to the table without its `parse` arm fails here, and one added to the
    // enum without its spelling is a compile error in `name`'s exhaustive match. `parse` also
    // debug-asserts that `name` is its inverse, so a mistyped arm fails wherever it is reached.
    let mut seen = std::collections::HashSet::new();
    for (name, k) in Key::all() {
        assert!(seen.insert(name), "{name} listed twice");
        assert_eq!(Key::parse(name.as_bytes()), Some(k));
        assert_eq!(k.name(), name);
        assert!(name.starts_with("air."), "{name}");
    }
    assert_eq!(Key::parse(b"air.not_a_key"), None);
}

// ---- Entry points and roles ----

#[test]
fn kernel_roles_decode() {
    let k = Kernel::standard();
    let f = k.f;
    let abi = k.decode().unwrap();
    assert_eq!(abi.entries.len(), 1);
    let e = abi.entry(f).unwrap();
    assert_eq!(
        e.stage,
        Stage::Kernel {
            max_total_threads: None
        }
    );
    assert!(e.outputs.is_empty());
    let roles: Vec<_> = e.params.iter().map(|p| (p.param, p.role.clone())).collect();
    let slot = |i| Slot {
        index: AbiInt::Const(i),
        count: AbiInt::Const(1),
    };
    assert_eq!(
        roles,
        vec![
            (
                0,
                ParamRole::Buffer(BufferArg {
                    location: slot(0),
                    access: Access::ReadWrite,
                    space: AddrSpace::Device,
                    buffer_size: None,
                    layout: None,
                    raster_order_group: None,
                })
            ),
            (
                1,
                ParamRole::Texture(TextureArg {
                    location: slot(2),
                    access: Access::Sample,
                    raster_order_group: None,
                })
            ),
            (2, ParamRole::Sampler { location: slot(3) }),
            (3, ParamRole::Builtin(Builtin::ThreadPositionInGrid)),
        ]
    );
    assert_eq!(e.params[0].ty.size, Some(4));
    assert_eq!(e.params[0].ty.align, Some(4));
    assert_eq!(e.params[0].name, Some(DiagName::new("buf")));
    assert!(e.params[3].unused);
    assert!(!e.params[0].unused);
    // The handle typing's seeds.
    let handles: Vec<_> = e.params.iter().map(|p| p.role.handle()).collect();
    assert_eq!(
        handles,
        vec![
            None,
            Some(AbiHandle::Texture(Access::Sample)),
            Some(AbiHandle::Sampler),
            None
        ]
    );
}

#[test]
fn unknown_key_is_refused() {
    let mut buf = buffer_arg(0);
    buf.push(S("air.no_such_key"));
    let k = Kernel::with(&[buf, texture_arg(1), sampler_arg(2), gid_arg(3)]);
    expect_abi(k.decode(), AbiError::UnknownKey);
}

/// MSL 3.2's program-scope buffer bindings (`!air.global_bindings`) are valid AIR this version
/// does not implement: refused as unsupported, not as malformed input.
#[test]
fn program_scope_bindings_are_unsupported_not_malformed() {
    let mut k = Kernel::standard();
    let binding = node(&mut k.b, &[S("air.global_binding")]);
    k.b.named_md("air.global_bindings", &[binding]);
    let err = k.decode().unwrap_err();
    let Error::Unsupported(u) = err else {
        panic!("not unsupported: {err:?}")
    };
    assert_eq!(u.feature, Feature::ProgramScopeBinding);
    // Any other named list outside the closed set is still a key outside the contract.
    let mut k = Kernel::standard();
    let other = node(&mut k.b, &[S("air.no_such_entry")]);
    k.b.named_md("air.no_such_list", &[other]);
    expect_abi(k.decode(), AbiError::UnknownKey);
}

#[test]
fn key_outside_its_role_is_refused() {
    let mut tex = texture_arg(1);
    tex.extend([S("air.buffer_size"), I(16)]);
    let k = Kernel::with(&[buffer_arg(0), tex, sampler_arg(2), gid_arg(3)]);
    expect_abi(k.decode(), AbiError::UnknownKey);
    // A builtin takes no location.
    let mut gid = gid_arg(3);
    gid.extend([S("air.location_index"), I(0), I(1)]);
    let k = Kernel::with(&[buffer_arg(0), texture_arg(1), sampler_arg(2), gid]);
    expect_abi(k.decode(), AbiError::UnknownKey);
}

#[test]
fn missing_key_is_refused() {
    let sampler = vec![I(2), S("air.sampler")];
    let k = Kernel::with(&[buffer_arg(0), texture_arg(1), sampler, gid_arg(3)]);
    expect_abi(k.decode(), AbiError::MissingKey);
}

/// `buffer_arg(i)` without its `air.address_space` key and value, as an older compiler writes a
/// buffer's node (the AIR 2.2 of macOS 13's system libraries).
fn buffer_arg_without_space<'a>(i: u64) -> Vec<O<'a>> {
    buffer_arg(i)
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !(6..=7).contains(i))
        .map(|(_, o)| o)
        .collect()
}

#[test]
fn a_buffer_without_an_address_space_takes_its_pointer_s() {
    // Parameters 0 and 1 are device pointers and 2 a constant one (`kernel_params`).
    let k = Kernel::with(&[
        buffer_arg_without_space(0),
        buffer_arg(1),
        buffer_arg_without_space(2),
        gid_arg(3),
    ]);
    let f = k.f;
    let abi = k.decode().unwrap();
    let spaces: Vec<_> = abi
        .entry(f)
        .unwrap()
        .params
        .iter()
        .map(|p| match &p.role {
            ParamRole::Buffer(b) => Some(b.space),
            _ => None,
        })
        .collect();
    assert_eq!(
        spaces,
        [
            Some(AddrSpace::Device),
            Some(AddrSpace::Device),
            Some(AddrSpace::Constant),
            None
        ]
    );
    // An `array_ref` of buffers is a thread-space pointer, which does not carry its elements'
    // space: the key is still needed.
    let mut b = ModuleBuilder::new();
    let thread = b.ptr(AddrSpace::Thread);
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &[thread]);
    let ops = buffer_arg_without_space(0);
    let arg = node(&mut b, &ops);
    let inputs = node(&mut b, &[N(arg)]);
    let outputs = node(&mut b, &[]);
    let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
    b.named_md("air.kernel", &[e]);
    let m = b.finish().unwrap();
    expect_abi(
        decode(&m, &PointeeTable::default(), &mut meter()),
        AbiError::MissingKey,
    );
}

#[test]
fn wrong_value_type_is_refused() {
    let sampler = vec![
        I(2),
        S("air.sampler"),
        S("air.location_index"),
        S("three"),
        I(1),
    ];
    let k = Kernel::with(&[buffer_arg(0), texture_arg(1), sampler, gid_arg(3)]);
    expect_abi(k.decode(), AbiError::BadValue);
}

#[test]
fn repeated_key_is_refused() {
    let mut buf = buffer_arg(0);
    buf.extend([S("air.arg_name"), S("again")]);
    let k = Kernel::with(&[buf, texture_arg(1), sampler_arg(2), gid_arg(3)]);
    expect_abi(k.decode(), AbiError::BadValue);
    // Two access keys.
    let mut tex = texture_arg(1);
    tex.push(S("air.read"));
    let k = Kernel::with(&[buffer_arg(0), tex, sampler_arg(2), gid_arg(3)]);
    expect_abi(k.decode(), AbiError::BadValue);
}

#[test]
fn role_must_match_parameter_type() {
    // The texture (param 1, a device pointer) described as a sampler (constant pointer).
    let k = Kernel::with(&[buffer_arg(0), sampler_arg(1), sampler_arg(2), gid_arg(3)]);
    expect_abi(k.decode(), AbiError::RoleTypeMismatch);
    // A builtin on a pointer parameter.
    let k = Kernel::with(&[gid_arg(0), texture_arg(1), sampler_arg(2), gid_arg(3)]);
    expect_abi(k.decode(), AbiError::RoleTypeMismatch);
    // A buffer whose address space disagrees with its pointer.
    let mut buf = buffer_arg(0);
    buf[7] = I(2);
    let k = Kernel::with(&[buf, texture_arg(1), sampler_arg(2), gid_arg(3)]);
    expect_abi(k.decode(), AbiError::RoleTypeMismatch);
}

#[test]
fn every_parameter_needs_exactly_one_binding() {
    let k = Kernel::with(&[buffer_arg(0), texture_arg(1), sampler_arg(2)]);
    expect_abi(k.decode(), AbiError::SignatureMismatch);
    let k = Kernel::with(&[buffer_arg(0), texture_arg(1), sampler_arg(2), gid_arg(2)]);
    expect_abi(k.decode(), AbiError::SignatureMismatch);
    let k = Kernel::with(&[buffer_arg(0), texture_arg(1), sampler_arg(2), gid_arg(7)]);
    expect_abi(k.decode(), AbiError::SignatureMismatch);
}

#[test]
fn array_ref_tables_are_thread_pointers() {
    let mut b = ModuleBuilder::new();
    let thread = b.ptr(AddrSpace::Thread);
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &[thread]);
    let i32t = b.int(32);
    let count = b.simple_global("count", i32t, AddrSpace::Constant, None);
    let arg = node(
        &mut b,
        &[
            I(0),
            S("air.texture"),
            S("air.location_index"),
            I(0),
            G(count),
            S("air.read"),
        ],
    );
    let inputs = node(&mut b, &[N(arg)]);
    let outputs = node(&mut b, &[]);
    let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
    b.named_md("air.kernel", &[e]);
    let m = b.finish().unwrap();
    let abi = decode(&m, &PointeeTable::default(), &mut meter()).unwrap();
    let ParamRole::Texture(t) = &abi.entries[0].params[0].role else {
        panic!("texture")
    };
    assert_eq!(t.location.count, AbiInt::Global(count));
}

#[test]
fn kernel_attributes_and_predicates() {
    let mut b = ModuleBuilder::new();
    let params = kernel_params(&mut b);
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &params);
    let i8t = b.int(8);
    let zero = b.constant(i8t, Const::Int(0));
    let pred = b.simple_global("pred", i8t, AddrSpace::Constant, Some(zero));
    let pnode = node(&mut b, &[G(pred), S("bool"), S("use_tex")]);
    let mut tex = vec![I(1), S("air.function_constant"), N(pnode)];
    tex.extend(texture_arg(1).into_iter().skip(1));
    let args = [buffer_arg(0), tex, sampler_arg(2), gid_arg(3)];
    let arg_nodes: Vec<_> = args.iter().map(|a| N(node(&mut b, a))).collect();
    let inputs = node(&mut b, &arg_nodes);
    let outputs = node(&mut b, &[]);
    let attr = node(&mut b, &[S("air.max_work_group_size"), I(64)]);
    let e = node(&mut b, &[F(f), N(outputs), N(inputs), N(attr)]);
    b.named_md("air.kernel", &[e]);
    let m = b.finish().unwrap();
    let abi = decode(&m, &PointeeTable::default(), &mut meter()).unwrap();
    let e = &abi.entries[0];
    assert_eq!(
        e.stage,
        Stage::Kernel {
            max_total_threads: Some(64)
        }
    );
    assert_eq!(
        e.params[1].predicate,
        Some(Predicate {
            global: pred,
            name: DiagName::new("use_tex")
        })
    );
}

/// A kernel whose root carries the trailing attribute `attr` builds.
fn kernel_with_attribute(
    attr: impl FnOnce(&mut ModuleBuilder) -> O<'static>,
) -> Result<AirAbi, Error> {
    let mut b = ModuleBuilder::new();
    let params = kernel_params(&mut b);
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &params);
    let args = [buffer_arg(0), texture_arg(1), sampler_arg(2), gid_arg(3)];
    let arg_nodes: Vec<_> = args.iter().map(|a| N(node(&mut b, a))).collect();
    let inputs = node(&mut b, &arg_nodes);
    let outputs = node(&mut b, &[]);
    let attr = attr(&mut b);
    let e = node(&mut b, &[F(f), N(outputs), N(inputs), attr]);
    b.named_md("air.kernel", &[e]);
    let m = b.finish().unwrap();
    decode(&m, &PointeeTable::default(), &mut meter())
}

/// An attribute the kernel's stage does not read is refused naming the key the root states: a
/// string operand, a keyed node's first string, or another stage's attribute. An operand that
/// states no key is `UnknownKey`.
#[test]
fn an_unread_stage_attribute_is_refused_by_its_key() {
    let named = |name: &str| AbiError::UnknownStageAttribute(DiagName::new(name));
    expect_abi(
        kernel_with_attribute(|_| S("air.invented_stage_attribute")),
        named("air.invented_stage_attribute"),
    );
    expect_abi(
        kernel_with_attribute(|b| N(node(b, &[S("air.invented_stage_attribute"), I(1)]))),
        named("air.invented_stage_attribute"),
    );
    expect_abi(
        kernel_with_attribute(|_| S("early_fragment_tests")),
        named("early_fragment_tests"),
    );
    expect_abi(
        kernel_with_attribute(|b| {
            N(node(
                b,
                &[
                    S("air.patch"),
                    S("triangle"),
                    S("air.patch_control_point"),
                    I(3),
                ],
            ))
        }),
        named("air.patch"),
    );
    expect_abi(kernel_with_attribute(|_| I(1)), AbiError::UnknownKey);
    expect_abi(
        kernel_with_attribute(|b| N(node(b, &[I(1)]))),
        AbiError::UnknownKey,
    );
    // The kernel's own attribute is read.
    let abi = kernel_with_attribute(|b| N(node(b, &[S("air.max_work_group_size"), I(32)])))
        .expect("the kernel's own attribute decodes");
    assert_eq!(
        abi.entries[0].stage,
        Stage::Kernel {
            max_total_threads: Some(32)
        }
    );
}

/// The standard kernel whose texture is gated by a predicate global built by `global`.
fn kernel_with_predicate(
    global: impl FnOnce(&mut ModuleBuilder) -> GlobalId,
) -> Result<AirAbi, Error> {
    let mut b = ModuleBuilder::new();
    let params = kernel_params(&mut b);
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &params);
    let pred = global(&mut b);
    let pnode = node(&mut b, &[G(pred), S("bool"), S("use_tex")]);
    let mut tex = vec![I(1), S("air.function_constant"), N(pnode)];
    tex.extend(texture_arg(1).into_iter().skip(1));
    let args = [buffer_arg(0), tex, sampler_arg(2), gid_arg(3)];
    let arg_nodes: Vec<_> = args.iter().map(|a| N(node(&mut b, a))).collect();
    let inputs = node(&mut b, &arg_nodes);
    let outputs = node(&mut b, &[]);
    let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
    b.named_md("air.kernel", &[e]);
    let m = b.finish().unwrap();
    decode(&m, &PointeeTable::default(), &mut meter())
}

#[test]
fn predicates_name_a_static_initializer_integer_global() {
    // As the frontend writes them: `internal addrspace(2) global i8 0`, and a few `i16`/`i32`
    // predicates.
    for bits in [8, 16, 32] {
        let abi = kernel_with_predicate(|b| {
            let t = b.int(bits);
            let zero = b.constant(t, Const::Int(0));
            b.simple_global("pred", t, AddrSpace::Constant, Some(zero))
        });
        assert!(abi.is_ok(), "i{bits}: {abi:?}");
    }
    let refused = |abi: Result<AirAbi, Error>| {
        let site = Site {
            entry: Some(DiagName::new("k")),
            arg: Some(1),
        };
        assert_eq!(
            abi,
            Err(site.err(AbiError::BadValue, Some("air.function_constant")))
        );
    };
    // A function-constant initializer is the constant itself, not a predicate.
    refused(kernel_with_predicate(|b| {
        fc_global(b, "fc0", Some(Section::FcInitializer))
    }));
    // Not an integer.
    refused(kernel_with_predicate(|b| {
        let t = b.float();
        let zero = b.constant(t, Const::Undef);
        b.simple_global("pred", t, AddrSpace::Constant, Some(zero))
    }));
    // A declaration: the static initializer has nothing to store into.
    refused(kernel_with_predicate(|b| {
        let t = b.int(8);
        b.simple_global("pred", t, AddrSpace::Constant, None)
    }));
    // Another address space.
    refused(kernel_with_predicate(|b| {
        let t = b.int(8);
        let zero = b.constant(t, Const::Int(0));
        b.simple_global("pred", t, AddrSpace::Device, Some(zero))
    }));
}

/// A vertex function returning `{ float4, float2 }` and a fragment function reading the
/// varying and writing one color.
fn graphics(outputs_extra: bool) -> Result<AirAbi, Error> {
    let mut b = ModuleBuilder::new();
    let f32t = b.float();
    let v4 = b.vector(f32t, 4);
    let v2 = b.vector(f32t, 2);
    let i32t = b.int(32);
    let ret = b.struct_ty(&[v4, v2]);
    let vs = entry_fn(&mut b, "vs", ret, &[i32t]);
    let fs = entry_fn(&mut b, "fs", v4, &[v4, v2]);
    let pos = node(
        &mut b,
        &[
            S("air.position"),
            S("air.invariant"),
            S("air.arg_type_name"),
            S("float4"),
        ],
    );
    let uv = node(
        &mut b,
        &[
            S("air.vertex_output"),
            S("generated(2uvDv2_f)"),
            S("air.shared"),
        ],
    );
    let mut outs = vec![N(pos), N(uv)];
    if outputs_extra {
        let ps = node(&mut b, &[S("air.point_size")]);
        outs.push(N(ps));
    }
    let vs_out = node(&mut b, &outs);
    let vid = node(&mut b, &[I(0), S("air.vertex_id")]);
    let vs_in = node(&mut b, &[N(vid)]);
    let vs_e = node(&mut b, &[F(vs), N(vs_out), N(vs_in)]);
    b.named_md("air.vertex", &[vs_e]);
    let rt = node(&mut b, &[S("air.render_target"), I(0), I(0)]);
    let fs_out = node(&mut b, &[N(rt)]);
    let fpos = node(
        &mut b,
        &[
            I(0),
            S("air.position"),
            S("air.center"),
            S("air.no_perspective"),
        ],
    );
    let fuv = node(
        &mut b,
        &[
            I(1),
            S("air.fragment_input"),
            S("generated(2uvDv2_f)"),
            S("air.centroid"),
            S("air.perspective"),
        ],
    );
    let fs_in = node(&mut b, &[N(fpos), N(fuv)]);
    let fs_e = node(
        &mut b,
        &[F(fs), N(fs_out), N(fs_in), S("early_fragment_tests")],
    );
    b.named_md("air.fragment", &[fs_e]);
    let m = b.finish().unwrap();
    decode(&m, &PointeeTable::default(), &mut meter())
}

#[test]
fn vertex_and_fragment_interfaces_decode() {
    let abi = graphics(false).unwrap();
    let [vs, fs] = &abi.entries[..] else {
        panic!("two entries")
    };
    assert_eq!(vs.stage, Stage::Vertex { patch: None });
    assert_eq!(vs.params[0].role, ParamRole::Builtin(Builtin::VertexId));
    let key = UserKey(b"generated(2uvDv2_f)".as_slice().into());
    assert_eq!(
        vs.outputs
            .iter()
            .map(|o| (o.role.clone(), o.shared))
            .collect::<Vec<_>>(),
        vec![
            (OutputRole::Position { invariant: true }, false),
            (OutputRole::VertexOutput(key.clone()), true),
        ]
    );
    assert_eq!(
        fs.stage,
        Stage::Fragment {
            early_fragment_tests: true
        }
    );
    assert_eq!(
        fs.params[0].role,
        ParamRole::Builtin(Builtin::Position(Some(Interpolation::Smooth {
            sampling: Sampling::Center,
            perspective: false
        })))
    );
    assert_eq!(
        fs.params[1].role,
        ParamRole::FragmentInput(Varying {
            key,
            interpolation: Interpolation::Smooth {
                sampling: Sampling::Centroid,
                perspective: true
            }
        })
    );
    assert_eq!(
        fs.outputs[0].role,
        OutputRole::RenderTarget {
            attachment: AbiInt::Const(0),
            index: 0,
            raster_order_group: None
        }
    );
}

/// Every AIR up to 2.1 restates the MSL member's interpolation qualifier on `air.vertex_output` as
/// well as on the fragment input that receives it, and restates it partly (a
/// `[[centroid_perspective]]` output carries `air.centroid` alone). AIR 2.2 and later write the
/// same module without the keys, so a vertex output decodes with or without them, and to the same
/// role either way. Which keys are present is read from the node, so one case covers the range.
///
/// The spellings below are the emitter's whole set, measured by compiling the seven MSL
/// interpolation qualifiers at every reachable deployment target: AIR 1.6
/// (`-mios-version-min=8.0`), 1.8 (`10.11`), 1.11 (`10.12`), 2.0 (`10.13`) and 2.1 (`10.14`)
/// write them, and 2.2 (`10.15`) and later write none. `air.centroid` alone is the only partial
/// one; the matching `air.fragment_input` spells `air.centroid, air.perspective` at every
/// version, and a bare sampling key there is still refused
/// (`interpolation_must_be_complete`), so accepting the bare form here does not make a bare
/// `air.centroid` mean anything where interpolation is read.
#[test]
fn a_vertex_output_accepts_the_interpolation_keys_older_air_states_on_it() {
    let vertex_output = |keys: &[&'static str]| {
        let mut b = ModuleBuilder::new();
        let f32t = b.float();
        let v4 = b.vector(f32t, 4);
        let v2 = b.vector(f32t, 2);
        let i32t = b.int(32);
        let ret = b.struct_ty(&[v4, v2]);
        let vs = entry_fn(&mut b, "vs", ret, &[i32t]);
        let pos = node(&mut b, &[S("air.position")]);
        let mut ops = vec![S("air.vertex_output"), S("generated(2uvDv2_f)")];
        ops.extend(keys.iter().map(|&k| S(k)));
        ops.extend([S("air.arg_type_name"), S("float2")]);
        let uv = node(&mut b, &ops);
        let outs = node(&mut b, &[N(pos), N(uv)]);
        let vid = node(&mut b, &[I(0), S("air.vertex_id")]);
        let ins = node(&mut b, &[N(vid)]);
        let e = node(&mut b, &[F(vs), N(outs), N(ins)]);
        b.named_md("air.vertex", &[e]);
        let m = b.finish().unwrap();
        decode(&m, &PointeeTable::default(), &mut meter())
    };
    let key = UserKey(b"generated(2uvDv2_f)".as_slice().into());
    // Each spelling Apple's AIR 2.1 emitter writes on a vertex output: the complete pairs, the
    // partial `air.centroid` of a `[[centroid_perspective]]` member, and `air.flat`; and the
    // AIR 2.2 and later form, which writes none.
    for keys in [
        &[][..],
        &["air.center", "air.perspective"],
        &["air.center", "air.no_perspective"],
        &["air.centroid"],
        &["air.centroid", "air.no_perspective"],
        &["air.sample", "air.perspective"],
        &["air.sample", "air.no_perspective"],
        &["air.flat"],
    ] {
        let abi = vertex_output(keys).unwrap_or_else(|e| panic!("{keys:?}: {e:?}"));
        let out = &abi.entries[0].outputs[1];
        assert_eq!(out.role, OutputRole::VertexOutput(key.clone()), "{keys:?}");
    }
    // A key this role does not define is still refused there (`air.invariant` is the position
    // output's).
    expect_abi(vertex_output(&["air.invariant"]), AbiError::UnknownKey);
}

/// A fragment function reading `uint` input `mask` with the sample-mask keys `keys`, declaring
/// `[[early_fragment_tests]]` where `early`.
fn fragment_mask(keys: &[&'static str], early: bool) -> Result<AirAbi, Error> {
    let mut b = ModuleBuilder::new();
    let f32t = b.float();
    let v4 = b.vector(f32t, 4);
    let i32t = b.int(32);
    let fs = entry_fn(&mut b, "fs", v4, &[i32t]);
    let rt = node(&mut b, &[S("air.render_target"), I(0), I(0)]);
    let fs_out = node(&mut b, &[N(rt)]);
    let mut ops = vec![I(0)];
    ops.extend(keys.iter().map(|&k| S(k)));
    ops.extend([
        S("air.arg_type_name"),
        S("uint"),
        S("air.arg_name"),
        S("mask"),
    ]);
    let mask = node(&mut b, &ops);
    let fs_in = node(&mut b, &[N(mask)]);
    let mut entry = vec![F(fs), N(fs_out), N(fs_in)];
    if early {
        entry.push(S("early_fragment_tests"));
    }
    let e = node(&mut b, &entry);
    b.named_md("air.fragment", &[e]);
    let m = b.finish().unwrap();
    decode(&m, &PointeeTable::default(), &mut meter())
}

/// `[[sample_mask, post_depth_coverage]]` (`air.sample_mask_in` with `air.post_depth_coverage`,
/// as Apple's compiler writes it) decodes as the coverage after the depth test, in a function that
/// declares `[[early_fragment_tests]]`, which Metal's compiler requires of it; without them, and
/// on another builtin, the key is refused.
#[test]
fn post_depth_coverage_decodes_with_early_tests() {
    let role = |abi: AirAbi| abi.entries[0].params[0].role.clone();
    assert_eq!(
        role(fragment_mask(&["air.sample_mask_in"], false).unwrap()),
        ParamRole::Builtin(Builtin::SampleMaskIn)
    );
    let post = ["air.sample_mask_in", "air.post_depth_coverage"];
    assert_eq!(
        role(fragment_mask(&post, true).unwrap()),
        ParamRole::Builtin(Builtin::PostDepthCoverage)
    );
    expect_abi(fragment_mask(&post, false), AbiError::MissingKey);
    expect_abi(
        fragment_mask(&["air.sample_id", "air.post_depth_coverage"], true),
        AbiError::UnknownKey,
    );
}

#[test]
fn outputs_must_match_the_return_value() {
    expect_abi(graphics(true), AbiError::SignatureMismatch);
}

#[test]
fn kernels_have_no_outputs() {
    let mut b = ModuleBuilder::new();
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &[]);
    let rt = node(&mut b, &[S("air.render_target"), I(0), I(0)]);
    let outs = node(&mut b, &[N(rt)]);
    let ins = node(&mut b, &[]);
    let e = node(&mut b, &[F(f), N(outs), N(ins)]);
    b.named_md("air.kernel", &[e]);
    let m = b.finish().unwrap();
    expect_abi(
        decode(&m, &PointeeTable::default(), &mut meter()),
        AbiError::SignatureMismatch,
    );
}

/// One entry of the named-metadata list `list` with the single value input `input` and, when
/// given, the single output `output`.
fn single_io(list: &str, input: &[O<'_>], output: Option<&[O<'_>]>) -> Result<AirAbi, Error> {
    let mut b = ModuleBuilder::new();
    let i32t = b.int(32);
    let ret = match output {
        Some(_) => b.float(),
        None => b.void(),
    };
    let f = entry_fn(&mut b, "e", ret, &[i32t]);
    let outs: Vec<_> = output.iter().map(|o| N(node(&mut b, o))).collect();
    let outs = node(&mut b, &outs);
    let arg = node(&mut b, input);
    let ins = node(&mut b, &[N(arg)]);
    let e = node(&mut b, &[F(f), N(outs), N(ins)]);
    b.named_md(list, &[e]);
    let m = b.finish().unwrap();
    decode(&m, &PointeeTable::default(), &mut meter())
}

/// An output's `air.render_target` states its attachment and then its dual-source index, except
/// on an emitter with no `[[index(n)]]` to state: AIR 1.8 and earlier write the attachment alone
/// and the trailing attribute keys follow it directly. Both decode, to the same attachment, and
/// the shorter one to index 0, the only index its language version has. The operand after the
/// attachment decides, not the recorded version: an index is an integer, an attribute key a
/// string. Transcribed from `xcrun metal -c -mmacosx-version-min=10.11` (`air64_v18`) and
/// `-mmacosx-version-min=10.12` (`air64_v111`).
#[test]
fn a_render_target_without_a_dual_source_index_is_index_zero() {
    let varying: &[O<'_>] = &[I(0), S("air.fragment_input"), S("user(v)"), S("air.flat")];
    // AIR 1.11 and later; AIR 1.8 and earlier.
    let both: &[O<'_>] = &[S("air.render_target"), I(0), I(0)];
    let attachment_only: &[O<'_>] = &[S("air.render_target"), I(0)];
    let want = OutputRole::RenderTarget {
        attachment: AbiInt::Const(0),
        index: 0,
        raster_order_group: None,
    };
    for (shape, output) in [
        ("attachment and index", both),
        ("attachment only", attachment_only),
    ] {
        let abi = single_io("air.fragment", varying, Some(output))
            .unwrap_or_else(|e| panic!("{shape}: {e:?}"));
        assert_eq!(abi.entries[0].outputs[0].role, want, "{shape}");
    }
    // A stated index is still read, so the two shapes are told apart rather than conflated.
    let second_source: &[O<'_>] = &[S("air.render_target"), I(0), I(1)];
    let abi = single_io("air.fragment", varying, Some(second_source)).unwrap();
    assert_eq!(
        abi.entries[0].outputs[0].role,
        OutputRole::RenderTarget {
            attachment: AbiInt::Const(0),
            index: 1,
            raster_order_group: None
        }
    );
    // A fragment function's `[[color(n)]]` input states the attachment alone at every version,
    // AIR 2.9 included, so an input is unchanged by this.
    let color_input: &[O<'_>] = &[I(0), S("air.render_target"), I(0)];
    let abi = single_io("air.fragment", color_input, Some(both)).unwrap();
    assert_eq!(
        abi.entries[0].params[0].role,
        ParamRole::ColorInput {
            index: AbiInt::Const(0),
            raster_order_group: None
        }
    );
}

/// A buffer member of an argument buffer states no `air.address_space` before AIR 2.5: the key
/// arrived there, and the member has no parameter whose pointer type could carry it. The holding
/// struct's field at the member's byte offset carries it instead, reached from the argument
/// buffer's parameter pointee, so a member of a typed-pointer module reads its true space
/// (`an_argument_buffer_member_takes_its_space_from_the_holding_field`). Where the bitcode states
/// no holding type at all -- opaque pointers erase the pointee -- the member decodes to device
/// space, which Metal cannot tell from `constant` in an argument buffer (both are
/// `setBuffer:offset:atIndex:`) and which claims no uniform-block containment. A member that
/// states a space keeps it, and a top-level buffer with no parameter pointer still needs the key.
/// Transcribed from `xcrun metal -c -mmacosx-version-min=12.0` (`air64_v24`) and `=13.0`
/// (`air64_v25`).
#[test]
fn an_argument_buffer_member_without_an_address_space_is_device() {
    let member = |space: Option<u64>| {
        let mut m = vec![
            I(2),
            S("air.buffer"),
            S("air.location_index"),
            I(2),
            I(1),
            S("air.read"),
        ];
        if let Some(raw) = space {
            m.extend([S("air.address_space"), I(raw)]);
        }
        m.extend([S("air.arg_type_size"), I(16)]);
        m
    };
    let decoded = |space: Option<u64>| {
        let mut b = ModuleBuilder::new();
        let params = kernel_params(&mut b);
        let void = b.void();
        let f = entry_fn(&mut b, "k", void, &params);
        let member = node(&mut b, &member(space));
        let layout = node(
            &mut b,
            &[
                I(0),
                I(16),
                I(0),
                S("float4"),
                S("b"),
                S("air.indirect_argument"),
                N(member),
            ],
        );
        let buf = vec![
            I(0),
            S("air.indirect_buffer"),
            S("air.buffer_size"),
            I(16),
            S("air.location_index"),
            I(0),
            I(1),
            S("air.read"),
            S("air.address_space"),
            I(1),
            S("air.struct_type_info"),
            N(layout),
            S("air.arg_type_size"),
            I(16),
        ];
        let args = [buf, texture_arg(1), sampler_arg(2), gid_arg(3)];
        let arg_nodes: Vec<_> = args.iter().map(|a| N(node(&mut b, a))).collect();
        let inputs = node(&mut b, &arg_nodes);
        let outputs = node(&mut b, &[]);
        let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
        b.named_md("air.kernel", &[e]);
        let m = b.finish().unwrap();
        let abi = decode(&m, &PointeeTable::default(), &mut meter())?;
        let ParamRole::ArgumentBuffer(arg) = &abi.entries[0].params[0].role else {
            panic!("argument buffer")
        };
        let info = abi.struct_info(arg.layout.unwrap()).unwrap();
        let MemberKind::Indirect(member) = &info.members[0].kind else {
            panic!("indirect member")
        };
        let ParamRole::Buffer(b) = &member.role else {
            panic!("buffer member, not {:?}", member.role)
        };
        Ok::<AddrSpace, Error>(b.space)
    };
    // No key: device. Device and constant stated: as stated.
    assert_eq!(decoded(None).unwrap(), AddrSpace::Device);
    assert_eq!(decoded(Some(1)).unwrap(), AddrSpace::Device);
    assert_eq!(decoded(Some(2)).unwrap(), AddrSpace::Constant);
    // The default is the member's alone: a top-level buffer still takes its parameter's pointer
    // space (`a_buffer_without_an_address_space_takes_its_pointer_s`), and one with no carrier at
    // all is still `MissingKey` (`missing_key_is_refused`).
}

/// The member's own field of the struct the argument buffer's parameter points to decides its
/// address space when its node states none. The field is selected by the member's byte offset --
/// no type name, no function name, and no recorded AIR version is read -- so one reading covers
/// every emitter that writes the key-less shape. Three answers, by what the module states:
///
/// - a pointer field: that pointer's space, so a `constant` member reads as constant and a
///   `device` member as device, exactly as AIR 2.5 and later spell them;
/// - a field that is not a pointer: the module's types and the member's buffer role disagree, so
///   `BadValue` rather than a space picked for it;
/// - no holding type (opaque-pointer bitcode): the documented device default
///   (`an_argument_buffer_member_without_an_address_space_is_device`).
///
/// The field offsets are the holding struct's own, so a padding field before the member does not
/// shift the answer. The layout shape is that of `xcrun metal -c -mmacosx-version-min=12.0`
/// (`air64_v24`) on an argument buffer of a `constant float4 *` and a `device float4 *`.
#[test]
fn an_argument_buffer_member_takes_its_space_from_the_holding_field() {
    // One argument-buffer member at `offset`, stating no address space.
    let member_at = |b: &mut ModuleBuilder, id: u64, offset: u64| {
        let arg = node(
            b,
            &[
                I(id),
                S("air.buffer"),
                S("air.location_index"),
                I(id),
                I(1),
                S("air.read"),
                S("air.arg_type_size"),
                I(16),
            ],
        );
        vec![
            I(offset),
            I(16),
            I(0),
            S("float4"),
            S("m"),
            S("air.indirect_argument"),
            N(arg),
        ]
    };
    // A kernel taking one argument buffer whose holding struct is `fields`, with members at the
    // byte offsets `at`. `holder` selects whether the parameter's pointee is recorded.
    let decoded = |fields: &dyn Fn(&mut ModuleBuilder) -> Vec<TypeId>, at: &[u64], holder: bool| {
        let mut b = ModuleBuilder::new();
        // The argument buffer is a `constant` pointer, as its node states (`check::role_type`);
        // the rest are the standard kernel's texture, sampler and grid position.
        let dev = b.ptr(AddrSpace::Device);
        let cst = b.ptr(AddrSpace::Constant);
        let i32t = b.int(32);
        let v3 = b.vector(i32t, 3);
        let void = b.void();
        let f = entry_fn(&mut b, "k", void, &[cst, dev, cst, v3]);
        let field_tys = fields(&mut b);
        let ab = b.struct_ty(&field_tys);
        let mut layout_ops = Vec::new();
        for (i, &offset) in at.iter().enumerate() {
            layout_ops.extend(member_at(&mut b, i as u64, offset));
        }
        let layout = node(&mut b, &layout_ops);
        let buf = vec![
            I(0),
            S("air.indirect_buffer"),
            S("air.location_index"),
            I(0),
            I(1),
            S("air.read"),
            S("air.address_space"),
            I(2),
            S("air.struct_type_info"),
            N(layout),
        ];
        let args = [buf, texture_arg(1), sampler_arg(2), gid_arg(3)];
        let arg_nodes: Vec<_> = args.iter().map(|a| N(node(&mut b, a))).collect();
        let inputs = node(&mut b, &arg_nodes);
        let outputs = node(&mut b, &[]);
        let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
        b.named_md("air.kernel", &[e]);
        let m = b.finish().unwrap();
        // Parameter 0 of function 0 points at the argument buffer; the rest are opaque.
        let pointees = if holder {
            PointeeTable::from_params(&[&[Some(ab), None, None, None]])
        } else {
            PointeeTable::default()
        };
        let abi = decode(&m, &pointees, &mut meter())?;
        let ParamRole::ArgumentBuffer(arg) = &abi.entries[0].params[0].role else {
            panic!("argument buffer")
        };
        let info = abi.struct_info(arg.layout.unwrap()).unwrap();
        let spaces: Vec<AddrSpace> = info
            .members
            .iter()
            .map(|mem| match &mem.kind {
                MemberKind::Indirect(member) => match &member.role {
                    ParamRole::Buffer(b) => b.space,
                    other => panic!("buffer member, not {other:?}"),
                },
                other => panic!("indirect member, not {other:?}"),
            })
            .collect();
        Ok::<Vec<AddrSpace>, Error>(spaces)
    };

    // `struct { constant float4 *ro; device float4 *rw; }`: each member reads its field's space,
    // so the constant member is no longer read as device.
    let two_pointers = |b: &mut ModuleBuilder| {
        let cst = b.ptr(AddrSpace::Constant);
        let dev = b.ptr(AddrSpace::Device);
        vec![cst, dev]
    };
    assert_eq!(
        decoded(&two_pointers, &[0, 8], true).unwrap(),
        [AddrSpace::Constant, AddrSpace::Device]
    );
    // The same metadata with no pointee recorded keeps the device default, so the field is what
    // moved the answer and not the member's own node.
    assert_eq!(
        decoded(&two_pointers, &[0, 8], false).unwrap(),
        [AddrSpace::Device, AddrSpace::Device]
    );

    // `struct { array<device float4 *, 2> arr; }`: an array member's field is the array, whose
    // elements carry the space.
    let array_of_pointers = |b: &mut ModuleBuilder| {
        let dev = b.ptr(AddrSpace::Device);
        let arr = b.array(dev, 2);
        vec![arr]
    };
    assert_eq!(
        decoded(&array_of_pointers, &[0], true).unwrap(),
        [AddrSpace::Device]
    );
    let array_of_constants = |b: &mut ModuleBuilder| {
        let cst = b.ptr(AddrSpace::Constant);
        let arr = b.array(cst, 2);
        vec![arr]
    };
    assert_eq!(
        decoded(&array_of_constants, &[0], true).unwrap(),
        [AddrSpace::Constant]
    );

    // A leading padding field does not shift the answer: the member at offset 8 is the pointer
    // field there, not the second field in declaration order.
    let padded = |b: &mut ModuleBuilder| {
        let i8t = b.int(8);
        let pad = b.array(i8t, 8);
        let cst = b.ptr(AddrSpace::Constant);
        vec![pad, cst]
    };
    assert_eq!(decoded(&padded, &[8], true).unwrap(), [AddrSpace::Constant]);

    // A field that is not a pointer contradicts the member's buffer role, so the space is not
    // guessed: `air.address_space` is `BadValue`.
    let not_a_pointer = |b: &mut ModuleBuilder| {
        let i32t = b.int(32);
        let v4 = b.vector(i32t, 4);
        vec![v4]
    };
    expect_abi(decoded(&not_a_pointer, &[0], true), AbiError::BadValue);
    // A pointer into a space no buffer can live in is refused by the same arm that refuses a
    // stated one (`air.address_space` of a thread-space pointer).
    let thread_pointer = |b: &mut ModuleBuilder| {
        let thr = b.ptr(AddrSpace::Thread);
        vec![thr]
    };
    expect_abi(decoded(&thread_pointer, &[0], true), AbiError::BadValue);
    // No field at the member's offset is no statement at all, so the default stands.
    assert_eq!(
        decoded(&two_pointers, &[4], true).unwrap(),
        [AddrSpace::Device]
    );
}

/// A role belongs to the stages that define it: a stage input, varying, builtin, or output of
/// another stage is inconsistent ABI.
#[test]
fn roles_belong_to_their_stage() {
    let varying: &[O<'_>] = &[I(0), S("air.fragment_input"), S("user(v)"), S("air.flat")];
    let color: &[O<'_>] = &[I(0), S("air.render_target"), I(0)];
    let rt: &[O<'_>] = &[S("air.render_target"), I(0), I(0)];
    let position: &[O<'_>] = &[S("air.position")];
    let refused: [(&str, &[O<'_>], Option<&[O<'_>]>); 7] = [
        ("air.kernel", varying, None),
        ("air.kernel", &[I(0), S("air.vertex_id")], None),
        ("air.kernel", color, None),
        (
            "air.fragment",
            &[I(0), S("air.thread_position_in_grid")],
            Some(rt),
        ),
        (
            "air.vertex",
            &[I(0), S("air.position_in_patch")],
            Some(position),
        ),
        ("air.vertex", &[I(0), S("air.vertex_id")], Some(rt)),
        (
            "air.fragment",
            &[I(0), S("air.front_facing")],
            Some(position),
        ),
    ];
    for (list, input, output) in refused {
        expect_abi(single_io(list, input, output), AbiError::UnknownKey);
    }
    let accepted: [(&str, &[O<'_>], Option<&[O<'_>]>); 6] = [
        (
            "air.kernel",
            &[I(0), S("air.thread_position_in_grid")],
            None,
        ),
        (
            "air.kernel",
            &[I(0), S("air.thread_index_in_simdgroup")],
            None,
        ),
        (
            "air.fragment",
            &[I(0), S("air.thread_index_in_simdgroup")],
            Some(rt),
        ),
        ("air.fragment", varying, Some(rt)),
        ("air.fragment", color, Some(rt)),
        ("air.vertex", &[I(0), S("air.vertex_id")], Some(position)),
    ];
    for (list, input, output) in accepted {
        assert!(single_io(list, input, output).is_ok(), "{list}");
    }
}

/// `[[base_vertex]]` and `[[base_instance]]` are a vertex function's builtins, as Apple's compiler
/// allows them: both in a vertex function, `[[base_instance]]` alone after tessellation, neither
/// in a kernel or fragment function.
#[test]
fn draw_bases_belong_to_vertex_functions() {
    let position: &[O<'_>] = &[S("air.position")];
    let rt: &[O<'_>] = &[S("air.render_target"), I(0), I(0)];
    for (key, builtin) in [
        ("air.base_vertex", Builtin::BaseVertex),
        ("air.base_instance", Builtin::BaseInstance),
    ] {
        let abi = single_io("air.vertex", &[I(0), S(key)], Some(position))
            .unwrap_or_else(|e| panic!("{key}: {e:?}"));
        assert_eq!(abi.entries[0].params[0].role, ParamRole::Builtin(builtin));
        expect_abi(
            single_io("air.kernel", &[I(0), S(key)], None),
            AbiError::UnknownKey,
        );
        expect_abi(
            single_io("air.fragment", &[I(0), S(key)], Some(rt)),
            AbiError::UnknownKey,
        );
    }
    let after_tessellation = |key: &'static str| {
        module_with(|b| {
            let i32t = b.int(32);
            let void = b.void();
            let f = entry_fn(b, "post", void, &[i32t]);
            let arg = node(b, &[I(0), S(key)]);
            let ins = node(b, &[N(arg)]);
            let outs = node(b, &[]);
            let patch = node(
                b,
                &[
                    S("air.patch"),
                    S("triangle"),
                    S("air.patch_control_point"),
                    I(3),
                ],
            );
            let e = node(b, &[F(f), N(outs), N(ins), N(patch)]);
            b.named_md("air.vertex", &[e]);
        })
    };
    let abi = after_tessellation("air.base_instance").expect("base_instance after tessellation");
    assert_eq!(
        abi.entries[0].params[0].role,
        ParamRole::Builtin(Builtin::BaseInstance)
    );
    expect_abi(after_tessellation("air.base_vertex"), AbiError::UnknownKey);
}

/// AIR 2.2 (macOS 13's compiler) follows a vertex input's location with its attribute's user key,
/// `generated(__air_placeholder__)` for an attribute it does not name. A vertex input binds by its
/// location, so the key is read and dropped; a string anywhere else, or a second one, stays an
/// unknown key, as does that string after any other node's location.
#[test]
fn a_vertex_input_s_location_may_carry_its_user_key() {
    let position: &[O<'_>] = &[S("air.position")];
    let input = |key: Option<&'static str>, more: &[O<'static>]| {
        let mut ops = vec![
            I(0),
            S("air.vertex_input"),
            S("air.location_index"),
            I(3),
            I(1),
        ];
        ops.extend(key.map(S));
        ops.extend([S("air.arg_type_name"), S("int"), S("air.arg_name"), S("a")]);
        ops.extend_from_slice(more);
        ops
    };
    let unkeyed = single_io("air.vertex", &input(None, &[]), Some(position)).unwrap();
    for key in ["generated(__air_placeholder__)", "generated(1aDv2_f)"] {
        let keyed = single_io("air.vertex", &input(Some(key), &[]), Some(position)).unwrap();
        assert_eq!(keyed, unkeyed, "{key}");
        assert_eq!(
            keyed.entries[0].params[0].role,
            ParamRole::VertexInput {
                location: Slot {
                    index: AbiInt::Const(3),
                    count: AbiInt::Const(1),
                }
            }
        );
    }
    let key = S("generated(__air_placeholder__)");
    let refused: [&[O<'_>]; 4] = [
        // A second string after the location.
        &[
            I(0),
            S("air.vertex_input"),
            S("air.location_index"),
            I(3),
            I(1),
            key,
            key,
        ],
        // The string after another attribute.
        &input(None, &[key]),
        // The string after a stage-in member's or a buffer's location.
        &[
            I(0),
            S("air.stage_in"),
            S("air.location_index"),
            I(3),
            I(1),
            key,
        ],
        &[
            I(0),
            S("air.buffer"),
            S("air.location_index"),
            I(3),
            I(1),
            key,
            S("air.read"),
        ],
    ];
    for ops in refused {
        expect_abi(
            single_io("air.vertex", ops, Some(position)),
            AbiError::UnknownKey,
        );
    }
}

/// A fragment input states its interpolation in full at every AIR version, so a sampling key
/// without a perspective key is incomplete there and refused. `air.centroid` alone is the
/// spelling AIR 1.6 to 2.1 write on an `air.vertex_output`
/// (`a_vertex_output_accepts_the_interpolation_keys_older_air_states_on_it`): accepting it there
/// must not make it readable here, where the interpolation is what the rasterizer is set from.
#[test]
fn interpolation_must_be_complete() {
    let fragment_input = |keys: &[&'static str]| {
        let mut b = ModuleBuilder::new();
        let f32t = b.float();
        let v4 = b.vector(f32t, 4);
        let void = b.void();
        let fs = entry_fn(&mut b, "fs", void, &[v4]);
        let mut ops = vec![I(0), S("air.fragment_input"), S("user(locn0)")];
        ops.extend(keys.iter().map(|&k| S(k)));
        let arg = node(&mut b, &ops);
        let ins = node(&mut b, &[N(arg)]);
        let outs = node(&mut b, &[]);
        let e = node(&mut b, &[F(fs), N(outs), N(ins)]);
        b.named_md("air.fragment", &[e]);
        let m = b.finish().unwrap();
        decode(&m, &PointeeTable::default(), &mut meter())
    };
    for keys in [
        &["air.center"][..],
        &["air.centroid"],
        &["air.sample"],
        &["air.perspective"],
        &["air.no_perspective"],
    ] {
        expect_abi(fragment_input(keys), AbiError::BadValue);
    }
    // The complete pairs the emitter writes on a fragment input do read, at every version.
    for (keys, want) in [
        (
            &["air.centroid", "air.perspective"][..],
            Interpolation::Smooth {
                sampling: Sampling::Centroid,
                perspective: true,
            },
        ),
        (
            &["air.center", "air.no_perspective"],
            Interpolation::Smooth {
                sampling: Sampling::Center,
                perspective: false,
            },
        ),
        (
            &["air.sample", "air.perspective"],
            Interpolation::Smooth {
                sampling: Sampling::Sample,
                perspective: true,
            },
        ),
        (&["air.flat"], Interpolation::Flat),
    ] {
        let abi = fragment_input(keys).unwrap_or_else(|e| panic!("{keys:?}: {e:?}"));
        let ParamRole::FragmentInput(v) = &abi.entries[0].params[0].role else {
            panic!("{keys:?}: a fragment input")
        };
        assert_eq!(v.interpolation, want, "{keys:?}");
    }
}

// ---- Struct layouts ----

#[test]
fn struct_layouts_decode_once_and_nest() {
    let mut b = ModuleBuilder::new();
    let params = kernel_params(&mut b);
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &params);
    let inner = node(
        &mut b,
        &[
            I(0),
            I(4),
            I(0),
            S("uint"),
            S("a"),
            I(4),
            I(4),
            I(0),
            S("uint"),
            S("b"),
        ],
    );
    let tex_arg = node(
        &mut b,
        &[
            I(1),
            S("air.texture"),
            S("air.location_index"),
            I(1),
            I(1),
            S("air.read"),
        ],
    );
    let outer = node(
        &mut b,
        &[
            S("air.struct_type_info"),
            N(inner),
            I(0),
            I(8),
            I(0),
            S("Inner"),
            S("x"),
            S("air.indirect_argument"),
            I(0),
            S("air.struct_type_info"),
            N(inner),
            I(8),
            I(8),
            I(2),
            S("Inner"),
            S("y"),
            I(24),
            I(8),
            I(0),
            S("texture2d<float>"),
            S("t"),
            S("air.indirect_argument"),
            N(tex_arg),
        ],
    );
    let mut buf = vec![
        I(0),
        S("air.indirect_buffer"),
        S("air.buffer_size"),
        I(32),
        S("air.location_index"),
        I(0),
        I(1),
        S("air.read"),
        S("air.address_space"),
        I(1),
        S("air.struct_type_info"),
        N(outer),
    ];
    buf.extend([S("air.arg_type_size"), I(32)]);
    let args = [buf, texture_arg(1), sampler_arg(2), gid_arg(3)];
    let arg_nodes: Vec<_> = args.iter().map(|a| N(node(&mut b, a))).collect();
    let inputs = node(&mut b, &arg_nodes);
    let outputs = node(&mut b, &[]);
    let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
    b.named_md("air.kernel", &[e]);
    let m = b.finish().unwrap();
    let abi = decode(&m, &PointeeTable::default(), &mut meter()).unwrap();
    // `inner` is decoded once and shared.
    assert_eq!(abi.structs.len(), 2);
    let ParamRole::ArgumentBuffer(arg) = &abi.entries[0].params[0].role else {
        panic!("argument buffer")
    };
    let outer = abi.struct_info(arg.layout.unwrap()).unwrap();
    assert_eq!(outer.members.len(), 3);
    assert_eq!(outer.members[0].layout, outer.members[1].layout);
    assert_eq!(
        outer.members[0].kind,
        MemberKind::IndirectStruct { first_id: 0 }
    );
    assert_eq!(outer.members[1].array_len, 2);
    let MemberKind::Indirect(t) = &outer.members[2].kind else {
        panic!("indirect texture")
    };
    assert_eq!(t.ordinal, 1);
    assert_eq!(
        outer.members[2].handle(),
        Some(AbiHandle::Texture(Access::Read))
    );
    let inner = abi.struct_info(outer.members[0].layout.unwrap()).unwrap();
    assert_eq!(
        inner
            .members
            .iter()
            .map(|m| (m.offset, m.size))
            .collect::<Vec<_>>(),
        vec![(0, 4), (4, 4)]
    );
}

/// An argument-buffer member's node leads with its ordinal in its layout, not its Metal `[[id]]`:
/// the id is the enclosing structs' first ids plus the role's own `air.location_index` (the
/// finding; the shape of corpus row 6405a9f3, whose `colorTexArray` leads with 7 and binds at 9).
#[test]
fn argument_buffer_member_ordinals_are_not_metal_ids() {
    let mut b = ModuleBuilder::new();
    let params = kernel_params(&mut b);
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &params);
    let constant = |b: &mut ModuleBuilder, ordinal, id| {
        node(
            b,
            &[
                I(ordinal),
                S("air.indirect_constant"),
                S("air.location_index"),
                I(id),
                I(1),
            ],
        )
    };
    // `struct Bounds { ushort2 min [[id(0)]]; ushort2 max [[id(1)]]; }`
    let (min, max) = (constant(&mut b, 0, 0), constant(&mut b, 1, 1));
    let bounds = node(
        &mut b,
        &[
            I(0),
            I(4),
            I(0),
            S("ushort2"),
            S("min"),
            S("air.indirect_argument"),
            N(min),
            I(4),
            I(4),
            I(0),
            S("ushort2"),
            S("max"),
            S("air.indirect_argument"),
            N(max),
        ],
    );
    // `struct Args { uint mode [[id(0)]]; uint blend [[id(1)]]; Bounds bounds [[id(2)]];
    //                texture2d<half, read_write> tex [[id(9)]]; }`
    let (mode, blend) = (constant(&mut b, 0, 0), constant(&mut b, 1, 1));
    let tex = node(
        &mut b,
        &[
            I(3),
            S("air.texture"),
            S("air.location_index"),
            I(9),
            I(1),
            S("air.read_write"),
        ],
    );
    let args = node(
        &mut b,
        &[
            I(0),
            I(4),
            I(0),
            S("uint"),
            S("mode"),
            S("air.indirect_argument"),
            N(mode),
            I(4),
            I(4),
            I(0),
            S("uint"),
            S("blend"),
            S("air.indirect_argument"),
            N(blend),
            S("air.struct_type_info"),
            N(bounds),
            I(8),
            I(8),
            I(0),
            S("Bounds"),
            S("bounds"),
            S("air.indirect_argument"),
            I(2),
            I(16),
            I(8),
            I(0),
            S("texture2d<half, read_write>"),
            S("tex"),
            S("air.indirect_argument"),
            N(tex),
        ],
    );
    let buf = [
        I(0),
        S("air.indirect_buffer"),
        S("air.location_index"),
        I(0),
        I(1),
        S("air.read"),
        S("air.address_space"),
        I(1),
        S("air.struct_type_info"),
        N(args),
    ];
    let arg_nodes = [
        N(node(&mut b, &buf)),
        N(node(&mut b, &texture_arg(1))),
        N(node(&mut b, &sampler_arg(2))),
        N(node(&mut b, &gid_arg(3))),
    ];
    let inputs = node(&mut b, &arg_nodes);
    let outputs = node(&mut b, &[]);
    let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
    b.named_md("air.kernel", &[e]);
    let m = b.finish().unwrap();
    let abi = decode(&m, &PointeeTable::default(), &mut meter()).unwrap();

    // Every argument member as (name, ordinal, Metal `[[id]]`), walking the layout with the
    // `first_id` of each enclosing struct added to the role's own location index.
    let ParamRole::ArgumentBuffer(arg) = &abi.entries[0].params[0].role else {
        panic!("argument buffer")
    };
    let mut found = Vec::new();
    let mut stack = vec![(arg.layout.expect("a layout"), 0)];
    while let Some((layout, base)) = stack.pop() {
        for (position, member) in abi.struct_info(layout).unwrap().members.iter().enumerate() {
            match &member.kind {
                MemberKind::Indirect(a) => {
                    let (ParamRole::IndirectConstant { location }
                    | ParamRole::Texture(TextureArg { location, .. })) = &a.role
                    else {
                        panic!("{:?}", a.role)
                    };
                    let AbiInt::Const(index) = location.index else {
                        panic!("a literal index")
                    };
                    // The ordinal is the member's position in its layout.
                    assert_eq!(a.ordinal as usize, position);
                    found.push((member.name.to_string(), a.ordinal, base + index));
                }
                MemberKind::IndirectStruct { first_id } => {
                    stack.push((member.layout.unwrap(), base + first_id));
                }
                MemberKind::Plain => panic!("{member:?}"),
            }
        }
    }
    found.sort_by_key(|&(_, _, id)| id);
    let found: Vec<_> = found
        .iter()
        .map(|(n, o, id)| (n.as_str(), *o, *id))
        .collect();
    assert_eq!(
        found,
        [
            ("mode", 0, 0),
            ("blend", 1, 1),
            ("min", 0, 2),
            ("max", 1, 3),
            ("tex", 3, 9),
        ]
    );
}

/// A role that requires one of several access keys names all of them when none is given.
#[test]
fn a_missing_access_names_the_keys_the_role_takes() {
    let alternatives = |spelled: &str| -> Vec<Key> {
        spelled
            .split('|')
            .map(|k| Key::parse(k.as_bytes()).expect("a key"))
            .collect()
    };
    assert_eq!(
        alternatives(TEXTURE_ACCESS),
        [Key::Sample, Key::Read, Key::Write, Key::ReadWrite]
    );
    assert_eq!(
        alternatives(BUFFER_ACCESS),
        [Key::Read, Key::Write, Key::ReadWrite]
    );
    let at = |r: Result<AirAbi, Error>| match r.expect_err("decode must fail") {
        Error::InvalidInput(i) => {
            assert_eq!(i.kind, InputErrorKind::Abi(AbiError::MissingKey));
            i.at
        }
        other => panic!("not an input error: {other:?}"),
    };
    let here = |entry: &str, arg, key| {
        Location::metadata(Some(DiagName::new(entry)), Some(arg), Some(key))
    };

    let texture = vec![I(1), S("air.texture"), S("air.location_index"), I(2), I(1)];
    let k = Kernel::with(&[buffer_arg(0), texture, sampler_arg(2), gid_arg(3)]);
    assert_eq!(at(k.decode()), here("k", 1, TEXTURE_ACCESS));
    let buffer: Vec<_> = buffer_arg(0)
        .into_iter()
        .filter(|o| !matches!(o, S("air.read_write")))
        .collect();
    let k = Kernel::with(&[buffer, texture_arg(1), sampler_arg(2), gid_arg(3)]);
    assert_eq!(at(k.decode()), here("k", 0, BUFFER_ACCESS));

    let texture = ci_with_input(
        0,
        &[I(0), S("air.ci_texture"), S("air.location_count"), I(1)],
    );
    assert_eq!(at(texture), here("k", 0, TEXTURE_ACCESS));
    let pointer = ci_with_input(
        2,
        &[
            I(2),
            S("air.ci_pointer"),
            S("air.arg_type_size"),
            I(64),
            S("air.arg_type_align_size"),
            I(16),
        ],
    );
    assert_eq!(at(pointer), here("k", 2, BUFFER_ACCESS));
}

#[test]
fn a_layout_containing_itself_is_refused() {
    let k = Kernel::with(&[
        vec![
            I(0),
            S("air.buffer"),
            S("air.location_index"),
            I(0),
            I(1),
            S("air.read"),
            S("air.address_space"),
            I(1),
            S("air.struct_type_info"),
            S("placeholder"),
        ],
        texture_arg(1),
        sampler_arg(2),
        gid_arg(3),
    ]);
    // Make the `air.struct_type_info` operand a node that names itself as a nested layout.
    let mut parts = k.b.finish_parts().unwrap();
    let mut me = meter();
    let zero = parts
        .consts
        .iter()
        .find(|(_, c)| c.value == Const::Int(0) && parts.types.get(c.ty) == Some(&Type::Int(32)))
        .map(|(id, _)| id)
        .unwrap();
    let zero = parts.metadata.push(Md::Value(zero), &mut me).unwrap();
    let key = parts
        .symbols
        .intern(b"air.struct_type_info", &mut me)
        .unwrap();
    let key = parts.metadata.push(Md::String(key), &mut me).unwrap();
    let name = parts.symbols.intern(b"n", &mut me).unwrap();
    let name = parts.metadata.push(Md::String(name), &mut me).unwrap();
    let cyclic = MdId::new(u32::try_from(parts.metadata.len()).unwrap());
    let ops = [key, cyclic, zero, zero, zero, name, name].map(Some);
    let got = parts
        .metadata
        .push(
            Md::Node {
                distinct: true,
                ops: ops.into(),
            },
            &mut me,
        )
        .unwrap();
    assert_eq!(got, cyclic);
    // Point the buffer argument's `air.struct_type_info` at it.
    let arg = parts
        .metadata
        .iter()
        .find(|(_, md)| matches!(md, Md::Node { ops, .. } if ops.len() == 10))
        .map(|(id, _)| id)
        .unwrap();
    if let Some(Md::Node { ops, .. }) = parts.metadata.get_mut(arg) {
        ops[9] = Some(cyclic);
    }
    let m = Module::new(parts, &mut meter()).unwrap();
    expect_abi(
        decode(&m, &PointeeTable::default(), &mut meter()),
        AbiError::BadValue,
    );
}

/// A kernel whose two device-buffer parameters carry the layouts `layouts` builds: an argument
/// buffer (its layout may hold argument-buffer members) and a buffer.
fn layout_kernel(layouts: impl FnOnce(&mut ModuleBuilder) -> [MdId; 2]) -> Result<AirAbi, Error> {
    decode(
        &layout_module(layouts),
        &PointeeTable::default(),
        &mut meter(),
    )
}

/// The module of [`layout_kernel`].
fn layout_module(layouts: impl FnOnce(&mut ModuleBuilder) -> [MdId; 2]) -> Module {
    let mut b = ModuleBuilder::new();
    let params = kernel_params(&mut b);
    let void = b.void();
    let f = entry_fn(&mut b, "k", void, &params);
    let [l0, l1] = layouts(&mut b);
    let buffer = |i: u64, layout: MdId| {
        vec![
            I(i),
            S(if i == 0 {
                "air.indirect_buffer"
            } else {
                "air.buffer"
            }),
            S("air.location_index"),
            I(i),
            I(1),
            S("air.read"),
            S("air.address_space"),
            I(1),
            S("air.struct_type_info"),
            N(layout),
        ]
    };
    let args = [buffer(0, l0), buffer(1, l1), sampler_arg(2), gid_arg(3)];
    let arg_nodes: Vec<_> = args.iter().map(|a| N(node(&mut b, a))).collect();
    let inputs = node(&mut b, &arg_nodes);
    let outputs = node(&mut b, &[]);
    let e = node(&mut b, &[F(f), N(outputs), N(inputs)]);
    b.named_md("air.kernel", &[e]);
    b.finish().unwrap()
}

/// A layout with one plain `uint` member.
fn leaf_layout(b: &mut ModuleBuilder) -> MdId {
    node(b, &[I(0), I(4), I(0), S("uint"), S("a")])
}

/// A layout whose one member is the nested layout `inner`.
fn wrap_layout(b: &mut ModuleBuilder, inner: MdId) -> MdId {
    node(
        b,
        &[
            S("air.struct_type_info"),
            N(inner),
            I(0),
            I(4),
            I(0),
            S("S"),
            S("s"),
        ],
    )
}

/// A layout whose one member is an argument-buffer binding with layout `inner`.
fn indirect_layout(b: &mut ModuleBuilder, inner: MdId) -> MdId {
    indirect_member_layout(b, "air.indirect_buffer", inner)
}

/// A layout whose one member is a binding of `role` (`air.buffer` or `air.indirect_buffer`) with
/// layout `inner`.
fn indirect_member_layout(b: &mut ModuleBuilder, role: &str, inner: MdId) -> MdId {
    let arg = node(
        b,
        &[
            I(0),
            S(role),
            S("air.location_index"),
            I(0),
            I(1),
            S("air.read"),
            S("air.address_space"),
            I(1),
            S("air.struct_type_info"),
            N(inner),
        ],
    );
    node(
        b,
        &[
            I(0),
            I(8),
            I(0),
            S("device S*"),
            S("p"),
            S("air.indirect_argument"),
            N(arg),
        ],
    )
}

fn layout_chain(
    b: &mut ModuleBuilder,
    levels: u64,
    wrap: fn(&mut ModuleBuilder, MdId) -> MdId,
) -> MdId {
    let mut top = leaf_layout(b);
    for _ in 1..levels {
        top = wrap(b, top);
    }
    top
}

/// `air.indirect_argument` is an argument buffer's member key. Metal writes it only in an
/// `air.indirect_buffer`'s layout (a struct with a pointer or resource member, at any depth, makes
/// its buffer one), so an `air.buffer` whose layout holds one is `UnknownKey` naming it: directly,
/// nested by value, or as an argument buffer's `air.buffer` member. Plain layouts, and argument
/// buffers holding such members, decode.
#[test]
fn an_air_buffer_layout_holding_an_argument_buffer_member_is_refused() {
    let refused = |r: Result<AirAbi, Error>, arg| match r.expect_err("decode must fail") {
        Error::InvalidInput(i) => {
            assert_eq!(i.kind, InputErrorKind::Abi(AbiError::UnknownKey));
            if let Some(arg) = arg {
                assert_eq!(
                    i.at,
                    Location::metadata(
                        Some(DiagName::new("k")),
                        Some(arg),
                        Some("air.indirect_argument")
                    )
                );
            }
        }
        other => panic!("not an input error: {other:?}"),
    };
    // The second parameter, an `air.buffer`: a pointer member, and one nested by value.
    refused(
        layout_kernel(|b| {
            let leaf = leaf_layout(b);
            [leaf, indirect_layout(b, leaf)]
        }),
        Some(1),
    );
    refused(
        layout_kernel(|b| {
            let leaf = leaf_layout(b);
            let pointer = indirect_layout(b, leaf);
            [leaf, wrap_layout(b, pointer)]
        }),
        Some(1),
    );
    // The first parameter, an argument buffer, whose `air.buffer` member's layout holds one.
    refused(
        layout_kernel(|b| {
            let leaf = leaf_layout(b);
            let pointer = indirect_layout(b, leaf);
            [
                indirect_member_layout(b, "air.buffer", pointer),
                leaf_layout(b),
            ]
        }),
        None,
    );
    // An argument buffer holding argument-buffer members (nested by value too), its `air.buffer`
    // member over a plain layout, and an `air.buffer` over nested plain layouts all decode.
    for at_depth in [false, true] {
        let abi = layout_kernel(|b| {
            let leaf = leaf_layout(b);
            let pointer = indirect_layout(b, leaf);
            let held = if at_depth {
                wrap_layout(b, pointer)
            } else {
                pointer
            };
            let plain = wrap_layout(b, leaf);
            [held, wrap_layout(b, plain)]
        });
        assert!(abi.is_ok(), "{abi:?}");
    }
    let abi = layout_kernel(|b| {
        let leaf = leaf_layout(b);
        [indirect_member_layout(b, "air.buffer", leaf), leaf]
    });
    assert!(abi.is_ok(), "{abi:?}");
}

/// The Metal front end names a member's nested layout with `air.struct_type_info, !n` BEFORE the
/// member's offset ([`wrap_layout`]). A layout that names it as a bare node AFTER the member's
/// name is not that grammar: the next member then starts on a node where its offset is due, so
/// the layout is `BadValue` at the buffer's `air.struct_type_info`, not a guess at what the node
/// meant. The hand-written public fixture `kernel_cloned_continuation_uniform_read.ll` used to
/// spell its nested `Coef` layouts this way.
#[test]
fn a_nested_layout_named_after_its_member_is_refused() {
    let trailing = |b: &mut ModuleBuilder| {
        let leaf = leaf_layout(b);
        let bare = node(
            b,
            &[
                I(0),
                I(4),
                I(0),
                S("S"),
                S("a"),
                N(leaf),
                I(4),
                I(4),
                I(0),
                S("S"),
                S("b"),
                N(leaf),
            ],
        );
        [leaf_layout(b), bare]
    };
    match layout_kernel(trailing).expect_err("decode must fail") {
        Error::InvalidInput(i) => {
            assert_eq!(i.kind, InputErrorKind::Abi(AbiError::BadValue));
            assert_eq!(
                i.at,
                Location::metadata(
                    Some(DiagName::new("k")),
                    Some(1),
                    Some("air.struct_type_info")
                )
            );
        }
        other => panic!("not an input error: {other:?}"),
    }
    // The same two members in Metal's spelling decode, sharing the one nested layout.
    let abi = layout_kernel(|b| {
        let leaf = leaf_layout(b);
        let metal = node(
            b,
            &[
                S("air.struct_type_info"),
                N(leaf),
                I(0),
                I(4),
                I(0),
                S("S"),
                S("a"),
                S("air.struct_type_info"),
                N(leaf),
                I(4),
                I(4),
                I(0),
                S("S"),
                S("b"),
            ],
        );
        [leaf_layout(b), metal]
    })
    .unwrap();
    let ParamRole::Buffer(buffer) = &abi.entries[0].params[1].role else {
        panic!("buffer")
    };
    let layout = abi.struct_info(buffer.layout.unwrap()).unwrap();
    assert_eq!(layout.members.len(), 2);
    assert!(layout.members[0].layout.is_some());
    assert_eq!(layout.members[0].layout, layout.members[1].layout);
}

/// Layout nesting is capped (the reader never overflows its stack). Untrusted metadata can chain
/// layouts, directly or through argument-buffer members, arbitrarily deep.
#[test]
fn deeply_nested_layouts_are_refused_without_recursing() {
    let cap = Limit::LayoutNesting.fixed_cap().unwrap();
    for wrap in [wrap_layout, indirect_layout] {
        // 100k levels, a few bytes of metadata each.
        let deep = layout_kernel(|b| {
            let top = layout_chain(b, 100_000, wrap);
            [top, leaf_layout(b)]
        });
        assert_eq!(deep, Err(Error::LimitExceeded(Limit::LayoutNesting)));
        // Exactly at the cap decodes.
        let abi = layout_kernel(|b| {
            let top = layout_chain(b, cap, wrap);
            [top, leaf_layout(b)]
        })
        .unwrap();
        assert_eq!(abi.structs.len() as u64, cap + 1);
        assert_eq!(
            layout_kernel(|b| {
                let top = layout_chain(b, cap + 1, wrap);
                [top, leaf_layout(b)]
            }),
            Err(Error::LimitExceeded(Limit::LayoutNesting))
        );
    }
    // A layout shared by reference counts its full height: one at the cap, decoded first, then
    // wrapped once more by a later parameter's layout, is one past the cap.
    assert_eq!(
        layout_kernel(|b| {
            let top = layout_chain(b, cap, wrap_layout);
            [top, wrap_layout(b, top)]
        }),
        Err(Error::LimitExceeded(Limit::LayoutNesting))
    );
}

/// The layout decoder's stack use does not grow with nesting: a chain at the cap, directly or
/// through argument-buffer members, decodes (and one level more is refused) on a thread with a
/// small explicit stack, in debug builds too, well below the default 2 MiB test-thread stack.
#[test]
fn layouts_at_the_nesting_cap_decode_on_a_small_stack() {
    const STACK: usize = 256 * 1024;
    let cap = Limit::LayoutNesting.fixed_cap().unwrap();
    for wrap in [wrap_layout, indirect_layout] {
        let chain = |levels: u64| {
            layout_module(|b| {
                let top = layout_chain(b, levels, wrap);
                [top, leaf_layout(b)]
            })
        };
        let (at_cap, past_cap) = (chain(cap), chain(cap + 1));
        let decoded = std::thread::scope(|s| {
            std::thread::Builder::new()
                .stack_size(STACK)
                .spawn_scoped(s, || {
                    let structs = |m: &Module| {
                        decode(m, &PointeeTable::default(), &mut meter()).map(|a| a.structs.len())
                    };
                    (structs(&at_cap), structs(&past_cap))
                })
                .unwrap()
                .join()
                .unwrap()
        });
        assert_eq!(
            decoded,
            (
                Ok(cap as usize + 1),
                Err(Error::LimitExceeded(Limit::LayoutNesting))
            )
        );
    }
}

/// A pointer parameter's size and layout against its pointee.
#[test]
fn pointee_size_and_layout_checks() {
    let mut b = ModuleBuilder::new();
    let f32t = b.float();
    let i32t = b.int(32);
    let st = b.struct_ty(&[f32t, i32t]);
    let arr = b.array(i32t, 4);
    let m = b.finish().unwrap();
    let mut me = meter();
    let layouts = TypeLayouts::new(m.data_layout(), m.types(), &mut me).unwrap();
    let site = Site::module();
    let member = |offset, size, array_len, layout| Member {
        offset,
        size,
        array_len,
        layout,
        kind: MemberKind::Plain,
        render_target: None,
        raster_order_group: None,
        type_name: DiagName::new("t"),
        name: DiagName::new("n"),
        key: UserKey(b"n".as_slice().into()),
    };
    let abi = |members: Vec<Member>| AirAbi {
        air_version: None,
        language: None,
        compile: CompileOptions::default(),
        limits: ModuleLimits::default(),
        entries: vec![],
        function_constants: vec![],
        static_samplers: vec![],
        visible_function_refs: vec![],
        visible_functions: vec![],
        core_image_kernels: vec![],
        linkage_signature: None,
        structs: vec![StructInfo { members }],
        producer: None,
    };
    let param = |size, align, buffer_size, layout| ParamBinding {
        param: 0,
        role: ParamRole::Buffer(BufferArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(1),
            },
            access: Access::Read,
            space: AddrSpace::Device,
            buffer_size,
            layout,
            raster_order_group: None,
        }),
        unused: false,
        predicate: None,
        ty: ArgTypeInfo {
            size,
            align,
            name: None,
        },
        name: None,
    };
    let check = |abi: &AirAbi, p: &ParamBinding, pointee| {
        pointee_check(abi, &layouts, p, pointee, &site, &mut meter()).map_err(abi_kind)
    };
    let fields = abi(vec![member(0, 4, 0, None), member(4, 4, 0, None)]);
    let s0 = Some(StructRef(0));
    assert_eq!(
        check(&fields, &param(Some(8), Some(4), Some(8), s0), st),
        Ok(())
    );
    // Size disagrees with the pointee.
    assert_eq!(
        check(&fields, &param(Some(12), Some(4), None, s0), st),
        Err(AbiError::TypeSizeMismatch)
    );
    // Alignment is not a power of two.
    assert_eq!(
        check(&fields, &param(Some(8), Some(3), None, s0), st),
        Err(AbiError::TypeSizeMismatch)
    );
    // A reference to an array: the buffer size is a multiple of the element.
    assert_eq!(
        check(&fields, &param(Some(8), None, Some(24), s0), st),
        Ok(())
    );
    assert_eq!(
        check(&fields, &param(Some(8), None, Some(20), s0), st),
        Err(AbiError::TypeSizeMismatch)
    );
    // A member starting past the end.
    let past = abi(vec![member(8, 4, 0, None)]);
    assert_eq!(
        check(&past, &param(Some(8), None, None, s0), st),
        Err(AbiError::StructInfoMismatch)
    );
    // An array member extending past the end.
    let long = abi(vec![member(4, 4, 2, None)]);
    assert_eq!(
        check(&long, &param(Some(8), None, None, s0), st),
        Err(AbiError::StructInfoMismatch)
    );
    // A bitfield: its storage type may extend past the end, its start may not.
    let bitfield = abi(vec![member(0, 4, 0, None), member(6, 4, 0, None)]);
    assert_eq!(
        check(&bitfield, &param(Some(8), None, None, s0), st),
        Ok(())
    );
    // An array pointee.
    let whole = abi(vec![member(0, 4, 4, None)]);
    assert_eq!(check(&whole, &param(Some(16), None, None, s0), arr), Ok(()));

    // The member walk follows nested layouts with an explicit stack: a chain far past the
    // nesting cap (decode refuses one, but the walk does not rely on that) is checked on a
    // thread with a small explicit stack, and its innermost member is still reached.
    let depth = 4096u32;
    let chain = |leaf: Member| {
        let mut structs: Vec<StructInfo> = (1..depth)
            .map(|i| StructInfo {
                members: vec![member(0, 8, 0, Some(StructRef(i)))],
            })
            .collect();
        structs.push(StructInfo {
            members: vec![leaf],
        });
        AirAbi {
            structs,
            ..abi(vec![])
        }
    };
    let (fits, past) = (chain(member(4, 4, 0, None)), chain(member(8, 4, 0, None)));
    let deep = std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn_scoped(s, || {
                (
                    check(&fits, &param(Some(8), None, None, s0), st),
                    check(&past, &param(Some(8), None, None, s0), st),
                )
            })
            .unwrap()
            .join()
            .unwrap()
    });
    assert_eq!(deep, (Ok(()), Err(AbiError::StructInfoMismatch)));
}

// ---- Module-level lists ----

fn module_with(f: impl FnOnce(&mut ModuleBuilder)) -> Result<AirAbi, Error> {
    let mut b = ModuleBuilder::new();
    f(&mut b);
    let m = b.finish().unwrap();
    decode(&m, &PointeeTable::default(), &mut meter())
}

#[test]
fn module_level_metadata_decodes() {
    let abi = module_with(|b| {
        let v = node(b, &[I(2), I(8), I(0)]);
        b.named_md("air.version", &[v]);
        let l = node(b, &[S("Metal"), I(3), I(1), I(0)]);
        b.named_md("air.language_version", &[l]);
        let o1 = node(b, &[S("air.compile.denorms_disable")]);
        let o2 = node(b, &[S("air.compile.fast_math_enable")]);
        let o3 = node(b, &[S("air.compile.framebuffer_fetch_disable")]);
        b.named_md("air.compile_options", &[o1, o2, o3]);
        let f1 = node(b, &[I(7), S("air.max_textures"), I(128)]);
        let f2 = node(b, &[I(1), S("wchar_size"), I(4)]);
        b.named_md("llvm.module.flags", &[f1, f2]);
        let id = node(b, &[S("producer")]);
        b.named_md("llvm.ident", &[id]);
    })
    .unwrap();
    assert_eq!(
        abi.air_version,
        Some(Version {
            major: 2,
            minor: 8,
            patch: 0
        })
    );
    assert_eq!(
        abi.language,
        Some(LanguageVersion(Version {
            major: 3,
            minor: 1,
            patch: 0
        }))
    );
    assert_eq!(
        abi.compile,
        CompileOptions {
            fast_math: Some(true),
            denorms_disable: true,
            framebuffer_fetch: Some(false),
        }
    );
    assert_eq!(abi.limits.max_textures, Some(128));
    assert_eq!(abi.limits.max_samplers, None);
    assert_eq!(abi.producer, Some(DiagName::new("producer")));
}

#[test]
fn closed_module_level_sets() {
    // Unknown named metadata.
    expect_abi(
        module_with(|b| {
            let n = node(b, &[]);
            b.named_md("air.not_a_list", &[n]);
        }),
        AbiError::UnknownKey,
    );
    // Unknown compile option.
    expect_abi(
        module_with(|b| {
            let n = node(b, &[S("air.compile.something_else")]);
            b.named_md("air.compile_options", &[n]);
        }),
        AbiError::UnknownKey,
    );
    // Contradictory compile options.
    expect_abi(
        module_with(|b| {
            let a = node(b, &[S("air.compile.fast_math_enable")]);
            let c = node(b, &[S("air.compile.fast_math_disable")]);
            b.named_md("air.compile_options", &[a, c]);
        }),
        AbiError::BadValue,
    );
    // Unknown module flag.
    expect_abi(
        module_with(|b| {
            let n = node(b, &[I(8), S("PIC Level"), I(2)]);
            b.named_md("llvm.module.flags", &[n]);
        }),
        AbiError::UnknownKey,
    );
    // A language other than Metal.
    expect_abi(
        module_with(|b| {
            let n = node(b, &[S("OpenCL"), I(1), I(2), I(0)]);
            b.named_md("air.language_version", &[n]);
        }),
        AbiError::BadValue,
    );
}

/// What `-gline-tables-only` and `-frecord-sources` add to a module (Xcode's
/// `MTL_ENABLE_DEBUG_INFO`): the compile units, the two debug-info module flags and, in an unlinked
/// `.air`, the command line, the working directory and the recorded sources. The module decodes to
/// the ABI it has without them.
#[test]
fn debug_info_and_recorded_sources_leave_the_abi_unchanged() {
    let textures = |b: &mut ModuleBuilder| node(b, &[I(7), S("air.max_textures"), I(128)]);
    let plain = module_with(|b| {
        let f = textures(b);
        b.named_md("llvm.module.flags", &[f]);
    })
    .unwrap();
    let debug = module_with(|b| {
        let dwarf = node(b, &[I(7), S("Dwarf Version"), I(4)]);
        let version = node(b, &[I(2), S("Debug Info Version"), I(3)]);
        let f = textures(b);
        b.named_md("llvm.module.flags", &[dwarf, version, f]);
        let unit = b.md(Md::Debug);
        b.named_md("llvm.dbg.cu", &[unit]);
        let line = node(
            b,
            &[S("metal -c -gline-tables-only -frecord-sources a.metal")],
        );
        b.named_md("llvm.commandline", &[line]);
        let dir = node(b, &[S("/src")]);
        b.named_md("llvm_utils.workingdir", &[dir]);
        let source = node(b, &[S("/src/a.metal"), S("#include \"a.h\"")]);
        let header = node(b, &[S("/src/a.h"), S("")]);
        let files = node(b, &[N(source), N(header)]);
        b.named_md("llvm_utils.sources", &[files]);
    })
    .unwrap();
    assert_eq!(debug, plain);
}

/// The debug-info and recorded-source lists are read by their shapes: a compile unit that is not a
/// debug-info record, a recorded source without its text, or a file that is not a node is refused.
#[test]
fn debug_info_and_recorded_sources_have_closed_shapes() {
    expect_abi(
        module_with(|b| {
            let n = node(b, &[S("not a compile unit")]);
            b.named_md("llvm.dbg.cu", &[n]);
        }),
        AbiError::BadValue,
    );
    expect_abi(
        module_with(|b| {
            let file = node(b, &[S("/src/a.metal")]);
            let files = node(b, &[N(file)]);
            b.named_md("llvm_utils.sources", &[files]);
        }),
        AbiError::MissingKey,
    );
    expect_abi(
        module_with(|b| {
            let files = node(b, &[S("/src/a.metal"), S("")]);
            b.named_md("llvm_utils.sources", &[files]);
        }),
        AbiError::BadValue,
    );
    expect_abi(
        module_with(|b| {
            let n = node(b, &[S("metal"), S("-c")]);
            b.named_md("llvm.commandline", &[n]);
        }),
        AbiError::UnknownKey,
    );
}

fn fc_global(b: &mut ModuleBuilder, name: &str, section: Option<Section>) -> GlobalId {
    let i16t = b.int(16);
    let undef = b.constant(i16t, Const::Undef);
    fc_global_with(b, name, section, Some(undef))
}

fn fc_global_with(
    b: &mut ModuleBuilder,
    name: &str,
    section: Option<Section>,
    init: Option<ConstId>,
) -> GlobalId {
    fc_global_as(b, name, section, init, AddrSpace::Constant, true, true)
}

/// A function-constant initializer global with each property the check reads under the caller's
/// control: its section, initializer, address space, `constant`, and `externally_initialized`.
fn fc_global_as(
    b: &mut ModuleBuilder,
    name: &str,
    section: Option<Section>,
    init: Option<ConstId>,
    space: AddrSpace,
    constant: bool,
    externally_initialized: bool,
) -> GlobalId {
    let i16t = b.int(16);
    let name = b.sym(name);
    b.global(Global {
        name,
        value_ty: i16t,
        space,
        constant,
        init,
        externally_initialized,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::External,
        align: None,
        section,
    })
}

#[test]
fn function_constants_decode_and_are_checked() {
    let mut g0 = None;
    let abi = module_with(|b| {
        let g = fc_global(b, "fc0", Some(Section::FcInitializer));
        let h = fc_global(b, "fc1", Some(Section::FcInitializer));
        g0 = Some(g);
        let a = node(b, &[G(g), S("short"), S("K"), I(0), B(true)]);
        let c = node(b, &[G(h), S("short"), S("L"), I(3)]);
        b.named_md("air.function_constants", &[a, c]);
    })
    .unwrap();
    let fc = &abi.function_constants;
    assert_eq!(fc.len(), 2);
    assert_eq!(fc[0].index, 0);
    assert_eq!(fc[0].global, g0.unwrap());
    assert_eq!(fc[0].flag, Some(true));
    assert_eq!(fc[1].flag, None);
    assert_eq!(fc[1].name, DiagName::new("L"));
    // Duplicate index.
    expect_abi(
        module_with(|b| {
            let g = fc_global(b, "fc0", Some(Section::FcInitializer));
            let h = fc_global(b, "fc1", Some(Section::FcInitializer));
            let a = node(b, &[G(g), S("short"), S("K"), I(0)]);
            let c = node(b, &[G(h), S("short"), S("L"), I(0)]);
            b.named_md("air.function_constants", &[a, c]);
        }),
        AbiError::BadValue,
    );
    // AIR 2.1 places the initializer global in no section (`air.fc_initializer` arrived with
    // AIR 2.2), and the record is what names it, so the section is not part of the test.
    let abi = module_with(|b| {
        let g = fc_global(b, "fc0", None);
        let a = node(b, &[G(g), S("short"), S("K"), I(0)]);
        b.named_md("air.function_constants", &[a]);
    })
    .unwrap();
    assert_eq!(abi.function_constants.len(), 1);
    assert_eq!(abi.function_constants[0].index, 0);
    // A global in another section is not one.
    expect_abi(
        module_with(|b| {
            let g = fc_global(b, "fc0", Some(Section::StaticInit));
            let a = node(b, &[G(g), S("short"), S("K"), I(0)]);
            b.named_md("air.function_constants", &[a]);
        }),
        AbiError::BadValue,
    );
    // Not an initializer global: one the host does not write, one that is not `constant` (which
    // the reading analysis takes for granted), and one outside the constant address space. Each
    // is refused in the section AIR 2.5 writes and in the earlier shape with none.
    for section in [Some(Section::FcInitializer), None] {
        let cases: [(AddrSpace, bool, bool); 3] = [
            (AddrSpace::Constant, true, false),
            (AddrSpace::Constant, false, true),
            (AddrSpace::Device, true, true),
        ];
        for (space, constant, external) in cases {
            expect_abi(
                module_with(|b| {
                    let i16t = b.int(16);
                    let undef = b.constant(i16t, Const::Undef);
                    let g = fc_global_as(b, "fc0", section, Some(undef), space, constant, external);
                    let a = node(b, &[G(g), S("short"), S("K"), I(0)]);
                    b.named_md("air.function_constants", &[a]);
                }),
                AbiError::BadValue,
            );
        }
    }
    // An initializer global whose initializer is not `undef`: a value, or none (a declaration).
    for init in [Some(Const::Int(7)), Some(Const::Null), None] {
        expect_abi(
            module_with(|b| {
                let i16t = b.int(16);
                let init = init.map(|v| b.constant(i16t, v));
                let g = fc_global_with(b, "fc0", Some(Section::FcInitializer), init);
                let a = node(b, &[G(g), S("short"), S("K"), I(0)]);
                b.named_md("air.function_constants", &[a]);
            }),
            AbiError::BadValue,
        );
    }
}

fn sampler_global(b: &mut ModuleBuilder, words: &[u64]) -> GlobalId {
    let i64t = b.int(64);
    let (ty, init) = match words {
        [w] => (i64t, b.constant(i64t, Const::Int(*w))),
        _ => {
            let elems: Vec<ConstId> = words
                .iter()
                .map(|w| b.constant(i64t, Const::Int(*w)))
                .collect();
            let arr = b.array(i64t, words.len() as u64);
            (arr, b.constant(arr, Const::Aggregate(elems.into())))
        }
    };
    let name = b.sym("__air_sampler_state");
    b.global(Global {
        name,
        value_ty: ty,
        space: AddrSpace::Constant,
        constant: true,
        init: Some(init),
        externally_initialized: false,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: None,
        section: None,
    })
}

#[test]
fn static_samplers_decode_from_their_globals() {
    let mut globals = vec![];
    let abi = module_with(|b| {
        // AIR 2.8: [2 x i64]; AIR 2.6: one i64 with bit 63 set.
        let a = sampler_global(b, &[0x007b_ff00_0008_0a49, 0x4000]);
        let c = sampler_global(b, &[0x807b_ff00_0008_0a49]);
        globals = vec![a, c];
        let na = node(b, &[S("air.sampler_state"), G(a)]);
        let nc = node(b, &[S("air.sampler_state"), G(c)]);
        b.named_md("air.sampler_states", &[na, nc]);
    })
    .unwrap();
    let [a, c] = &abi.static_samplers[..] else {
        panic!("two samplers")
    };
    assert_eq!((a.global, c.global), (globals[0], globals[1]));
    assert_eq!(a.state.mag_filter, Filter::Linear);
    assert_eq!(a.state.lod_bias.to_f32(), 2.0);
    assert_eq!(c.state.lod_bias.to_f32(), 0.0);
    assert_eq!(
        SamplerState {
            lod_bias: Half(0),
            ..a.state
        },
        c.state
    );
    // An AIR 2.7 module of one-word samplers says so in a module flag (an `i1`), which changes
    // nothing decoded.
    let flagged = |value: O<'static>| {
        module_with(move |b| {
            let v = node(b, &[I(2), I(7), I(0)]);
            b.named_md("air.version", &[v]);
            let f = node(
                b,
                &[
                    I(7),
                    S("air.workaround.sampler_state_variable_length_encoding"),
                    value,
                ],
            );
            b.named_md("llvm.module.flags", &[f]);
            let g = sampler_global(b, &[0x807b_ff00_0008_0a49]);
            let n = node(b, &[S("air.sampler_state"), G(g)]);
            b.named_md("air.sampler_states", &[n]);
        })
    };
    for value in [B(true), B(false)] {
        let abi = flagged(value).unwrap();
        assert_eq!(abi.static_samplers[0].state, c.state);
        assert_eq!(abi.limits, ModuleLimits::default());
    }
    expect_abi(flagged(I(1)), AbiError::BadValue);
    // An undecodable word.
    expect_abi(
        module_with(|b| {
            let g = sampler_global(b, &[0x007b_ff00_0008_0a49 | (1 << 60), 0]);
            let n = node(b, &[S("air.sampler_state"), G(g)]);
            b.named_md("air.sampler_states", &[n]);
        }),
        AbiError::BadSamplerState,
    );
    // A sampler state list entry that is not a sampler state.
    expect_abi(
        module_with(|b| {
            let g = sampler_global(b, &[0, 0]);
            let n = node(b, &[S("air.sampler"), G(g)]);
            b.named_md("air.sampler_states", &[n]);
        }),
        AbiError::UnknownKey,
    );
}

#[test]
fn visible_function_references_and_signatures() {
    let abi = module_with(|b| {
        let void = b.void();
        let ty = b.fn_ty(void, &[]);
        let f = b.declare("surface", ty);
        let r = node(
            b,
            &[S("air.visible_function_reference"), F(f), S("surface")],
        );
        b.named_md("air.visible_function_references", &[r]);
        let a = node(b, &[]);
        let s = node(b, &[S("air.linked_constants_signature"), N(a), N(a)]);
        b.named_md("air.signature", &[s]);
    })
    .unwrap();
    assert_eq!(abi.visible_function_refs.len(), 1);
    assert!(matches!(
        abi.visible_function_refs[0].target,
        VisibleTarget::Function(_)
    ));
    assert_eq!(
        abi.visible_function_refs[0].symbol,
        VisibleSymbol(b"surface".as_slice().into())
    );
    let stubs: Vec<_> = abi.visible_function_stubs().collect();
    assert!(matches!(
        stubs.as_slice(),
        [(_, VisibleSymbol(s))] if &**s == b"surface"
    ));
    assert_eq!(
        abi.linkage_signature,
        Some(LinkageSignature {
            kind: LinkageSignatureKind::LinkedConstants,
            operands: 2
        })
    );
}

/// A function renamed to its qualified name after it was produced (Apple's compiler at macOS 14)
/// wraps the signature that produced it, which is how the module is read. The wrapper takes exactly
/// a signature node and a name; a rename wrapped in a rename, the wrapper itself among them, is
/// unknown.
#[test]
fn a_renamed_function_s_signature_is_the_one_it_wraps() {
    let renamed = |shape: &dyn Fn(&mut ModuleBuilder, MdId) -> MdId| {
        module_with(|b| {
            let hash = node(b, &[S("air.module_signature"), S("0123ABCD")]);
            let c0 = node(b, &[I(0), S("bool"), I(6)]);
            let c1 = node(b, &[I(1), S("uchar"), I(7)]);
            let inner = node(
                b,
                &[S("air.linked_constants_signature"), N(hash), N(c0), N(c1)],
            );
            let s = shape(b, inner);
            b.named_md("air.signature", &[s]);
        })
    };
    let wrap = |b: &mut ModuleBuilder, inner: MdId| {
        node(
            b,
            &[
                S("air.rename_qualified_function_signature"),
                N(inner),
                S("qualified_name"),
            ],
        )
    };
    let read = |shape: &dyn Fn(&mut ModuleBuilder, MdId) -> MdId| {
        renamed(shape).map(|abi| abi.linkage_signature)
    };
    let linked = Some(LinkageSignature {
        kind: LinkageSignatureKind::LinkedConstants,
        operands: 3,
    });
    assert_eq!(read(&|_, inner| inner).unwrap(), linked);
    assert_eq!(read(&wrap).unwrap(), linked);
    // A rename of a rename.
    expect_abi(
        renamed(&|b, inner| {
            let once = wrap(b, inner);
            wrap(b, once)
        }),
        AbiError::UnknownKey,
    );
    // The wrapper without its name, and with an operand past it.
    expect_abi(
        renamed(&|b, inner| node(b, &[S("air.rename_qualified_function_signature"), N(inner)])),
        AbiError::MissingKey,
    );
    expect_abi(
        renamed(&|b, inner| {
            node(
                b,
                &[
                    S("air.rename_qualified_function_signature"),
                    N(inner),
                    S("qualified_name"),
                    S("more"),
                ],
            )
        }),
        AbiError::UnknownKey,
    );
}

// ---- Visible functions (`!air.visible`) ----

/// `float f(int x, <padding>)` plus helper functions, with `!air.visible` entry `visible`
/// (built from the function and its helpers).
fn visible_module(
    visible: impl FnOnce(&mut ModuleBuilder, FuncId, [FuncId; 3]) -> Vec<MdId>,
) -> Result<AirAbi, Error> {
    module_with(|b| {
        let float = b.float();
        let i32t = b.int(32);
        let ptr = b.ptr(AddrSpace::Thread);
        let f = entry_fn(b, "f", float, &[i32t, ptr]);
        let void = b.void();
        let h = [
            entry_fn(b, "load", void, &[ptr]),
            entry_fn(b, "store", void, &[ptr]),
            entry_fn(b, "size", void, &[]),
        ];
        let entries = visible(b, f, h);
        b.named_md("air.visible", &entries);
    })
}

fn visible_output(b: &mut ModuleBuilder) -> MdId {
    let o = node(
        b,
        &[S("air.visible_output"), S("air.arg_type_name"), S("float")],
    );
    node(b, &[N(o)])
}

fn visible_inputs(b: &mut ModuleBuilder) -> MdId {
    let x = node(
        b,
        &[
            I(0),
            S("air.visible_input"),
            S("air.arg_type_name"),
            S("int"),
            S("air.arg_name"),
            S("x"),
        ],
    );
    let pad = node(b, &[I(1), S("air.visible_padding")]);
    node(b, &[N(x), N(pad)])
}

fn stitching_type(b: &mut ModuleBuilder, h: [FuncId; 3]) -> MdId {
    let descriptor = node(b, &[S("air.float_type"), I(4), I(4), Z]);
    node(
        b,
        &[
            S("air.stitching_type"),
            N(descriptor),
            F(h[0]),
            F(h[1]),
            F(h[2]),
            Z,
        ],
    )
}

#[test]
fn visible_functions_decode() {
    let abi = visible_module(|b, f, h| {
        let outputs = visible_output(b);
        let inputs = visible_inputs(b);
        let ty = stitching_type(b, h);
        let named = node(b, &[S("air.stitching_argument"), N(ty), S("x")]);
        let unnamed = node(b, &[S("air.stitching_argument"), N(ty)]);
        let info = node(b, &[S("air.stitching_info"), N(ty), N(named), N(unnamed)]);
        vec![node(b, &[F(f), N(outputs), N(inputs), N(info)])]
    })
    .unwrap();
    let [v] = abi.visible_functions.as_slice() else {
        panic!("one visible function: {:?}", abi.visible_functions);
    };
    assert_eq!(abi.visible_function(v.func), Some(v));
    assert_eq!(
        v.params,
        vec![
            VisibleParam {
                param: 0,
                kind: VisibleParamKind::Input {
                    type_name: Some(DiagName::new("int")),
                    name: Some(DiagName::new("x")),
                },
            },
            VisibleParam {
                param: 1,
                kind: VisibleParamKind::Padding,
            },
        ]
    );
    assert_eq!(
        v.outputs,
        vec![VisibleOutput {
            type_name: Some(DiagName::new("float"))
        }]
    );
    let stitching = v.stitching.as_ref().expect("stitching info");
    let helpers = stitching.result.expect("a result type").helpers;
    assert!(helpers[..3].iter().all(Option::is_some) && helpers[3].is_none());
    assert_eq!(stitching.arguments.len(), 2);
    assert_eq!(stitching.arguments[0].name, Some(DiagName::new("x")));
    assert_eq!(stitching.arguments[1].name, None);
    assert!(abi.entries.is_empty());

    // A plain `[[visible]]` function carries no stitching info; a `void` stitchable function
    // has a `null` result.
    let abi = visible_module(|b, f, _| {
        let outputs = visible_output(b);
        let inputs = visible_inputs(b);
        vec![node(b, &[F(f), N(outputs), N(inputs)])]
    })
    .unwrap();
    assert_eq!(abi.visible_functions[0].stitching, None);
    let abi = visible_module(|b, f, _| {
        let outputs = visible_output(b);
        let inputs = visible_inputs(b);
        let info = node(b, &[S("air.stitching_info"), Z]);
        vec![node(b, &[F(f), N(outputs), N(inputs), N(info)])]
    })
    .unwrap();
    assert_eq!(
        abi.visible_functions[0].stitching,
        Some(Stitching {
            result: None,
            arguments: vec![]
        })
    );
}

#[test]
fn visible_function_grammar_is_strict() {
    // Replace one input node.
    let with_input = |input: &'static [O<'static>]| {
        visible_module(move |b, f, _| {
            let outputs = visible_output(b);
            let x = node(b, input);
            let pad = node(b, &[I(1), S("air.visible_padding")]);
            let inputs = node(b, &[N(x), N(pad)]);
            vec![node(b, &[F(f), N(outputs), N(inputs)])]
        })
    };
    // A key the visible grammar does not define, a known role of another list, and a repeat.
    expect_abi(
        with_input(&[I(0), S("air.visible_input"), S("air.arg_unused")]),
        AbiError::UnknownKey,
    );
    expect_abi(
        with_input(&[I(0), S("air.buffer"), S("air.arg_name"), S("x")]),
        AbiError::UnknownKey,
    );
    expect_abi(
        with_input(&[I(0), S("air.visible_input"), S("air.not_a_key")]),
        AbiError::UnknownKey,
    );
    expect_abi(
        with_input(&[
            I(0),
            S("air.visible_input"),
            S("air.arg_name"),
            S("x"),
            S("air.arg_name"),
            S("y"),
        ]),
        AbiError::BadValue,
    );
    expect_abi(
        with_input(&[I(0), S("air.visible_padding"), S("air.arg_name"), S("x")]),
        AbiError::UnknownKey,
    );
    // Parameter bindings against the signature: a repeated, out-of-range, or missing binding.
    expect_abi(
        with_input(&[I(1), S("air.visible_input")]),
        AbiError::SignatureMismatch,
    );
    expect_abi(
        with_input(&[I(2), S("air.visible_input")]),
        AbiError::SignatureMismatch,
    );
    expect_abi(
        visible_module(|b, f, _| {
            let outputs = visible_output(b);
            let pad = node(b, &[I(1), S("air.visible_padding")]);
            let inputs = node(b, &[N(pad)]);
            vec![node(b, &[F(f), N(outputs), N(inputs)])]
        }),
        AbiError::SignatureMismatch,
    );
    // An output only on a non-`void` function, and only a type name.
    expect_abi(
        visible_module(|b, f, _| {
            let outputs = node(b, &[]);
            let inputs = visible_inputs(b);
            vec![node(b, &[F(f), N(outputs), N(inputs)])]
        }),
        AbiError::SignatureMismatch,
    );
    expect_abi(
        visible_module(|b, _, h| {
            let outputs = visible_output(b);
            let inputs = node(b, &[]);
            vec![node(b, &[F(h[2]), N(outputs), N(inputs)])]
        }),
        AbiError::SignatureMismatch,
    );
    expect_abi(
        visible_module(|b, f, _| {
            let o = node(b, &[S("air.visible_output"), S("air.arg_name"), S("r")]);
            let outputs = node(b, &[N(o)]);
            let inputs = visible_inputs(b);
            vec![node(b, &[F(f), N(outputs), N(inputs)])]
        }),
        AbiError::UnknownKey,
    );
    // A function listed twice, and a declaration (a visible function is defined here).
    expect_abi(
        visible_module(|b, f, _| {
            let outputs = visible_output(b);
            let inputs = visible_inputs(b);
            let v = node(b, &[F(f), N(outputs), N(inputs)]);
            vec![v, v]
        }),
        AbiError::BadValue,
    );
    expect_abi(
        visible_module(|b, _, _| {
            let void = b.void();
            let ty = b.fn_ty(void, &[]);
            let d = b.declare("elsewhere", ty);
            let outputs = node(b, &[]);
            let inputs = node(b, &[]);
            vec![node(b, &[F(d), N(outputs), N(inputs)])]
        }),
        AbiError::SignatureMismatch,
    );
    // Not a function, and an extra operand after the stitching info.
    expect_abi(
        visible_module(|b, _, _| {
            let outputs = node(b, &[]);
            let inputs = node(b, &[]);
            vec![node(b, &[I(0), N(outputs), N(inputs)])]
        }),
        AbiError::BadValue,
    );
    expect_abi(
        visible_module(|b, f, _| {
            let outputs = visible_output(b);
            let inputs = visible_inputs(b);
            let info = node(b, &[S("air.stitching_info"), Z]);
            vec![node(b, &[F(f), N(outputs), N(inputs), N(info), N(info)])]
        }),
        AbiError::UnknownKey,
    );
}

#[test]
fn stitching_info_grammar_is_strict() {
    let with_info = |info: fn(&mut ModuleBuilder, [FuncId; 3]) -> MdId| {
        visible_module(move |b, f, h| {
            let outputs = visible_output(b);
            let inputs = visible_inputs(b);
            let info = info(b, h);
            vec![node(b, &[F(f), N(outputs), N(inputs), N(info)])]
        })
    };
    // Wrong tags.
    expect_abi(
        with_info(|b, _| node(b, &[S("air.stitching_argument"), Z])),
        AbiError::UnknownKey,
    );
    expect_abi(
        with_info(|b, h| {
            let ty = stitching_type(b, h);
            let arg = node(b, &[S("air.stitching_type"), N(ty)]);
            node(b, &[S("air.stitching_info"), Z, N(arg)])
        }),
        AbiError::UnknownKey,
    );
    // A result that is neither `null` nor a node, and an argument that is not a node.
    expect_abi(
        with_info(|b, _| node(b, &[S("air.stitching_info"), I(0)])),
        AbiError::BadValue,
    );
    expect_abi(
        with_info(|b, _| node(b, &[S("air.stitching_info"), Z, Z])),
        AbiError::BadValue,
    );
    // A stitching type with a helper that is not a function, with three helpers, with five,
    // and with a descriptor that is not a node.
    expect_abi(
        with_info(|b, h| {
            let d = node(b, &[]);
            let ty = node(
                b,
                &[S("air.stitching_type"), N(d), F(h[0]), I(1), F(h[2]), Z],
            );
            node(b, &[S("air.stitching_info"), N(ty)])
        }),
        AbiError::BadValue,
    );
    expect_abi(
        with_info(|b, h| {
            let d = node(b, &[]);
            let ty = node(b, &[S("air.stitching_type"), N(d), F(h[0]), F(h[1]), Z]);
            node(b, &[S("air.stitching_info"), N(ty)])
        }),
        AbiError::MissingKey,
    );
    expect_abi(
        with_info(|b, h| {
            let d = node(b, &[]);
            let ty = node(
                b,
                &[S("air.stitching_type"), N(d), F(h[0]), F(h[1]), Z, Z, Z],
            );
            node(b, &[S("air.stitching_info"), N(ty)])
        }),
        AbiError::UnknownKey,
    );
    expect_abi(
        with_info(|b, h| {
            let ty = node(
                b,
                &[S("air.stitching_type"), S("float"), F(h[0]), F(h[1]), Z, Z],
            );
            node(b, &[S("air.stitching_info"), N(ty)])
        }),
        AbiError::BadValue,
    );
    // An argument with a name that is not a string, or with an extra operand.
    expect_abi(
        with_info(|b, h| {
            let ty = stitching_type(b, h);
            let arg = node(b, &[S("air.stitching_argument"), N(ty), I(0)]);
            node(b, &[S("air.stitching_info"), Z, N(arg)])
        }),
        AbiError::BadValue,
    );
    expect_abi(
        with_info(|b, h| {
            let ty = stitching_type(b, h);
            let arg = node(b, &[S("air.stitching_argument"), N(ty), S("x"), S("y")]);
            node(b, &[S("air.stitching_info"), Z, N(arg)])
        }),
        AbiError::UnknownKey,
    );
}

// ---- Core Image kernels (`!air.ci`) ----

/// The functions of [`core_image_module`]: `k` returns a `float4` from a texture wrapped in a
/// struct, a sampler wrapped in a struct, a constant pointer, a `float2`, a bare and a wrapped
/// matrix, and a `float`; `w` returns nothing and takes a bare device pointer and a `float2`;
/// `agg` returns an array and takes nothing.
struct CiFns {
    k: FuncId,
    w: FuncId,
    agg: FuncId,
}

/// A module with the functions of [`CiFns`] and the `!air.ci` entries `ci` builds.
fn core_image_module(
    ci: impl FnOnce(&mut ModuleBuilder, &CiFns) -> Vec<MdId>,
) -> Result<AirAbi, Error> {
    module_with(|b| {
        let float = b.float();
        let f2 = b.vector(float, 2);
        let f3 = b.vector(float, 3);
        let f4 = b.vector(float, 4);
        let dev = b.ptr(AddrSpace::Device);
        let cst = b.ptr(AddrSpace::Constant);
        let texture = b.struct_ty(&[dev]);
        let sampler = b.struct_ty(&[cst]);
        let m4 = b.array(f4, 4);
        let m3 = b.array(f3, 3);
        let m3 = b.struct_ty(&[m3]);
        let void = b.void();
        let fns = CiFns {
            k: entry_fn(b, "k", f4, &[texture, sampler, cst, f2, m4, m3, float]),
            w: entry_fn(b, "w", void, &[dev, f2]),
            agg: entry_fn(b, "agg", m4, &[]),
        };
        let entries = ci(b, &fns);
        b.named_md("air.ci", &entries);
    })
}

/// `k`'s input nodes, one per parameter.
fn ci_k_inputs(b: &mut ModuleBuilder) -> Vec<MdId> {
    [
        &[
            I(0),
            S("air.ci_texture"),
            S("air.location_count"),
            I(1),
            S("air.sample"),
            S("air.arg_type_name"),
            S("texture2d<float, sample>"),
            S("air.arg_name"),
            S("src.coerce0"),
        ][..],
        &[
            I(1),
            S("air.ci_sampler"),
            S("air.arg_type_name"),
            S("sampler"),
            S("air.arg_name"),
            S("src.coerce1"),
        ],
        &[
            I(2),
            S("air.ci_pointer"),
            S("air.read"),
            S("air.arg_type_size"),
            I(64),
            S("air.arg_type_align_size"),
            I(16),
            S("air.arg_type_name"),
            S("float4x4"),
        ],
        &[
            I(3),
            S("air.ci_builtin"),
            S("air.arg_type_name"),
            S("float2"),
        ],
        &[
            I(4),
            S("air.ci_matrix"),
            S("air.arg_type_name"),
            S("float4x4"),
            S("air.arg_name"),
            S("m"),
        ],
        &[I(5), S("air.ci_matrix")],
        // Attributes may come in any order.
        &[
            I(6),
            S("air.ci_builtin"),
            S("air.arg_name"),
            S("scale"),
            S("air.arg_type_name"),
            S("float"),
        ],
    ]
    .iter()
    .map(|ops| node(b, ops))
    .collect()
}

/// `!{!"air.ci_builtin", !"air.arg_type_name", !"float4"}` in an outputs list.
fn ci_output(b: &mut ModuleBuilder) -> MdId {
    let o = node(
        b,
        &[S("air.ci_builtin"), S("air.arg_type_name"), S("float4")],
    );
    node(b, &[N(o)])
}

/// An `!air.ci` entry of `f` with the outputs list `outputs` and the input nodes `inputs`.
fn ci_entry(b: &mut ModuleBuilder, f: FuncId, outputs: MdId, inputs: &[MdId]) -> MdId {
    let ops: Vec<O<'_>> = inputs.iter().map(|&i| N(i)).collect();
    let inputs = node(b, &ops);
    node(b, &[F(f), N(outputs), N(inputs)])
}

/// `w`'s entry: no output, a written texture (attributes in another order), and a `float2`.
fn ci_w_entry(b: &mut ModuleBuilder, w: FuncId) -> MdId {
    let outputs = node(b, &[]);
    let tex = node(
        b,
        &[
            I(0),
            S("air.ci_texture"),
            S("air.write"),
            S("air.location_count"),
            I(1),
        ],
    );
    let dc = node(b, &[I(1), S("air.ci_builtin")]);
    ci_entry(b, w, outputs, &[tex, dc])
}

/// The standard module with `k`'s input `index` replaced by `ops`.
fn ci_with_input(index: usize, ops: &'static [O<'static>]) -> Result<AirAbi, Error> {
    core_image_module(move |b, f| {
        let outputs = ci_output(b);
        let mut inputs = ci_k_inputs(b);
        inputs[index] = node(b, ops);
        vec![ci_entry(b, f.k, outputs, &inputs)]
    })
}

#[test]
fn core_image_kernels_decode() {
    let abi = core_image_module(|b, f| {
        let outputs = ci_output(b);
        let inputs = ci_k_inputs(b);
        vec![ci_entry(b, f.k, outputs, &inputs), ci_w_entry(b, f.w)]
    })
    .unwrap();
    let [k, w] = abi.core_image_kernels.as_slice() else {
        panic!("two kernels: {:?}", abi.core_image_kernels);
    };
    let name = |s: &str| Some(DiagName::new(s));
    let param = |param, role, type_name: Option<DiagName>, name: Option<DiagName>| CoreImageParam {
        param,
        role,
        type_name,
        name,
    };
    assert_eq!(k.name, DiagName::new("k"));
    assert_eq!(
        k.output,
        Some(CoreImageOutput {
            type_name: name("float4")
        })
    );
    assert_eq!(
        k.params,
        vec![
            param(
                0,
                CoreImageRole::Texture {
                    location_count: 1,
                    access: Access::Sample,
                },
                name("texture2d<float, sample>"),
                name("src.coerce0"),
            ),
            param(
                1,
                CoreImageRole::Sampler,
                name("sampler"),
                name("src.coerce1")
            ),
            param(
                2,
                CoreImageRole::Pointer {
                    access: Access::Read,
                    size: 64,
                    align: 16,
                },
                name("float4x4"),
                None,
            ),
            param(3, CoreImageRole::Builtin, name("float2"), None),
            param(4, CoreImageRole::Matrix, name("float4x4"), name("m")),
            param(5, CoreImageRole::Matrix, None, None),
            param(6, CoreImageRole::Builtin, name("float"), name("scale")),
        ]
    );
    assert_eq!(w.output, None);
    assert_eq!(
        w.params,
        vec![
            param(
                0,
                CoreImageRole::Texture {
                    location_count: 1,
                    access: Access::Write,
                },
                None,
                None,
            ),
            param(1, CoreImageRole::Builtin, None, None),
        ]
    );
    // A Core Image kernel list declares no entry point and no visible function.
    assert!(abi.entries.is_empty() && abi.visible_functions.is_empty());
    // A sampler handle may also be passed as a bare constant pointer.
    let abi = ci_with_input(2, &[I(2), S("air.ci_sampler")]).unwrap();
    assert_eq!(
        abi.core_image_kernels[0].params[2].role,
        CoreImageRole::Sampler
    );

    // Every texture and pointer access the roles define.
    for (access, want) in [
        ("air.read", Access::Read),
        ("air.write", Access::Write),
        ("air.read_write", Access::ReadWrite),
    ] {
        let abi = core_image_module(|b, f| {
            let outputs = ci_output(b);
            let mut inputs = ci_k_inputs(b);
            inputs[0] = node(
                b,
                &[
                    I(0),
                    S("air.ci_texture"),
                    S("air.location_count"),
                    I(2),
                    S(access),
                ],
            );
            inputs[2] = node(
                b,
                &[
                    I(2),
                    S("air.ci_pointer"),
                    S("air.arg_type_align_size"),
                    I(4),
                    S("air.arg_type_size"),
                    I(12),
                    S(access),
                ],
            );
            vec![ci_entry(b, f.k, outputs, &inputs)]
        })
        .unwrap();
        let params = &abi.core_image_kernels[0].params;
        assert_eq!(
            params[0].role,
            CoreImageRole::Texture {
                location_count: 2,
                access: want,
            }
        );
        assert_eq!(
            params[2].role,
            CoreImageRole::Pointer {
                access: want,
                size: 12,
                align: 4,
            }
        );
    }
}

#[test]
fn core_image_grammar_is_strict() {
    // An unknown key, a role of another list, and an attribute of another list.
    expect_abi(
        ci_with_input(3, &[I(3), S("air.ci_builtin"), S("air.not_a_key")]),
        AbiError::UnknownKey,
    );
    expect_abi(
        ci_with_input(3, &[I(3), S("air.visible_input")]),
        AbiError::UnknownKey,
    );
    expect_abi(
        ci_with_input(3, &[I(3), S("air.buffer")]),
        AbiError::UnknownKey,
    );
    expect_abi(
        ci_with_input(3, &[I(3), S("air.ci_builtin"), S("air.arg_unused")]),
        AbiError::UnknownKey,
    );
    // A key another Core Image role defines.
    expect_abi(
        ci_with_input(
            1,
            &[I(1), S("air.ci_sampler"), S("air.location_count"), I(1)],
        ),
        AbiError::UnknownKey,
    );
    expect_abi(
        ci_with_input(4, &[I(4), S("air.ci_matrix"), S("air.read")]),
        AbiError::UnknownKey,
    );
    expect_abi(
        ci_with_input(
            2,
            &[
                I(2),
                S("air.ci_pointer"),
                S("air.sample"),
                S("air.arg_type_size"),
                I(64),
                S("air.arg_type_align_size"),
                I(16),
            ],
        ),
        AbiError::UnknownKey,
    );
    // A repeated key and a second access.
    expect_abi(
        ci_with_input(
            3,
            &[
                I(3),
                S("air.ci_builtin"),
                S("air.arg_name"),
                S("a"),
                S("air.arg_name"),
                S("b"),
            ],
        ),
        AbiError::BadValue,
    );
    expect_abi(
        ci_with_input(
            0,
            &[
                I(0),
                S("air.ci_texture"),
                S("air.location_count"),
                I(1),
                S("air.sample"),
                S("air.write"),
            ],
        ),
        AbiError::BadValue,
    );
    // Missing required keys.
    expect_abi(
        ci_with_input(0, &[I(0), S("air.ci_texture"), S("air.sample")]),
        AbiError::MissingKey,
    );
    expect_abi(
        ci_with_input(
            0,
            &[I(0), S("air.ci_texture"), S("air.location_count"), I(1)],
        ),
        AbiError::MissingKey,
    );
    for pointer in [
        &[
            I(2),
            S("air.ci_pointer"),
            S("air.arg_type_size"),
            I(64),
            S("air.arg_type_align_size"),
            I(16),
        ][..],
        &[
            I(2),
            S("air.ci_pointer"),
            S("air.read"),
            S("air.arg_type_align_size"),
            I(16),
        ],
        &[
            I(2),
            S("air.ci_pointer"),
            S("air.read"),
            S("air.arg_type_size"),
            I(64),
        ],
    ] {
        expect_abi(ci_with_input(2, pointer), AbiError::MissingKey);
    }
    // Wrong value types: a string count, a string parameter index, and a missing role.
    expect_abi(
        ci_with_input(
            0,
            &[
                I(0),
                S("air.ci_texture"),
                S("air.location_count"),
                S("1"),
                S("air.sample"),
            ],
        ),
        AbiError::BadValue,
    );
    expect_abi(
        ci_with_input(3, &[S("3"), S("air.ci_builtin")]),
        AbiError::BadValue,
    );
    expect_abi(ci_with_input(3, &[I(3)]), AbiError::MissingKey);

    // The output: only `air.ci_builtin`, only a type name, and at most one.
    let with_output = |ops: &'static [O<'static>], count: usize| {
        core_image_module(move |b, f| {
            let o = node(b, ops);
            let outputs = node(b, &vec![N(o); count]);
            let inputs = ci_k_inputs(b);
            vec![ci_entry(b, f.k, outputs, &inputs)]
        })
    };
    expect_abi(with_output(&[S("air.ci_matrix")], 1), AbiError::UnknownKey);
    expect_abi(
        with_output(&[S("air.ci_builtin"), S("air.arg_name"), S("r")], 1),
        AbiError::UnknownKey,
    );
    expect_abi(
        with_output(&[S("air.ci_builtin")], 2),
        AbiError::SignatureMismatch,
    );

    // The entry: a function, then exactly two lists.
    expect_abi(
        core_image_module(|b, _| {
            let outputs = ci_output(b);
            let inputs = node(b, &[]);
            vec![node(b, &[I(0), N(outputs), N(inputs)])]
        }),
        AbiError::BadValue,
    );
    expect_abi(
        core_image_module(|b, f| {
            let outputs = node(b, &[]);
            vec![node(b, &[F(f.agg), N(outputs)])]
        }),
        AbiError::MissingKey,
    );
    expect_abi(
        core_image_module(|b, f| {
            let outputs = ci_output(b);
            let inputs = node(b, &[]);
            vec![node(b, &[F(f.agg), N(outputs), N(inputs), N(inputs)])]
        }),
        AbiError::UnknownKey,
    );
}

#[test]
fn core_image_kernels_are_checked_against_signatures() {
    // Parameter bindings: a missing, a repeated, and an out-of-range binding.
    expect_abi(
        core_image_module(|b, f| {
            let outputs = ci_output(b);
            let inputs = ci_k_inputs(b);
            vec![ci_entry(b, f.k, outputs, &inputs[..6])]
        }),
        AbiError::SignatureMismatch,
    );
    expect_abi(
        ci_with_input(6, &[I(5), S("air.ci_matrix")]),
        AbiError::SignatureMismatch,
    );
    expect_abi(
        ci_with_input(6, &[I(7), S("air.ci_builtin")]),
        AbiError::SignatureMismatch,
    );
    // An output exactly for a non-`void` function.
    expect_abi(
        core_image_module(|b, f| {
            let outputs = node(b, &[]);
            let inputs = ci_k_inputs(b);
            vec![ci_entry(b, f.k, outputs, &inputs)]
        }),
        AbiError::SignatureMismatch,
    );
    expect_abi(
        core_image_module(|b, f| {
            let outputs = ci_output(b);
            let tex = node(
                b,
                &[
                    I(0),
                    S("air.ci_texture"),
                    S("air.location_count"),
                    I(1),
                    S("air.write"),
                ],
            );
            let dc = node(b, &[I(1), S("air.ci_builtin")]);
            vec![ci_entry(b, f.w, outputs, &[tex, dc])]
        }),
        AbiError::SignatureMismatch,
    );
    // A kernel listed twice, and a declaration (a kernel is defined here).
    expect_abi(
        core_image_module(|b, f| {
            let e = ci_w_entry(b, f.w);
            vec![e, e]
        }),
        AbiError::BadValue,
    );
    expect_abi(
        core_image_module(|b, _| {
            let void = b.void();
            let ty = b.fn_ty(void, &[]);
            let d = b.declare("elsewhere", ty);
            let outputs = node(b, &[]);
            vec![ci_entry(b, d, outputs, &[])]
        }),
        AbiError::SignatureMismatch,
    );
    // Each role against its parameter's type: a texture on a `float2`, a sampler on a wrapped
    // device pointer, a texture on a bare constant pointer, a builtin on a pointer, a matrix on
    // a `float`, a pointer on a `float2`, and a builtin on a matrix.
    for (index, ops) in [
        (
            3,
            &[
                I(3),
                S("air.ci_texture"),
                S("air.location_count"),
                I(1),
                S("air.sample"),
            ][..],
        ),
        (0, &[I(0), S("air.ci_sampler")]),
        (
            2,
            &[
                I(2),
                S("air.ci_texture"),
                S("air.location_count"),
                I(1),
                S("air.sample"),
            ],
        ),
        (2, &[I(2), S("air.ci_builtin")]),
        (6, &[I(6), S("air.ci_matrix")]),
        (
            3,
            &[
                I(3),
                S("air.ci_pointer"),
                S("air.read"),
                S("air.arg_type_size"),
                I(8),
                S("air.arg_type_align_size"),
                I(8),
            ],
        ),
        (4, &[I(4), S("air.ci_builtin")]),
    ] {
        expect_abi(ci_with_input(index, ops), AbiError::RoleTypeMismatch);
    }
    // The output is a scalar or vector number.
    expect_abi(
        core_image_module(|b, f| {
            let outputs = ci_output(b);
            vec![ci_entry(b, f.agg, outputs, &[])]
        }),
        AbiError::RoleTypeMismatch,
    );
}

/// The ABI cross-check for typed-pointer bitcode: a Core Image pointer's size and alignment against
/// its pointee. (The decoder calls this for every parameter with a typed pointee; a `PointeeTable`
/// only comes from bitcode, so the check is exercised directly, like `pointee_check`.)
#[test]
fn core_image_pointer_sizes_are_checked_against_typed_pointees() {
    let mut b = ModuleBuilder::new();
    let float = b.float();
    let f4 = b.vector(float, 4);
    let m4 = b.array(f4, 4);
    let m = b.finish().unwrap();
    let layouts = TypeLayouts::new(m.data_layout(), m.types(), &mut meter()).unwrap();
    let check = |role, pointee| {
        let p = CoreImageParam {
            param: 0,
            role,
            type_name: None,
            name: None,
        };
        core_image_pointee_check(&layouts, &p, pointee, &Site::module()).map_err(abi_kind)
    };
    let pointer = |size, align| CoreImageRole::Pointer {
        access: Access::Read,
        size,
        align,
    };
    assert_eq!(check(pointer(64, 16), m4), Ok(()));
    // The size disagrees with the pointee, and the alignment is not a power of two.
    assert_eq!(check(pointer(48, 16), m4), Err(AbiError::TypeSizeMismatch));
    assert_eq!(check(pointer(64, 12), m4), Err(AbiError::TypeSizeMismatch));
    // No other role states a pointee size.
    assert_eq!(check(CoreImageRole::Matrix, m4), Ok(()));
}

#[test]
fn control_point_fetches_are_exposed() {
    let abi = module_with(|b| {
        let ptr = b.ptr(AddrSpace::Thread);
        let void = b.void();
        let i32t = b.int(32);
        // Metal's shape: `{ T0, .. } (i32 index, ptr input)`, one member per field.
        let f32t = b.float();
        let float4 = b.vector(f32t, 4);
        let point = b.struct_ty(&[float4]);
        let fetch_ty = b.fn_ty(point, &[i32t, ptr]);
        let fetch = b.declare("fetch", fetch_ty);
        let f = entry_fn(b, "post", void, &[ptr]);
        let function = node(b, &[S("air.patch_control_point_function"), F(fetch)]);
        let field = node(
            b,
            &[
                S("air.location_index"),
                I(0),
                I(1),
                S("air.arg_type_name"),
                S("float4"),
                S("air.arg_name"),
                S("position"),
                S("air.arg_unused"),
            ],
        );
        let input = node(
            b,
            &[
                I(0),
                S("air.patch_control_point_input"),
                N(function),
                N(field),
            ],
        );
        let inputs = node(b, &[N(input)]);
        let outputs = node(b, &[]);
        let patch = node(
            b,
            &[
                S("air.patch"),
                S("triangle"),
                S("air.patch_control_point"),
                I(3),
            ],
        );
        let e = node(b, &[F(f), N(outputs), N(inputs), N(patch)]);
        b.named_md("air.vertex", &[e]);
    })
    .unwrap();
    let fetches: Vec<_> = abi.control_point_fetches().collect();
    assert_eq!(fetches.len(), 1);
    let ParamRole::PatchInput(PatchInput::ControlPoints { function, .. }) =
        &abi.entries[0].params[0].role
    else {
        panic!("a control-point input: {:?}", abi.entries[0].params);
    };
    assert_eq!(fetches, vec![*function]);
    let ParamRole::PatchInput(PatchInput::ControlPoints { fields, .. }) =
        &abi.entries[0].params[0].role
    else {
        unreachable!()
    };
    assert_eq!(fields.len(), 1);
    assert!(fields[0].unused && !fields[0].gated_off);
    assert_eq!(
        Kernel::standard()
            .decode()
            .unwrap()
            .control_point_fetches()
            .count(),
        0
    );
}

/// A post-tessellation vertex function's `air.patch` with each count form: `[[patch(quad, 4)]]`
/// states its count, `[[patch(quad)]]` none (the node ends at the kind), and a count key without a
/// count is malformed.
#[test]
fn a_patch_states_its_count_or_none() {
    let patch_of = |ops: &'static [&'static str], count: Option<u64>| {
        module_with(|b| {
            let void = b.void();
            let f = entry_fn(b, "post", void, &[]);
            let mut operands: Vec<O> = ops.iter().map(|&o| S(o)).collect();
            operands.extend(count.map(I));
            let patch = node(b, &operands);
            let empty = node(b, &[]);
            let e = node(b, &[F(f), N(empty), N(empty), N(patch)]);
            b.named_md("air.vertex", &[e]);
        })
    };
    let stage = |r: Result<AirAbi, Error>| r.expect("the patch decodes").entries[0].stage;
    assert_eq!(
        stage(patch_of(
            &["air.patch", "quad", "air.patch_control_point"],
            Some(4)
        )),
        Stage::Vertex {
            patch: Some(PatchInfo {
                kind: PatchKind::Quad,
                control_points: Some(4),
            })
        }
    );
    assert_eq!(
        stage(patch_of(&["air.patch", "quad"], None)),
        Stage::Vertex {
            patch: Some(PatchInfo {
                kind: PatchKind::Quad,
                control_points: None,
            })
        }
    );
    assert!(patch_of(&["air.patch", "triangle", "air.patch_control_point"], None).is_err());
    expect_abi(
        patch_of(&["air.patch", "triangle", "air.patch_count"], Some(3)),
        AbiError::MissingKey,
    );
}

// ---- Container cross-check ----

fn library_function(
    stage: FunctionStage,
    constants: Vec<ContainerConstant<'static>>,
) -> LibraryFunction<'static> {
    LibraryFunction {
        name: "k",
        stage,
        hash: [0; 32],
        air_version: (2, 8),
        language_version: (3, 1),
        bitcode: &[],
        public: FunctionPublicMetadata {
            constants,
            ..FunctionPublicMetadata::default()
        },
        private: vec![],
        raw_tags: vec![],
    }
}

/// The standard kernel with one function constant per `(index, value type)`, as the frontend
/// writes them (an `undef` initializer global in `air.fc_initializer`).
fn kernel_with_constants(
    constants: &[(u32, fn(&mut ModuleBuilder) -> TypeId)],
) -> (Module, AirAbi) {
    let mut k = Kernel::standard();
    let mut nodes = Vec::new();
    for (n, &(index, ty)) in constants.iter().enumerate() {
        let ty = ty(&mut k.b);
        let undef = k.b.constant(ty, Const::Undef);
        let name = k.b.sym(&format!("fc{n}"));
        let g = k.b.global(Global {
            name,
            value_ty: ty,
            space: AddrSpace::Constant,
            constant: true,
            init: Some(undef),
            externally_initialized: true,
            visibility: Default::default(),
            unnamed_addr: Default::default(),
            linkage: Linkage::Internal,
            align: None,
            section: Some(Section::FcInitializer),
        });
        nodes.push(node(&mut k.b, &[G(g), S("t"), S("K"), I(u64::from(index))]));
    }
    k.b.named_md("air.function_constants", &nodes);
    let m = k.b.finish().unwrap();
    let abi = decode(&m, &PointeeTable::default(), &mut meter()).unwrap();
    (m, abi)
}

fn cnst_raw(index: u16, data_type: u8) -> ContainerConstant<'static> {
    ContainerConstant {
        name: "K",
        data_type,
        index,
        flags: 0,
    }
}

fn cnst(index: u16, t: DataType) -> ContainerConstant<'static> {
    cnst_raw(index, u8::try_from(t as u32).unwrap())
}

fn check(abi: &mut AirAbi, m: &Module, f: &LibraryFunction<'_>) -> Result<(), Error> {
    abi.check_container(m, f, &mut meter())
}

#[test]
fn container_metadata_is_cross_checked() {
    let (m, mut abi) = kernel_with_constants(&[]);
    let kernel = |c| library_function(FunctionStage::Kernel, c);
    assert_eq!(check(&mut abi, &m, &kernel(vec![])), Ok(()));
    let err = check(
        &mut abi,
        &m,
        &library_function(FunctionStage::Fragment, vec![]),
    )
    .unwrap_err();
    assert_eq!(abi_kind(err), AbiError::ContainerMismatch);
    // Mesh and object functions are modeled stages: a kernel's module is no mesh or object
    // function.
    for stage in [FunctionStage::Mesh, FunctionStage::Object] {
        let err = check(&mut abi, &m, &library_function(stage, vec![])).unwrap_err();
        assert_eq!(abi_kind(err), AbiError::ContainerMismatch);
    }
    // Unmodeled stages are unsupported, not a mismatch.
    for (stage, tag) in [
        (FunctionStage::Intersection, 6),
        (FunctionStage::Other(5), 5),
    ] {
        let err = check(&mut abi, &m, &library_function(stage, vec![])).unwrap_err();
        let Error::InvalidInput(i) = err else {
            panic!("not an input error: {err:?}")
        };
        assert_eq!(i.kind, InputErrorKind::Unsupported(Feature::Stage(tag)));
        assert_eq!(i.class(), ErrorClass::Unsupported);
    }
    // A `CNST` index no constant declares.
    let err = check(&mut abi, &m, &kernel(vec![cnst_raw(0, 37)])).unwrap_err();
    assert_eq!(abi_kind(err), AbiError::ContainerMismatch);
}

/// A metallib function's bitcode is its own module holding only that entry: a second entry, even of
/// another stage, or none at all is a container mismatch.
#[test]
fn container_function_declares_exactly_one_entry() {
    let mut k = Kernel::standard();
    let void = k.b.void();
    let v = entry_fn(&mut k.b, "v", void, &[]);
    let outputs = node(&mut k.b, &[]);
    let inputs = node(&mut k.b, &[]);
    let e = node(&mut k.b, &[F(v), N(outputs), N(inputs)]);
    k.b.named_md("air.vertex", &[e]);
    let m = k.b.finish().unwrap();
    let mut abi = decode(&m, &PointeeTable::default(), &mut meter()).unwrap();
    assert_eq!(abi.entries.len(), 2);
    for stage in [FunctionStage::Kernel, FunctionStage::Vertex] {
        let err = check(&mut abi, &m, &library_function(stage, vec![])).unwrap_err();
        assert_eq!(abi_kind(err), AbiError::ContainerMismatch);
    }

    let m = ModuleBuilder::new().finish().unwrap();
    let mut abi = decode(&m, &PointeeTable::default(), &mut meter()).unwrap();
    assert!(abi.entries.is_empty());
    let err = check(
        &mut abi,
        &m,
        &library_function(FunctionStage::Kernel, vec![]),
    )
    .unwrap_err();
    assert_eq!(abi_kind(err), AbiError::ContainerMismatch);
}

#[test]
fn container_constants_record_their_data_type() {
    let (m, mut abi) = kernel_with_constants(&[
        (3, |b| b.int(16)),
        (1, |b| {
            let f = b.float();
            b.vector(f, 4)
        }),
    ]);
    let f = library_function(
        FunctionStage::Kernel,
        vec![cnst(1, DataType::Float4), cnst(3, DataType::UShort)],
    );
    assert_eq!(check(&mut abi, &m, &f), Ok(()));
    let fc = &abi.function_constants;
    assert_eq!(fc[0].data_type, Some(DataType::UShort));
    assert_eq!(fc[1].data_type, Some(DataType::Float4));
    // Checking again with the same types is idempotent.
    assert_eq!(check(&mut abi, &m, &f), Ok(()));
}

#[test]
fn container_data_type_must_match_the_initializer_type() {
    type Ty = fn(&mut ModuleBuilder) -> TypeId;
    // (value type, a matching MTLDataType, a mismatching one). `bool` is stored as `i8`, as in
    // every corpus declaration, so no `MTLDataType` matches an `i1`.
    let cases: [(Ty, Option<DataType>, DataType); 9] = [
        (|b| b.int(8), Some(DataType::Bool), DataType::Short),
        (|b| b.int(8), Some(DataType::UChar), DataType::Bool2),
        (|b| b.int(1), None, DataType::Bool),
        (|b| b.int(32), Some(DataType::Int), DataType::Float),
        (|b| b.int(64), Some(DataType::ULong), DataType::UInt),
        (|b| b.ty(Type::Half), Some(DataType::Half), DataType::Short),
        (
            |b| {
                let h = b.ty(Type::Half);
                b.vector(h, 3)
            },
            Some(DataType::Half3),
            DataType::Half4,
        ),
        (
            |b| {
                let i = b.int(32);
                b.vector(i, 2)
            },
            Some(DataType::UInt2),
            DataType::UInt,
        ),
        // A function constant is never a matrix or a resource.
        (|b| b.float(), Some(DataType::Float), DataType::Float2x2),
    ];
    for (ty, good, bad) in cases {
        let (m, mut abi) = kernel_with_constants(&[(0, ty)]);
        let with = |t: DataType| library_function(FunctionStage::Kernel, vec![cnst(0, t)]);
        if let Some(good) = good {
            assert_eq!(check(&mut abi.clone(), &m, &with(good)), Ok(()), "{good:?}");
        }
        let err = check(&mut abi, &m, &with(bad)).unwrap_err();
        assert_eq!(abi_kind(err), AbiError::ContainerMismatch, "{bad:?}");
        assert_eq!(abi.function_constants[0].data_type, None);
    }
}

#[test]
fn container_mismatch_leaves_the_abi_unchanged() {
    let (m, mut abi) = kernel_with_constants(&[(0, |b| b.int(32)), (1, |b| b.int(32))]);
    let before = abi.clone();
    // The first entry is sound; the second disagrees with its declaration.
    let f = library_function(
        FunctionStage::Kernel,
        vec![cnst(0, DataType::Int), cnst(1, DataType::Float)],
    );
    let err = check(&mut abi, &m, &f).unwrap_err();
    assert_eq!(abi_kind(err), AbiError::ContainerMismatch);
    assert_eq!(abi, before);
    // Two entries for one constant with different types.
    let f = library_function(
        FunctionStage::Kernel,
        vec![cnst(0, DataType::Int), cnst(0, DataType::UInt)],
    );
    let err = check(&mut abi, &m, &f).unwrap_err();
    assert_eq!(abi_kind(err), AbiError::ContainerMismatch);
    assert_eq!(abi, before);
}

#[test]
fn container_check_is_metered() {
    let (m, mut abi) = kernel_with_constants(&[(0, |b| b.int(32))]);
    let f = library_function(FunctionStage::Kernel, vec![cnst(0, DataType::Int); 8]);
    let mut tiny = Meter::with_caps(&Budget::default(), 4, u64::MAX);
    assert_eq!(
        abi.check_container(&m, &f, &mut tiny),
        Err(Error::LimitExceeded(Limit::WorkUnits))
    );
    assert_eq!(abi.function_constants[0].data_type, None);
}

// ---- Bounded work ----

#[test]
fn decode_work_is_metered() {
    let m = Kernel::standard().b.finish().unwrap();
    let mut tiny = Meter::with_caps(&Budget::default(), 3, u64::MAX);
    assert_eq!(
        decode(&m, &PointeeTable::default(), &mut tiny),
        Err(Error::LimitExceeded(Limit::WorkUnits))
    );
}

/// Decode `m` and return the meter it ran under.
fn metered_decode(m: &Module) -> Meter {
    let mut meter = meter();
    decode(m, &PointeeTable::default(), &mut meter).unwrap();
    meter
}

/// Work and the ledger peak grow with the number of decoded layouts (each is ticked, and its
/// members, arena slots, and memo entry are charged) and with the bytes of every retained name.
#[test]
fn decode_work_and_ledger_scale_with_layouts_and_names() {
    let chain = |levels: u64| {
        layout_module(|b| {
            let top = layout_chain(b, levels, wrap_layout);
            let leaf = leaf_layout(b);
            [top, leaf]
        })
    };
    let (small, large) = (metered_decode(&chain(2)), metered_decode(&chain(64)));
    // 62 more layouts (64 is the nesting cap), each with one member: at least one tick, one
    // `StructInfo`, one `Member`, and the member's two names ("S", "s") each.
    let extra = 62u64;
    assert!(
        large.work() >= small.work() + extra,
        "{large:?} vs {small:?}"
    );
    let per_layout = (std::mem::size_of::<StructInfo>() + std::mem::size_of::<Member>() + 2) as u64;
    assert!(
        large.ledger_peak() >= small.ledger_peak() + extra * per_layout,
        "{large:?} vs {small:?}"
    );

    // Names: one argument name of `len` bytes is charged byte for byte.
    let named = |len: usize| {
        let name = "n".repeat(len);
        let mut arg = buffer_arg(0);
        arg.truncate(arg.len() - 2);
        arg.extend([S("air.arg_name"), S(&name)]);
        let k = Kernel::with(&[arg, texture_arg(1), sampler_arg(2), gid_arg(3)]);
        metered_decode(&k.b.finish().unwrap())
    };
    let (short, long) = (named(1), named(1 << 16));
    assert!(long.ledger_peak() >= short.ledger_peak() + (1 << 16) - 1);
    assert!(long.ledger() >= short.ledger() + (1 << 16) - 1);
}

// ---- Constexpr sampler words (ported from the reflection tests) ----

#[test]
fn half_to_f32_is_exact() {
    for (bits, want) in [
        (0x0000, 0.0),
        (0x8000, -0.0),
        (0x3c00, 1.0),
        (0xb800, -0.5),
        (0x4000, 2.0),
        (0x7bff, 65504.0),
        (0x0001, 1.0 / 16_777_216.0),
        (0x0200, 1.0 / 32768.0),
        (0x4366, 947.0 / 256.0),
    ] {
        assert_eq!(Half(bits).to_f32(), want, "{bits:#06x}");
    }
    assert!(Half(0x7c00).to_f32().is_infinite());
    assert!(Half(0x7e00).to_f32().is_nan());
}

/// The bit map, pinned: every decoded field, and the bits that must be zero.
#[test]
fn every_sampler_bit_is_decoded_or_required_zero() {
    let decoded = 2
        | (3 << 3)
        | (1 << 6)
        | (1 << 9)
        | (2 << 11)
        | (2 << 13)
        | (1 << 15)
        | (3 << 16)
        | (4 << 20)
        | (0x4080 << 24)
        | (0x4200 << 40)
        | (1 << 56)
        | (2 << 58);
    let expected = SamplerState {
        address_s: AddressMode::Repeat,
        address_t: AddressMode::MirroredRepeat,
        address_r: AddressMode::ClampToEdge,
        mag_filter: Filter::Linear,
        min_filter: Filter::Bicubic,
        mip_filter: MipFilter::Linear,
        coordinates: Coordinates::Pixel,
        compare: CompareFunction::Greater,
        max_anisotropy: 5,
        lod_min_clamp: Half(0x4080),
        lod_max_clamp: Half(0x4200),
        border_color: BorderColor::OpaqueBlack,
        reduction: Reduction::Maximum,
        lod_bias: Half(0x3c00),
    };
    assert_eq!(SamplerState::from_words([decoded, 0x3c00]), Some(expected));
    for bit in 60..63 {
        assert_eq!(
            SamplerState::from_words([decoded | (1 << bit), 0x3c00]),
            None
        );
    }
    for bit in 16..64 {
        assert_eq!(
            SamplerState::from_words([decoded, 0x3c00 | (1 << bit)]),
            None
        );
    }
    for (shift, width, code) in [
        (0, 0x7_u64, 4_u64),
        (3, 0x7, 5),
        (6, 0x7, 7),
        (9, 0x3, 3),
        (11, 0x3, 3),
        (13, 0x3, 3),
        (16, 0xf, 9),
        (56, 0x3, 3),
        (58, 0x3, 3),
    ] {
        let word = (decoded & !(width << shift)) | (code << shift);
        assert_eq!(
            SamplerState::from_words([word, 0x3c00]),
            None,
            "shift {shift}"
        );
    }
}

/// Every `constexpr sampler` option as the Metal frontend encodes it (Apple metal version
/// 32023.883, one kernel per option; the words from the old reflection test suite).
#[test]
fn every_constexpr_sampler_option_decodes_to_what_the_frontend_emits() {
    let default_state = SamplerState {
        address_s: AddressMode::ClampToEdge,
        address_t: AddressMode::ClampToEdge,
        address_r: AddressMode::ClampToEdge,
        mag_filter: Filter::Nearest,
        min_filter: Filter::Nearest,
        mip_filter: MipFilter::None,
        coordinates: Coordinates::Normalized,
        compare: CompareFunction::Never,
        max_anisotropy: 1,
        lod_min_clamp: Half(0),
        lod_max_clamp: Half(0x7bff),
        border_color: BorderColor::TransparentBlack,
        reduction: Reduction::WeightedAverage,
        lod_bias: Half(0),
    };
    type Row = (&'static str, u64, u64, fn(&mut SamplerState));
    let rows: &[Row] = &[
        ("", 0x007b_ff00_0008_0049, 0, |_| {}),
        ("coord::pixel", 0x007b_ff00_0008_8049, 0, |s| {
            s.coordinates = Coordinates::Pixel
        }),
        ("filter::linear", 0x007b_ff00_0008_0a49, 0, |s| {
            s.mag_filter = Filter::Linear;
            s.min_filter = Filter::Linear;
        }),
        ("filter::bicubic", 0x007b_ff00_0008_1449, 0, |s| {
            s.mag_filter = Filter::Bicubic;
            s.min_filter = Filter::Bicubic;
        }),
        ("mag_filter::linear", 0x007b_ff00_0008_0249, 0, |s| {
            s.mag_filter = Filter::Linear
        }),
        ("address::clamp_to_zero", 0x007b_ff00_0008_0000, 0, |s| {
            s.address_s = AddressMode::ClampToZero;
            s.address_t = AddressMode::ClampToZero;
            s.address_r = AddressMode::ClampToZero;
        }),
        ("address::repeat", 0x007b_ff00_0008_0092, 0, |s| {
            s.address_s = AddressMode::Repeat;
            s.address_t = AddressMode::Repeat;
            s.address_r = AddressMode::Repeat;
        }),
        ("address::mirrored_repeat", 0x007b_ff00_0008_00db, 0, |s| {
            s.address_s = AddressMode::MirroredRepeat;
            s.address_t = AddressMode::MirroredRepeat;
            s.address_r = AddressMode::MirroredRepeat;
        }),
        (
            "border_color::opaque_black",
            0x017b_ff00_0008_0000,
            0,
            |s| {
                s.address_s = AddressMode::ClampToBorder;
                s.address_t = AddressMode::ClampToBorder;
                s.address_r = AddressMode::ClampToBorder;
                s.border_color = BorderColor::OpaqueBlack;
            },
        ),
        (
            "border_color::opaque_white",
            0x027b_ff00_0008_0000,
            0,
            |s| {
                s.address_s = AddressMode::ClampToBorder;
                s.address_t = AddressMode::ClampToBorder;
                s.address_r = AddressMode::ClampToBorder;
                s.border_color = BorderColor::OpaqueWhite;
            },
        ),
        (
            "s repeat, t mirrored_repeat, r clamp_to_zero",
            0x007b_ff00_0008_001a,
            0,
            |s| {
                s.address_s = AddressMode::Repeat;
                s.address_t = AddressMode::MirroredRepeat;
                s.address_r = AddressMode::ClampToZero;
            },
        ),
        ("mip_filter::nearest", 0x007b_ff00_0008_2049, 0, |s| {
            s.mip_filter = MipFilter::Nearest
        }),
        ("mip_filter::linear", 0x007b_ff00_0008_4049, 0, |s| {
            s.mip_filter = MipFilter::Linear
        }),
        ("max_anisotropy(5)", 0x007b_ff00_0048_0049, 0, |s| {
            s.max_anisotropy = 5
        }),
        ("max_anisotropy(16)", 0x007b_ff00_00f8_0049, 0, |s| {
            s.max_anisotropy = 16
        }),
        ("lod_clamp(2.0f, 3.0f)", 0x0042_0040_0008_0049, 0, |s| {
            s.lod_min_clamp = Half(0x4000);
            s.lod_max_clamp = Half(0x4200);
        }),
        ("lod_clamp(3.7f, 4.0f)", 0x0044_0043_6608_0049, 0, |s| {
            s.lod_min_clamp = Half(0x4366);
            s.lod_max_clamp = Half(0x4400);
        }),
        ("compare_func::less", 0x007b_ff00_0001_0049, 0, |s| {
            s.compare = CompareFunction::Less
        }),
        ("compare_func::less_equal", 0x007b_ff00_0002_0049, 0, |s| {
            s.compare = CompareFunction::LessEqual
        }),
        ("compare_func::greater", 0x007b_ff00_0003_0049, 0, |s| {
            s.compare = CompareFunction::Greater
        }),
        (
            "compare_func::greater_equal",
            0x007b_ff00_0004_0049,
            0,
            |s| s.compare = CompareFunction::GreaterEqual,
        ),
        ("compare_func::equal", 0x007b_ff00_0005_0049, 0, |s| {
            s.compare = CompareFunction::Equal
        }),
        ("compare_func::not_equal", 0x007b_ff00_0006_0049, 0, |s| {
            s.compare = CompareFunction::NotEqual
        }),
        ("compare_func::always", 0x007b_ff00_0007_0049, 0, |s| {
            s.compare = CompareFunction::Always
        }),
        ("lod_clamp(0.0f, 0.0f)", 0x0000_0000_0008_0049, 0, |s| {
            s.lod_max_clamp = Half(0)
        }),
        ("bias(2.0f)", 0x007b_ff00_0008_0049, 0x4000, |s| {
            s.lod_bias = Half(0x4000)
        }),
        ("bias(-0.5f)", 0x007b_ff00_0008_0049, 0xb800, |s| {
            s.lod_bias = Half(0xb800)
        }),
    ];
    for (options, word, second, apply) in rows {
        let mut expected = default_state;
        apply(&mut expected);
        assert_eq!(
            SamplerState::from_words([*word, *second]),
            Some(expected),
            "constexpr sampler s({options}) emits {word:#018x}, {second:#018x}",
        );
    }
}

/// Slot `index`, one wide.
fn reach_slot(index: u32) -> Slot {
    Slot {
        index: AbiInt::Const(index),
        count: AbiInt::Const(1),
    }
}

/// An object at slot `index`.
fn reach_object(index: u32) -> ObjectArg {
    ObjectArg {
        location: reach_slot(index),
        access: None,
    }
}

/// A read-only device buffer at slot `index`, of struct layout `layout`.
fn reach_buffer(index: u32, layout: Option<u32>) -> BufferArg {
    BufferArg {
        location: reach_slot(index),
        access: Access::Read,
        space: AddrSpace::Device,
        buffer_size: None,
        layout: layout.map(StructRef),
        raster_order_group: None,
    }
}

/// A texture at slot `index` declared with `access`.
fn reach_texture(index: u32, access: Access) -> ParamRole {
    ParamRole::Texture(TextureArg {
        location: reach_slot(index),
        access,
        raster_order_group: None,
    })
}

/// A struct member of kind `kind`, of struct layout `layout`.
fn reach_member(layout: Option<u32>, kind: MemberKind) -> Member {
    Member {
        offset: 0,
        size: 8,
        array_len: 0,
        layout: layout.map(StructRef),
        kind,
        render_target: None,
        raster_order_group: None,
        type_name: DiagName::new("m"),
        name: DiagName::new("m"),
        key: UserKey(b"m".as_slice().into()),
    }
}

/// An argument-buffer member of role `role`.
fn reach_indirect(role: ParamRole) -> MemberKind {
    MemberKind::Indirect(IndirectArg {
        ordinal: 0,
        role,
        ty: ArgTypeInfo::default(),
        name: None,
    })
}

/// A kernel taking parameters of `roles`, in order.
fn reach_entry(roles: Vec<ParamRole>) -> EntryPoint {
    EntryPoint {
        func: FuncId::new(0),
        stage: Stage::Kernel {
            max_total_threads: None,
        },
        params: roles
            .into_iter()
            .enumerate()
            .map(|(param, role)| ParamBinding {
                param: param as u32,
                role,
                unused: false,
                predicate: None,
                ty: ArgTypeInfo::default(),
                name: None,
            })
            .collect(),
        outputs: Vec::new(),
        name: DiagName::new("k"),
        user_annotation: None,
    }
}

/// An entry reaches an indirect command buffer through a `command_buffer` parameter, an argument
/// buffer's member, or one at any depth below it: a pointer member's layout, a struct member's; a
/// layout that refers to itself ends the walk. A table member does not count, nor does a command
/// buffer in a layout no buffer of the entry takes.
#[test]
fn an_entry_reaches_a_command_buffer_through_its_arguments_at_any_depth() {
    let structs = vec![
        // 0: a command buffer member.
        StructInfo {
            members: vec![reach_member(
                None,
                reach_indirect(ParamRole::CommandBuffer(reach_object(0))),
            )],
        },
        // 1: a pointer member to layout 0.
        StructInfo {
            members: vec![reach_member(
                None,
                reach_indirect(ParamRole::ArgumentBuffer(reach_buffer(0, Some(0)))),
            )],
        },
        // 2: a struct member of layout 0.
        StructInfo {
            members: vec![reach_member(
                Some(0),
                MemberKind::IndirectStruct { first_id: 0 },
            )],
        },
        // 3: a table member, and a pointer member to itself.
        StructInfo {
            members: vec![
                reach_member(
                    None,
                    reach_indirect(ParamRole::FunctionTable(reach_object(0))),
                ),
                reach_member(
                    None,
                    reach_indirect(ParamRole::ArgumentBuffer(reach_buffer(1, Some(3)))),
                ),
            ],
        },
    ];
    let out = ParamRole::Buffer(reach_buffer(0, None));
    for (roles, reaches) in [
        (
            vec![out.clone(), ParamRole::CommandBuffer(reach_object(1))],
            true,
        ),
        (
            vec![
                out.clone(),
                ParamRole::ArgumentBuffer(reach_buffer(1, Some(0))),
            ],
            true,
        ),
        (vec![ParamRole::Buffer(reach_buffer(1, Some(1)))], true),
        (
            vec![ParamRole::ArgumentBuffer(reach_buffer(1, Some(2)))],
            true,
        ),
        (
            vec![
                out.clone(),
                ParamRole::ArgumentBuffer(reach_buffer(1, Some(3))),
            ],
            false,
        ),
        (
            vec![out.clone(), ParamRole::FunctionTable(reach_object(1))],
            false,
        ),
        (vec![out], false),
    ] {
        let ep = reach_entry(roles);
        assert_eq!(
            ep.reaches_command_buffers(&structs),
            reaches,
            "{:?}",
            ep.params
        );
    }
}

/// An entry takes a texture it may write through a parameter declared `access::write` or
/// `access::read_write`, or such a member of an argument buffer at any depth (a pointer member's
/// layout, a struct member's); a texture declared `access::read` or `access::sample` does not
/// count, nor does a writable one in a layout no buffer of the entry takes.
#[test]
fn an_entry_takes_a_writable_texture_through_its_arguments_at_any_depth() {
    let structs = vec![
        // 0: a writable texture member.
        StructInfo {
            members: vec![reach_member(
                None,
                reach_indirect(reach_texture(0, Access::ReadWrite)),
            )],
        },
        // 1: a pointer member to layout 0.
        StructInfo {
            members: vec![reach_member(
                None,
                reach_indirect(ParamRole::ArgumentBuffer(reach_buffer(0, Some(0)))),
            )],
        },
        // 2: a struct member of layout 0.
        StructInfo {
            members: vec![reach_member(
                Some(0),
                MemberKind::IndirectStruct { first_id: 0 },
            )],
        },
        // 3: read and sampled texture members, and a pointer member to itself.
        StructInfo {
            members: vec![
                reach_member(None, reach_indirect(reach_texture(0, Access::Read))),
                reach_member(None, reach_indirect(reach_texture(1, Access::Sample))),
                reach_member(
                    None,
                    reach_indirect(ParamRole::ArgumentBuffer(reach_buffer(1, Some(3)))),
                ),
            ],
        },
    ];
    let out = ParamRole::Buffer(reach_buffer(0, None));
    for (roles, takes) in [
        (vec![out.clone(), reach_texture(0, Access::Write)], true),
        (vec![out.clone(), reach_texture(0, Access::ReadWrite)], true),
        (vec![out.clone(), reach_texture(0, Access::Read)], false),
        (vec![out.clone(), reach_texture(0, Access::Sample)], false),
        (
            vec![
                out.clone(),
                ParamRole::ArgumentBuffer(reach_buffer(1, Some(0))),
            ],
            true,
        ),
        (vec![ParamRole::Buffer(reach_buffer(1, Some(1)))], true),
        (
            vec![ParamRole::ArgumentBuffer(reach_buffer(1, Some(2)))],
            true,
        ),
        (
            vec![
                out.clone(),
                ParamRole::ArgumentBuffer(reach_buffer(1, Some(3))),
            ],
            false,
        ),
        (vec![out], false),
    ] {
        let ep = reach_entry(roles);
        assert_eq!(
            ep.takes_writable_textures(&structs),
            takes,
            "{:?}",
            ep.params
        );
    }
}
