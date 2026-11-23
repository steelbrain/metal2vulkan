//! Program-scope buffer bindings become buffer parameters: each load of the pointer a bound
//! buffer's global holds is a read of the entry's own parameter of that buffer, or of one
//! appended to it, one per role and type; what stays a read of a bound global is refused.

use super::*;
use crate::abi::GlobalBinding;
use crate::error::Feature;
use crate::ir::{Type, TypeId};

/// A global as Apple's compiler writes one for a program-scope binding: a constant of the constant
/// space holding a pointer, externally initialized and `undef`.
fn bound_global(m: &mut ModuleBuilder, name: &str) -> GlobalId {
    let ptr = m.ptr(AddrSpace::Constant);
    let undef = m.constant(ptr, Const::Undef);
    let name = m.sym(name);
    m.global(Global {
        name,
        value_ty: ptr,
        space: AddrSpace::Constant,
        constant: true,
        init: Some(undef),
        externally_initialized: true,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: None,
        section: None,
    })
}

/// A binding of `global` with `role`, named `name`.
fn binding(global: GlobalId, role: ParamRole, name: &str) -> GlobalBinding {
    GlobalBinding {
        global,
        role,
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new(name)),
    }
}

/// [`buffer`]'s role at `index` in `space` with `access`.
fn buffer_role(index: u32, space: AddrSpace, access: Access) -> ParamRole {
    match buffer(0, index).role {
        ParamRole::Buffer(arg) => ParamRole::Buffer(BufferArg {
            space,
            access,
            ..arg
        }),
        _ => unreachable!(),
    }
}

/// `kernel k(device i32* out [[buffer(0)]])` reading three bound globals: `c`, a `constant i32 *`,
/// `d` and `o`, `device i32 *`s. It stores `*c + c[1]` (`c` loaded twice) to `*d`, `*c` to `*o`
/// and `c[1]` to `out`, then what `extra` adds over the globals' addresses; `roles` binds the
/// globals.
fn entry_with(
    extra: impl FnOnce(&mut FunctionBuilder<'_>, [Value; 3]),
    roles: impl FnOnce([GlobalId; 3]) -> Vec<GlobalBinding>,
) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let globals = [
        bound_global(&mut m, "c"),
        bound_global(&mut m, "d"),
        bound_global(&mut m, "o"),
    ];
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let cst = m.ptr(AddrSpace::Constant);
    let fty = m.fn_ty(void, &[dev]);
    let addrs = globals.map(|g| m.global_addr(g));
    let one = {
        let i64t = m.int(64);
        m.const_int(i64t, 1)
    };
    let mut f = m.define("k", fty);
    let c = load(&mut f, cst, addrs[0]);
    let x = load(&mut f, i32t, c);
    let c2 = load(&mut f, cst, addrs[0]);
    let at = f.push(
        cst,
        InstKind::Gep {
            src_ty: i32t,
            base: c2,
            indices: vec![one].into(),
            inbounds: true,
        },
    );
    let y = load(&mut f, i32t, at);
    let sum = f.bin(BinOp::Add, x, y);
    let d = load(&mut f, dev, addrs[1]);
    store(&mut f, sum, d);
    let o = load(&mut f, dev, addrs[2]);
    store(&mut f, x, o);
    store(&mut f, y, arg(0));
    extra(&mut f, addrs);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let mut abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    abi.global_bindings = roles(globals);
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// The three bindings [`entry_with`] describes.
fn standard([c, d, o]: [GlobalId; 3]) -> Vec<GlobalBinding> {
    vec![
        binding(c, buffer_role(1, AddrSpace::Constant, Access::Read), "c"),
        binding(d, buffer_role(3, AddrSpace::Device, Access::ReadWrite), "d"),
        binding(o, buffer(0, 0).role, "o"),
    ]
}

fn refusal(e: Result<Entry, Error>) -> Feature {
    match e {
        Err(Error::Unsupported(u)) => u.feature,
        other => panic!("not a refusal: {other:?}"),
    }
}

/// `c`'s two loads read one appended `constant` parameter at `buffer(1)`, `d`'s an appended
/// `device` one at `buffer(3)`, in the order the body first loads them, and `o`'s the entry's own
/// `buffer(0)`, which binds the same role and type; no global is left.
#[test]
fn bound_buffers_become_buffer_parameters() {
    let e = entry_with(|_, _| {}, standard).unwrap();
    let slots: Vec<_> =
        e.ep.params
            .iter()
            .map(|p| match &p.role {
                ParamRole::Buffer(b) => (p.param, b.location.index, b.space),
                other => panic!("{other:?}"),
            })
            .collect();
    assert_eq!(
        slots,
        [
            (0, AbiInt::Const(0), AddrSpace::Device),
            (1, AbiInt::Const(1), AddrSpace::Constant),
            (2, AbiInt::Const(3), AddrSpace::Device),
        ]
    );
    let f = e.module.func(e.func).unwrap();
    let Some(Type::Function { params, .. }) = e.module.types().get(f.ty) else {
        panic!("the entry's type is not a function");
    };
    assert_eq!(params.len(), 3);
    assert_eq!(e.module.globals().len(), 0, "{}", body_text(&e));
    let text = body_text(&e);
    assert!(text.contains("%a1"), "{text}");
    assert!(text.contains("%a2"), "{text}");
}

/// Two bound buffers of two roles at one slot (`d` as a `device` buffer at `c`'s `buffer(1)`), and
/// a bound buffer at the slot of an entry parameter of another role (`o` read-only at the entry's
/// `buffer(0)`), are refused: Metal reads the one buffer bound there through both.
#[test]
fn a_bound_buffer_at_another_argument_s_slot_is_refused() {
    let two = |[c, d, o]: [GlobalId; 3]| {
        let mut b = standard([c, d, o]);
        b[1].role = buffer_role(1, AddrSpace::Device, Access::ReadWrite);
        b
    };
    assert_eq!(
        refusal(entry_with(|_, _| {}, two)),
        Feature::ProgramScopeSharedSlot
    );
    let param = |[c, d, o]: [GlobalId; 3]| {
        let mut b = standard([c, d, o]);
        b[2].role = buffer_role(0, AddrSpace::Device, Access::Read);
        b
    };
    assert_eq!(
        refusal(entry_with(|_, _| {}, param)),
        Feature::ProgramScopeSharedSlot
    );
}

/// A read of a bound buffer's global other than a load of its pointer (here its bits as an
/// `i64`) is refused, as is any read of threadgroup memory bound at program scope.
#[test]
fn a_bound_global_read_otherwise_is_refused() {
    let bits = |f: &mut FunctionBuilder<'_>, [c, _, _]: [Value; 3]| {
        let i64t = f.module().int(64);
        let v = load(f, i64t, c);
        let i32t = f.module().int(32);
        let v = f.push(
            i32t,
            InstKind::Cast {
                op: CastOp::Trunc,
                v,
            },
        );
        store(f, v, arg(0));
    };
    assert_eq!(
        refusal(entry_with(bits, standard)),
        Feature::ProgramScopeRead
    );
    let threadgroup = |[c, d, o]: [GlobalId; 3]| {
        let mut b = standard([c, d, o]);
        b[1].role = match buffer_role(0, AddrSpace::Threadgroup, Access::ReadWrite) {
            ParamRole::Buffer(arg) => ParamRole::Threadgroup(arg),
            _ => unreachable!(),
        };
        b
    };
    assert_eq!(
        refusal(entry_with(|_, _| {}, threadgroup)),
        Feature::ProgramScopeResource
    );
}
