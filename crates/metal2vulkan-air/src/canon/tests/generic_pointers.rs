//! MSL 4.1's generic pointers become pointers of the one space every pointer they may hold comes
//! from, their space queries answered; a generic pointer of two spaces, or of one only the run
//! knows, is refused as `AddressSpace(8)`.

use super::*;

/// What [`queried`]'s generic pointer `p` holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// `in + 1`, made generic, then offset.
    Device,
    /// `in` or `in + 2` as word 0 of `in` decides: two device pointers.
    Joined,
    /// `in` or null as word 0 of `in` decides.
    Nullable,
    /// `in` or the thread's own array as word 0 of `in` decides.
    Mixed,
    /// A generic pointer stored to the thread's array and loaded back.
    Loaded,
}

/// `kernel k(device float* out, device float* in)` with a generic pointer `p` that `held` gives:
/// `out[0] = *p`, `out[1] = is_device_pointer(p)`, `out[2] = is_thread_pointer(p)`.
fn queried(held: Held) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let f32t = m.float();
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let generic = m.ptr(AddrSpace::Generic);
    let words = m.array(f32t, 4);
    let (zero, one, two) = (
        m.const_int(i32t, 0),
        m.const_int(i32t, 1),
        m.const_int(i32t, 2),
    );
    let (yes, no) = (
        Value::Const(m.constant(f32t, Const::Float(1.0f32.to_bits().into()))),
        Value::Const(m.constant(f32t, Const::Float(0.0f32.to_bits().into()))),
    );
    let null = Value::Const(m.constant(generic, Const::Null));
    let query_ty = m.fn_ty(i1, &[generic]);
    let queries = [AddrSpace::Device, AddrSpace::Thread].map(|space| {
        let name = format!(
            "air.is_{}_pointer",
            if space == AddrSpace::Device {
                "device"
            } else {
                "thread"
            }
        );
        let decl = m.declare(&name, query_ty);
        let op = AirIntrinsic::parse(&name).expect("a space query");
        assert_eq!(
            op,
            AirIntrinsic::PointerSpace(crate::intrinsics::PointerSpace { space })
        );
        Callee::Air { decl, op }
    });
    let fty = m.fn_ty(void, &[dev, dev]);
    let mut f = m.define("k", fty);
    let local = f.push(
        thread,
        InstKind::Alloca {
            ty: words,
            count: one,
            align: None,
        },
    );
    let cast = |f: &mut FunctionBuilder<'_>, v: Value| {
        f.push(
            generic,
            InstKind::Cast {
                op: CastOp::AddrSpaceCast,
                v,
            },
        )
    };
    let at = |f: &mut FunctionBuilder<'_>, base: Value, k: Value| {
        f.push(
            generic,
            InstKind::Gep {
                src_ty: f32t,
                base,
                indices: Box::new([k]),
                inbounds: true,
            },
        )
    };
    let pick = |f: &mut FunctionBuilder<'_>, t: Value, e: Value| {
        let word = load(f, i32t, arg(1));
        let c = f.push(
            i1,
            InstKind::ICmp {
                pred: IntPredicate::Ne,
                a: word,
                b: zero,
            },
        );
        f.push(
            generic,
            InstKind::Select {
                c,
                t,
                f: e,
                fmf: FastMath::NONE,
            },
        )
    };
    let input = cast(&mut f, arg(1));
    let p = match held {
        Held::Device => at(&mut f, input, one),
        Held::Joined => {
            let other = at(&mut f, input, two);
            pick(&mut f, input, other)
        }
        Held::Nullable => pick(&mut f, input, null),
        Held::Mixed => {
            let mine = cast(&mut f, local);
            pick(&mut f, input, mine)
        }
        Held::Loaded => {
            store(&mut f, input, local);
            load(&mut f, generic, local)
        }
    };
    let x = load(&mut f, f32t, p);
    store(&mut f, x, arg(0));
    for (k, callee) in queries.into_iter().enumerate() {
        let b = call(&mut f, i1, callee, &[p]);
        let v = f.push(
            f32t,
            InstKind::Select {
                c: b,
                t: yes,
                f: no,
                fmf: FastMath::NONE,
            },
        );
        let k = f.module().const_int(i32t, k as u64 + 1);
        let o = f.push(
            dev,
            InstKind::Gep {
                src_ty: f32t,
                base: arg(0),
                indices: Box::new([k]),
                inbounds: true,
            },
        );
        store(&mut f, v, o);
    }
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0), buffer(1, 1)])], vec![]);
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// Whether any instruction of the specialized entry names the generic space in its type or in an
/// operand's.
fn names_generic(e: &Entry) -> bool {
    let types = e.module.types();
    let body = e.module.func(e.func).unwrap().body.as_ref().unwrap();
    let generic = |t: Option<TypeId>| {
        matches!(
            t.and_then(|t| types.get(t)),
            Some(Type::Ptr {
                space: AddrSpace::Generic
            })
        )
    };
    body.insts.values().any(|inst| {
        let mut any = generic(Some(inst.ty));
        inst.kind.for_each_value(|v| {
            any |= match v {
                Value::Inst(i) => generic(body.insts.get(i).map(|x| x.ty)),
                Value::Const(c) => generic(e.module.consts().ty(c)),
                Value::Arg(_) => false,
            };
        });
        any
    })
}

/// The constant floats the entry stores, in order (`None` for a value the run computes).
fn stored(e: &Entry) -> Vec<Option<f32>> {
    let body = e.module.func(e.func).unwrap().body.as_ref().unwrap();
    body.insts
        .values()
        .filter_map(|inst| match inst.kind {
            InstKind::Store { val, .. } => Some(match val {
                Value::Const(c) => match e.module.consts().get(c).map(|k| &k.value) {
                    Some(Const::Float(bits)) => Some(f32::from_bits(*bits as u32)),
                    _ => None,
                },
                _ => None,
            }),
            _ => None,
        })
        .collect()
}

fn refused(e: Result<Entry, Error>) -> bool {
    matches!(&e, Err(Error::Unsupported(u)) if u.feature == Feature::AddressSpace(8))
}

#[test]
fn a_generic_pointer_of_one_space_is_that_space_s_pointer() {
    for held in [Held::Device, Held::Joined] {
        let e = queried(held).unwrap_or_else(|e| panic!("{held:?}: {e}"));
        assert!(!names_generic(&e), "{held:?}: {}", body_text(&e));
        // `*p` is read; the queries answer device, not thread.
        assert_eq!(
            stored(&e)[1..],
            [Some(1.0), Some(0.0)],
            "{held:?}: {}",
            body_text(&e)
        );
    }
}

#[test]
fn a_query_of_a_pointer_a_null_may_reach_asks_whether_it_is_null() {
    let e = queried(Held::Nullable).unwrap_or_else(|e| panic!("{e}"));
    assert!(!names_generic(&e), "{}", body_text(&e));
    let text = body_text(&e);
    assert!(text.contains("icmp ne ptr addrspace(1)"), "{text}");
    // The device query is the run's; the thread query is false.
    assert_eq!(stored(&e)[1..], [None, Some(0.0)], "{text}");
}

#[test]
fn a_generic_pointer_of_two_spaces_or_of_memory_is_refused() {
    assert!(refused(queried(Held::Mixed)));
    assert!(refused(queried(Held::Loaded)));
}
