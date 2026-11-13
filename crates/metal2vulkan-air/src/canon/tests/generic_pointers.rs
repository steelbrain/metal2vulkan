//! MSL 4.1's generic pointers become pointers of the one space every pointer they may hold comes
//! from, their space queries answered; a generic pointer of two spaces, or of one only the run
//! knows, is refused as `GenericPointer`, naming how and at which instruction.

use super::*;
use crate::error::{DiagName, GenericPointer};

/// What [`queried`]'s generic pointer `p` holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// `in + 1`, made generic, then offset.
    Device,
    /// `in` or `in + 2` as word 0 of `in` decides: two device pointers.
    Joined,
    /// `in` or null as word 0 of `in` decides.
    Nullable,
    /// The thread's own array or null as word 0 of `in` decides; only queried (`out[0] = 0`).
    LocalOrNull,
    /// Null or the thread's own array as word 0 of `in` decides; only queried (`out[0] = 0`).
    NullOrLocal,
    /// `in` or the thread's own array as word 0 of `in` decides.
    Mixed,
    /// `in`, made generic, stored to the thread's array and loaded back.
    Loaded,
    /// As `Loaded`, stored twice: no one store decides the load (`canon::forward`).
    LoadedTwice,
    /// As `Loaded`, `in` or null as word 0 of `in` decides.
    LoadedNullable,
    /// `in` stored at byte 0 of the thread's array and another array's address at byte 8, and
    /// byte 0 loaded back.
    TwoSlots,
    /// As `TwoSlots`, loaded back at the float `in[0]` names: either slot.
    AnySlot,
    /// As `Loaded`, the array's float 0 written over it before the load.
    Overwritten,
    /// As `Loaded`, the array's address written to `out[3]` as an integer.
    Escaped,
    /// Field 0 of a `{ptr, i32}` built from `in` and 1.
    Field,
    /// Field 0 of a `{ptr, ptr}` of `in` and the thread's array.
    Fields,
    /// Field 0 of one of two `{ptr, i32}`s, of `in` and of `in + 2`, as word 0 of `in` decides.
    FieldSelect,
    /// As `Field`, of `in` or null as word 0 of `in` decides.
    FieldNullable,
    /// Element 0 of a `[2 x ptr]` of `in` and the thread's array.
    ArrayMixed,
    /// Field 0 of a `{ptr, i32}` loaded from the thread's array.
    FieldLoaded,
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
        Held::LocalOrNull => {
            let mine = cast(&mut f, local);
            pick(&mut f, mine, null)
        }
        Held::NullOrLocal => {
            let mine = cast(&mut f, local);
            pick(&mut f, null, mine)
        }
        Held::Mixed => {
            let mine = cast(&mut f, local);
            pick(&mut f, input, mine)
        }
        Held::Loaded => {
            store(&mut f, input, local);
            load(&mut f, generic, local)
        }
        Held::LoadedTwice => {
            store(&mut f, input, local);
            store(&mut f, input, local);
            load(&mut f, generic, local)
        }
        Held::LoadedNullable => {
            let either = pick(&mut f, input, null);
            store(&mut f, either, local);
            load(&mut f, generic, local)
        }
        Held::TwoSlots | Held::AnySlot => {
            store(&mut f, input, local);
            let other = f.push(
                thread,
                InstKind::Alloca {
                    ty: words,
                    count: one,
                    align: None,
                },
            );
            let theirs = cast(&mut f, other);
            let second = f.push(
                thread,
                InstKind::Gep {
                    src_ty: f32t,
                    base: local,
                    indices: Box::new([two]),
                    inbounds: true,
                },
            );
            store(&mut f, theirs, second);
            let from = if held == Held::AnySlot {
                let word = load(&mut f, i32t, arg(1));
                f.push(
                    thread,
                    InstKind::Gep {
                        src_ty: f32t,
                        base: local,
                        indices: Box::new([word]),
                        inbounds: true,
                    },
                )
            } else {
                local
            };
            load(&mut f, generic, from)
        }
        Held::Overwritten => {
            store(&mut f, input, local);
            store(&mut f, yes, local);
            load(&mut f, generic, local)
        }
        Held::Escaped => {
            store(&mut f, input, local);
            let i64t = f.module().int(64);
            let bits = f.push(
                i64t,
                InstKind::Cast {
                    op: CastOp::PtrToInt,
                    v: local,
                },
            );
            let three = f.module().const_int(i32t, 3);
            let o = f.push(
                dev,
                InstKind::Gep {
                    src_ty: f32t,
                    base: arg(0),
                    indices: Box::new([three]),
                    inbounds: true,
                },
            );
            store(&mut f, bits, o);
            load(&mut f, generic, local)
        }
        Held::Field
        | Held::Fields
        | Held::FieldSelect
        | Held::FieldNullable
        | Held::ArrayMixed
        | Held::FieldLoaded => {
            let pair = f.module().struct_ty(&[generic, i32t]);
            let both = f.module().struct_ty(&[generic, generic]);
            let array = f.module().array(generic, 2);
            // `val` inserted at field or element `k` of `agg`, or of the type's poison.
            let insert = |f: &mut FunctionBuilder<'_>, ty, agg: Option<Value>, val, k: u32| {
                let agg =
                    agg.unwrap_or_else(|| Value::Const(f.module().constant(ty, Const::Poison)));
                f.push(
                    ty,
                    InstKind::InsertValue {
                        agg,
                        val,
                        indices: Box::new([k]),
                    },
                )
            };
            let agg = match held {
                Held::Field => {
                    let s = insert(&mut f, pair, None, input, 0);
                    insert(&mut f, pair, Some(s), one, 1)
                }
                Held::Fields => {
                    let mine = cast(&mut f, local);
                    let s = insert(&mut f, both, None, mine, 1);
                    insert(&mut f, both, Some(s), input, 0)
                }
                Held::FieldSelect => {
                    let other = at(&mut f, input, two);
                    let a = insert(&mut f, pair, None, input, 0);
                    let b = insert(&mut f, pair, None, other, 0);
                    let word = load(&mut f, i32t, arg(1));
                    let c = f.push(
                        i1,
                        InstKind::ICmp {
                            pred: IntPredicate::Ne,
                            a: word,
                            b: zero,
                        },
                    );
                    f.push(
                        pair,
                        InstKind::Select {
                            c,
                            t: a,
                            f: b,
                            fmf: FastMath::NONE,
                        },
                    )
                }
                Held::FieldNullable => {
                    let either = pick(&mut f, input, null);
                    insert(&mut f, pair, None, either, 0)
                }
                Held::ArrayMixed => {
                    let mine = cast(&mut f, local);
                    let s = insert(&mut f, array, None, input, 0);
                    insert(&mut f, array, Some(s), mine, 1)
                }
                _ => load(&mut f, pair, local),
            };
            f.push(
                generic,
                InstKind::ExtractValue {
                    agg,
                    indices: Box::new([0]),
                },
            )
        }
    };
    let x = match held {
        Held::LocalOrNull | Held::NullOrLocal => no,
        _ => load(&mut f, f32t, p),
    };
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
/// operand's, at any depth of an aggregate or a vector.
fn names_generic(e: &Entry) -> bool {
    let types = e.module.types();
    let body = e.module.func(e.func).unwrap().body.as_ref().unwrap();
    fn holds(types: &crate::ir::TypeTable, t: TypeId) -> bool {
        match types.get(t) {
            Some(Type::Ptr { space }) => *space == AddrSpace::Generic,
            Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => holds(types, *elem),
            Some(Type::Struct { fields, .. }) => fields.iter().any(|&f| holds(types, f)),
            _ => false,
        }
    }
    let generic = |t: Option<TypeId>| t.is_some_and(|t| holds(types, t));
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

/// Whether `e` refuses a generic pointer for `how`, at an instruction of the kernel.
fn refused(e: Result<Entry, Error>, how: GenericPointer) -> bool {
    match &e {
        Err(Error::Unsupported(u)) => {
            let at = u.location.as_ref().and_then(|l| l.in_function());
            u.feature == Feature::GenericPointer(how)
                && at.is_some_and(|(f, _, i)| *f == DiagName::new("k") && i.is_some())
        }
        _ => false,
    }
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
    // The device query compares the device pointer with the device null.
    assert!(
        text.lines()
            .any(|l| l.contains("icmp ne") && l.contains("ptr addrspace(1) zeroinitializer")),
        "{text}"
    );
    // The device query is the run's; the thread query is false.
    assert_eq!(stored(&e)[1..], [None, Some(0.0)], "{text}");
}

/// A thread pointer or null, as a condition picks, is not null exactly where the condition picks
/// the thread pointer: the thread query is that condition, and the pointer select, read by the
/// queries alone, goes.
#[test]
fn a_query_of_a_variable_or_null_is_the_condition_that_picks() {
    for held in [Held::LocalOrNull, Held::NullOrLocal] {
        let e = queried(held).unwrap_or_else(|e| panic!("{held:?}: {e}"));
        let text = body_text(&e);
        assert!(!names_generic(&e), "{held:?}: {text}");
        assert!(
            !text
                .lines()
                .any(|l| l.contains("select") && l.contains("ptr")),
            "{held:?}: {text}"
        );
        // The device query is false; the thread query is the run's condition.
        assert_eq!(stored(&e), [Some(0.0), Some(0.0), None], "{held:?}: {text}");
    }
}

#[test]
fn a_generic_pointer_of_two_spaces_or_of_memory_is_refused() {
    assert!(refused(queried(Held::Mixed), GenericPointer::TwoSpaces));
    assert!(refused(queried(Held::AnySlot), GenericPointer::TwoSpaces));
    assert!(refused(queried(Held::Overwritten), GenericPointer::Memory));
    assert!(refused(queried(Held::Escaped), GenericPointer::Memory));
    // An array's elements share one space.
    assert!(refused(
        queried(Held::ArrayMixed),
        GenericPointer::TwoSpaces
    ));
    assert!(refused(queried(Held::FieldLoaded), GenericPointer::Memory));
}

/// A generic pointer an aggregate holds is in the group of what its field takes: a device
/// pointer, through one field of two of different spaces and through a `select` of two
/// aggregates. Where no null reaches the field, the queries are answered; where one may, the
/// device query compares the pointer with the device null.
#[test]
fn a_generic_pointer_an_aggregate_holds_is_its_field_s_space() {
    for (held, device) in [
        (Held::Field, Some(1.0)),
        (Held::Fields, Some(1.0)),
        (Held::FieldSelect, Some(1.0)),
        (Held::FieldNullable, None),
    ] {
        let e = queried(held).unwrap_or_else(|e| panic!("{held:?}: {e}"));
        let text = body_text(&e);
        assert!(!names_generic(&e), "{held:?}: {text}");
        let compared = text
            .lines()
            .any(|l| l.contains("icmp ne") && l.contains("ptr addrspace(1) zeroinitializer"));
        assert_eq!(compared, device.is_none(), "{held:?}: {text}");
        let words = stored(&e);
        assert_eq!(
            words[words.len().saturating_sub(2)..],
            [device, Some(0.0)],
            "{held:?}: {text}"
        );
    }
}

/// A generic pointer loaded from a slot of thread memory whose every access is visible is in the
/// group of the pointers stored there: a device pointer. Where no null is stored there, the
/// queries are answered; where one may be, the device query compares the pointer with the device
/// null. A slot one store writes, ahead of the load, is read as the pointer stored before the space
/// is decided (`canon::forward`), and no load is left; a slot written twice keeps its load, of the
/// device space.
#[test]
fn a_generic_pointer_loaded_from_a_thread_slot_is_its_stores_space_s() {
    for (held, device, kept) in [
        (Held::Loaded, Some(1.0), false),
        (Held::LoadedTwice, Some(1.0), true),
        (Held::TwoSlots, Some(1.0), false),
        (Held::LoadedNullable, None, false),
    ] {
        let e = queried(held).unwrap_or_else(|e| panic!("{held:?}: {e}"));
        let text = body_text(&e);
        assert!(!names_generic(&e), "{held:?}: {text}");
        assert_eq!(
            text.contains("load ptr addrspace(1),"),
            kept,
            "{held:?}: {text}"
        );
        let compared = text
            .lines()
            .any(|l| l.contains("icmp ne") && l.contains("ptr addrspace(1) zeroinitializer"));
        assert_eq!(compared, device.is_none(), "{held:?}: {text}");
        // The queries' words come last, after the slots' stores and `*p`'s.
        let words = stored(&e);
        assert_eq!(
            words[words.len().saturating_sub(2)..],
            [device, Some(0.0)],
            "{held:?}: {text}"
        );
    }
}
