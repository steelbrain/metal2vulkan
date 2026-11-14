//! Whether an entry may encode indirect commands: where its metadata names a command buffer it
//! takes, and where its body calls an indirect-command operation, whatever pointer the operation's
//! command operand is, or makes a call the walk cannot see into.

use super::*;
use crate::abi::ObjectArg;
use crate::intrinsics::IndirectCommand;

/// `kernel k(device void* data [[buffer(0)]], device void* out [[buffer(1)]])`, its body built by
/// `body` (before the `ret`), and its entry point taking `params`.
pub(super) fn module(
    params: Vec<ParamBinding>,
    body: impl FnOnce(&mut FunctionBuilder<'_>),
) -> (Module, EntryPoint) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(void, &[dev, dev]);
    let mut f = m.define("k", fty);
    body(&mut f);
    f.ret(None);
    let k = f.finish();
    (m.finish().unwrap(), kernel(k, params))
}

/// [`module`] with plain buffers, specialized.
fn specialized(body: impl FnOnce(&mut FunctionBuilder<'_>)) -> Entry {
    let (module, ep) = module(vec![buffer(0, 0), buffer(1, 1)], body);
    let abi = abi(vec![ep], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter()).unwrap()
}

/// `air.draw_primitives_render_command(cmd, 0, 3, 0, 3, 1, 0)`: command 0 of `cmd` draws one
/// triangle.
fn draw(f: &mut FunctionBuilder<'_>, cmd: Value) {
    let name = "air.draw_primitives_render_command";
    let void = f.module().void();
    let i32t = f.module().int(32);
    let dev = f.module().ptr(AddrSpace::Device);
    let ty = f
        .module()
        .fn_ty(void, &[dev, i32t, i32t, i32t, i32t, i32t, i32t]);
    let decl = f.module().declare(name, ty);
    let op = AirIntrinsic::parse(name).unwrap();
    assert_eq!(
        op,
        AirIntrinsic::IndirectCommand(IndirectCommand::DrawPrimitives)
    );
    let [zero, one, three] = [0, 1, 3].map(|v| f.module().const_int(i32t, v));
    let args = [cmd, zero, three, zero, three, one, zero];
    void_call(f, Callee::Air { decl, op }, &args);
}

pub(super) fn void_call(f: &mut FunctionBuilder<'_>, callee: Callee, args: &[Value]) {
    f.push_void(InstKind::Call(Box::new(Call {
        callee,
        args: args.into(),
        fmf: FastMath::NONE,
        tail: TailKind::None,
        attrs: None,
        range: None,
    })));
}

/// The pointer at byte 16 of `data`, a member no layout of the entry describes.
pub(super) fn loaded_handle(f: &mut FunctionBuilder<'_>) -> Value {
    let i8t = f.module().int(8);
    let i64t = f.module().int(64);
    let dev = f.module().ptr(AddrSpace::Device);
    let sixteen = f.module().const_int(i64t, 16);
    let at = f.push(
        dev,
        InstKind::Gep {
            src_ty: i8t,
            base: arg(0),
            indices: Box::new([sixteen]),
            inbounds: true,
        },
    );
    load(f, dev, at)
}

/// The pointer the first word of `data` holds as an integer.
pub(super) fn handle_from_integer(f: &mut FunctionBuilder<'_>) -> Value {
    let i64t = f.module().int(64);
    let dev = f.module().ptr(AddrSpace::Device);
    let word = load(f, i64t, arg(0));
    f.push(
        dev,
        InstKind::Cast {
            op: CastOp::IntToPtr,
            v: word,
        },
    )
}

/// A command buffer the metadata names counts whatever the body does with it; a kernel that takes
/// none and calls no operation does not.
#[test]
fn a_command_buffer_the_metadata_names_counts_untouched() {
    let command_buffer = ParamBinding {
        role: ParamRole::CommandBuffer(ObjectArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(1),
            },
            access: None,
        }),
        ..buffer(0, 0)
    };
    for (first, may) in [(command_buffer, true), (buffer(0, 0), false)] {
        let (module, ep) = module(vec![first, buffer(1, 1)], |f| {
            let i32t = f.module().int(32);
            let one = f.module().const_int(i32t, 1);
            store(f, one, arg(1));
        });
        let abi = abi(vec![ep], Vec::new());
        let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
        assert_eq!(e.may_encode_indirect_commands(), may);
    }
}

/// An operation on a command operand the metadata does not name counts, whatever the pointer: a
/// buffer's own address cast to a command buffer, a handle loaded from a member no layout
/// describes, a handle made from an integer. The same pointers stored rather than encoded through
/// reach no command buffer's records, which only an operation reaches.
#[test]
fn an_operation_on_a_cast_command_buffer_counts() {
    type Shape = fn(&mut FunctionBuilder<'_>) -> Value;
    let shapes: [(&str, Shape); 3] = [
        ("the buffer's own address", |_| arg(0)),
        ("a handle loaded from a member", loaded_handle),
        ("a handle made from an integer", handle_from_integer),
    ];
    for (what, shape) in shapes {
        let encodes = specialized(|f| {
            let cmd = shape(f);
            draw(f, cmd);
        });
        assert!(
            !encodes.ep.reaches_command_buffers(&encodes.structs),
            "{what}"
        );
        assert!(encodes.may_encode_indirect_commands(), "{what}");
        let stores = specialized(|f| {
            let cmd = shape(f);
            store(f, cmd, arg(1));
        });
        assert!(!stores.may_encode_indirect_commands(), "{what}");
    }
}

/// A call the walk cannot see into counts: an indirect call, an intrinsic outside the closed set, a
/// call to a declaration. Canonicalization refuses each, so the walk is run over the module as
/// built; a defined function's body is walked as the entry's is.
#[test]
fn a_call_the_walk_cannot_see_into_counts() {
    type Body = fn(&mut FunctionBuilder<'_>);
    let bodies: [(&str, Body, bool); 5] = [
        (
            "an indirect call",
            |f| {
                let void = f.module().void();
                let fn_ty = f.module().fn_ty(void, &[]);
                let callee = loaded_handle(f);
                void_call(f, Callee::Indirect { callee, fn_ty }, &[]);
            },
            true,
        ),
        (
            "an unknown intrinsic",
            |f| {
                let void = f.module().void();
                let ty = f.module().fn_ty(void, &[]);
                let decl = f.module().declare("air.not_an_operation", ty);
                void_call(f, Callee::UnknownIntrinsic(decl), &[]);
            },
            true,
        ),
        (
            "a declaration",
            |f| {
                let void = f.module().void();
                let ty = f.module().fn_ty(void, &[]);
                let decl = f.module().declare("elsewhere", ty);
                void_call(f, Callee::Direct(decl), &[]);
            },
            true,
        ),
        (
            "a defined function that encodes",
            |f| {
                let void = f.module().void();
                let dev = f.module().ptr(AddrSpace::Device);
                let ty = f.module().fn_ty(void, &[dev]);
                let mut g = f.module().define("helper", ty);
                draw(&mut g, arg(0));
                g.ret(None);
                let helper = g.finish();
                void_call(f, Callee::Direct(helper), &[arg(0)]);
            },
            true,
        ),
        (
            "a defined function that does not",
            |f| {
                let void = f.module().void();
                let ty = f.module().fn_ty(void, &[]);
                let mut g = f.module().define("helper", ty);
                g.ret(None);
                let helper = g.finish();
                void_call(f, Callee::Direct(helper), &[]);
            },
            false,
        ),
    ];
    for (what, body, may) in bodies {
        let (module, ep) = module(vec![buffer(0, 0), buffer(1, 1)], body);
        assert_eq!(calls_commands(&module, &ep), may, "{what}");
    }
}

/// A call of the indirect-command operation `name`, of `params` (after the command, a device
/// pointer), with `cmd` and `args`.
fn operation(
    f: &mut FunctionBuilder<'_>,
    name: &str,
    params: &[TypeId],
    cmd: Value,
    args: &[Value],
) {
    let void = f.module().void();
    let dev = f.module().ptr(AddrSpace::Device);
    let all: Vec<TypeId> = std::iter::once(dev).chain(params.iter().copied()).collect();
    let ty = f.module().fn_ty(void, &all);
    let decl = f.module().declare(name, ty);
    let op = AirIntrinsic::parse(name).unwrap();
    assert!(matches!(op, AirIntrinsic::IndirectCommand(_)), "{name}");
    let all: Vec<Value> = std::iter::once(cmd).chain(args.iter().copied()).collect();
    void_call(f, Callee::Air { decl, op }, &all);
}

/// `air.draw_primitives_render_command(cmd, 0, primitive_type, 0, 3, 1, 0)`.
fn draw_of(f: &mut FunctionBuilder<'_>, cmd: Value, primitive_type: Value) {
    let i32t = f.module().int(32);
    let [zero, one, three] = [0, 1, 3].map(|v| f.module().const_int(i32t, v));
    let args = [zero, primitive_type, zero, three, one, zero];
    operation(
        f,
        "air.draw_primitives_render_command",
        &[i32t; 6],
        cmd,
        &args,
    );
}

/// An entry that draws a constant primitive type writes that type's bit alone; one whose type
/// is loaded writes every type's; a value past Metal's five (a private type's, or none) writes a
/// point draw, as Metal's encoding does.
#[test]
fn a_draw_writes_its_constant_primitive_type() {
    for (raw, types) in [
        (0u64, 0b1),
        (3, 0b1000),
        (4, 0b1_0000),
        (5, 0b1),
        (9, 0b1),
        (255, 0b1),
    ] {
        let e = specialized(|f| {
            let i32t = f.module().int(32);
            let t = f.module().const_int(i32t, raw);
            draw_of(f, arg(0), t);
        });
        let writes = e.command_writes();
        assert_eq!(writes.operations, CommandWrites::DRAW, "{raw}");
        assert_eq!(writes.primitive_types, types, "{raw}");
        assert_eq!(
            writes.may_draw_primitive_type(3),
            types & 0b1000 != 0,
            "{raw}"
        );
        assert!(!writes.may_draw_primitive_type(5));
    }
    let loaded = specialized(|f| {
        let i32t = f.module().int(32);
        let t = load(f, i32t, arg(1));
        draw_of(f, arg(0), t);
    });
    assert_eq!(
        loaded.command_writes().primitive_types,
        CommandWrites::ALL_PRIMITIVE_TYPES
    );
}

/// Each kind of operation writes its own bit: an indexed draw also its primitive type, a set
/// barrier and a cleared one apart, and the operations of a defined function the entry calls
/// count as the entry's.
#[test]
fn each_operation_writes_its_bit() {
    let e = specialized(|f| {
        let i32t = f.module().int(32);
        let i64t = f.module().int(64);
        let dev = f.module().ptr(AddrSpace::Device);
        let [zero, one, six] = [0, 1, 6].map(|v| f.module().const_int(i32t, v));
        let stride = f.module().const_int(i64t, 16);
        operation(
            f,
            "air.draw_indexed_primitives_render_command.p1i16",
            &[i32t, i32t, i32t, dev, i32t, i32t, i32t],
            arg(0),
            &[zero, zero, six, arg(1), one, zero, zero],
        );
        operation(
            f,
            "air.set_barrier_render_command",
            &[i32t],
            arg(0),
            &[zero],
        );
        operation(
            f,
            "air.set_vertex_buffer_render_command.p1i8",
            &[i32t, dev, i64t, i32t],
            arg(0),
            &[zero, arg(1), stride, one],
        );
        let void = f.module().void();
        let ty = f.module().fn_ty(void, &[dev]);
        let mut g = f.module().define("helper", ty);
        operation(&mut g, "air.reset_render_command", &[i32t], arg(0), &[zero]);
        operation(
            &mut g,
            "air.clear_barrier_render_command",
            &[i32t],
            arg(0),
            &[zero],
        );
        g.ret(None);
        let helper = g.finish();
        void_call(f, Callee::Direct(helper), &[arg(0)]);
    });
    let writes = e.command_writes();
    assert_eq!(
        writes.operations,
        CommandWrites::DRAW_INDEXED
            | CommandWrites::BARRIER
            | CommandWrites::BUFFER
            | CommandWrites::RESET
            | CommandWrites::CLEAR_BARRIER
    );
    assert_eq!(writes.primitive_types, 0b1);
    assert!(!writes.may(CommandWrites::PIPELINE | CommandWrites::DRAW));
    assert_eq!(writes.kinds, CommandWrites::RENDER_COMMANDS);
    assert!(!writes.writes_kinds(CommandWrites::COMPUTE_COMMANDS));
}

/// Each operation writes commands of its intrinsic family's kind: a compute command's operations
/// compute commands alone (a kernel buffer setter among them), a render command's render
/// commands alone (above), and an entry that calls both writes both kinds.
#[test]
fn each_operation_writes_its_family_s_kind() {
    let compute = specialized(|f| {
        let i32t = f.module().int(32);
        let i64t = f.module().int(64);
        let dev = f.module().ptr(AddrSpace::Device);
        let [zero, one] = [0, 1].map(|v| f.module().const_int(i32t, v));
        let stride = f.module().const_int(i64t, 16);
        operation(f, "air.reset_compute_command", &[i32t], arg(0), &[zero]);
        operation(
            f,
            "air.set_barrier_compute_command",
            &[i32t],
            arg(0),
            &[zero],
        );
        operation(
            f,
            "air.set_kernel_buffer_compute_command.p1i8",
            &[i32t, dev, i64t, i32t],
            arg(0),
            &[zero, arg(1), stride, one],
        );
    });
    let writes = compute.command_writes();
    assert_eq!(
        writes.operations,
        CommandWrites::RESET | CommandWrites::BARRIER | CommandWrites::BUFFER
    );
    assert_eq!(writes.kinds, CommandWrites::COMPUTE_COMMANDS);
    assert!(!writes.writes_kinds(CommandWrites::RENDER_COMMANDS));
    let both = specialized(|f| {
        let i32t = f.module().int(32);
        let zero = f.module().const_int(i32t, 0);
        operation(f, "air.reset_compute_command", &[i32t], arg(0), &[zero]);
        operation(f, "air.reset_render_command", &[i32t], arg(0), &[zero]);
    });
    assert_eq!(both.command_writes().kinds, CommandWrites::ALL_KINDS);
    assert_eq!(CommandWrites::ANY.kinds, CommandWrites::ALL_KINDS);
}

/// A command buffer the metadata names and no operation reaches is written nothing; a call the
/// walk cannot see into may write anything.
#[test]
fn only_operations_and_unseen_calls_write() {
    let command_buffer = ParamBinding {
        role: ParamRole::CommandBuffer(ObjectArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(1),
            },
            access: None,
        }),
        ..buffer(0, 0)
    };
    let (named, ep) = module(vec![command_buffer, buffer(1, 1)], |_| {});
    let abi = abi(vec![ep], Vec::new());
    let e = specialize(named, &abi, &request(&[]), &mut meter()).unwrap();
    assert!(e.may_encode_indirect_commands());
    assert!(e.command_writes().is_empty());
    let (unseen, ep) = module(vec![buffer(0, 0), buffer(1, 1)], |f| {
        let void = f.module().void();
        let ty = f.module().fn_ty(void, &[]);
        let decl = f.module().declare("elsewhere", ty);
        void_call(f, Callee::Direct(decl), &[]);
    });
    assert_eq!(
        command_writes::command_writes(&unseen, &ep),
        CommandWrites::ANY
    );
}
