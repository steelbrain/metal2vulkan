//! L0 tests: the read-only test ([`MemoryPlan::read_only`]) and which read-only slots the function
//! reads in a loop at an offset every thread executing the read holds alike
//! ([`SlotPlan::uniform_loop_reads`]), which only a kernel on a profile that takes
//! `DriverWorkaround::UniformAddressStorageReadsSlow` asks: the Iris Xe RPL-P's, against which
//! these kernels are planned, while `moltenvk-m3`'s plan marks none.

use super::*;
use crate::air::abi::Builtin;
use crate::air::ir::BinOp;

/// `[[buffer(index)]]` in device space, declared `air.read` ([`buffer`] declares `air.read_write`).
fn read_buffer(index: u32) -> ParamRole {
    ParamRole::Buffer(BufferArg {
        location: slot(index),
        access: Access::Read,
        space: AddrSpace::Device,
        buffer_size: None,
        layout: None,
        raster_order_group: None,
    })
}

/// A `uint` `[[thread_position_in_grid]]` parameter.
fn position(k: &mut Kernel) {
    let i32t = k.mb.int(32);
    k.params.push(i32t);
    k.roles
        .push(ParamRole::Builtin(Builtin::ThreadPositionInGrid));
}

/// The plan of `e` on the committed profile that takes the workaround.
fn on_iris(e: &Entry) -> MemoryPlan {
    let iris = profile("intel-irisxe-rplp-anv", DevicePolicy::default()).expect("iris");
    plan(e, &iris).expect("analyze")
}

fn uniform_loop_reads(p: &MemoryPlan, param: u32) -> bool {
    p.slots
        .iter()
        .find(|s| s.param == param)
        .expect("slot")
        .uniform_loop_reads
}

fn read_only(p: &MemoryPlan, param: u32) -> bool {
    p.read_only(p.slots.iter().find(|s| s.param == param).expect("slot"))
}

/// The instruction pushed `n` after instruction `v`: a forward reference (the builder numbers
/// instructions in push order, void ones included).
fn after(v: Value, n: u32) -> Value {
    match v {
        Value::Inst(i) => Value::Inst(InstId::new(i.raw() + n)),
        other => panic!("not an instruction: {other:?}"),
    }
}

fn icmp(fb: &mut FunctionBuilder<'_>, pred: IntPredicate, a: Value, b: Value) -> Value {
    let i1 = fb.module().int(1);
    fb.push(i1, InstKind::ICmp { pred, a, b })
}

fn phi(fb: &mut FunctionBuilder<'_>, ty: TypeId, incoming: &[(Value, BlockId)]) -> Value {
    fb.push(
        ty,
        InstKind::Phi {
            incoming: incoming.into(),
            fmf: FastMath::NONE,
        },
    )
}

/// A convolution's tap loop: each thread sums `src[x]` times `weights[i]` while its own
/// `x = gid + i` stays below its own bound, so threads leave the loop after different
/// iterations. The counter `i` starts at 0 and steps by 1, so the threads in one iteration read
/// one weight: the weights' slot reads are uniform in a loop. The source's are at a per-thread
/// offset, and the output is written.
#[test]
fn a_tap_loop_reads_its_weights_uniformly_and_its_source_per_thread() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, read_buffer(0));
    k.param(AddrSpace::Device, read_buffer(1));
    k.param(AddrSpace::Device, buffer(2, AddrSpace::Device, None));
    position(&mut k);
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let f32t = fb.module().float();
        let zero = fb.module().const_int(i32t, 0);
        let one = fb.module().const_int(i32t, 1);
        let eight = fb.module().const_int(i32t, 8);
        let entry = BlockId::new(0);
        let (head, exit) = (fb.block(), fb.block());
        let gid = fb.arg(3);
        let end = fb.bin(BinOp::Add, gid, eight);
        fb.br(head);
        fb.switch_to(head);
        // `end`, the `br`, the phi, then its back-edge value.
        let i = phi(fb, i32t, &[(zero, entry), (after(end, 3), head)]);
        let inc = fb.bin(BinOp::Add, i, one);
        assert_eq!(inc, after(end, 3));
        let x = fb.bin(BinOp::Add, gid, i);
        let at_weight = gep(fb, f32t, fb.arg(0), &[i]);
        let w = load(fb, f32t, at_weight);
        let at_source = gep(fb, f32t, fb.arg(1), &[x]);
        let s = load(fb, f32t, at_source);
        let c = icmp(fb, IntPredicate::Slt, x, end);
        fb.cond_br(c, head, exit);
        fb.switch_to(exit);
        let at_out = gep(fb, f32t, fb.arg(2), &[gid]);
        store(fb, s, at_out);
        store(fb, w, at_out);
        fb.ret(None);
    });
    let plan = on_iris(&e);
    assert!(read_only(&plan, 0) && read_only(&plan, 1));
    assert!(!read_only(&plan, 2));
    assert!(uniform_loop_reads(&plan, 0), "the weights at the counter");
    assert!(
        !uniform_loop_reads(&ok(&e), 0),
        "a profile without the workaround asks nothing"
    );
    assert!(!uniform_loop_reads(&plan, 1), "the source at gid + i");
    assert!(!uniform_loop_reads(&plan, 2), "the written output");
}

/// An index merged at the join of a branch on the thread's position is per thread, though both
/// its incoming values are constants; the same kernel whose branch every thread takes alike (a
/// constant condition) reads uniformly. The loop's own exit is uniform in both.
#[test]
fn an_index_merged_at_a_divergent_join_is_not_uniform() {
    for (divergent, want) in [(true, false), (false, true)] {
        let mut k = Kernel::new();
        k.param(AddrSpace::Device, read_buffer(0));
        k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
        position(&mut k);
        let e = k.build(|fb| {
            let i32t = fb.module().int(32);
            let f32t = fb.module().float();
            let zero = fb.module().const_int(i32t, 0);
            let one = fb.module().const_int(i32t, 1);
            let two = fb.module().const_int(i32t, 2);
            let four = fb.module().const_int(i32t, 4);
            let eight = fb.module().const_int(i32t, 8);
            let (a, b, merge, head, exit) =
                (fb.block(), fb.block(), fb.block(), fb.block(), fb.block());
            let tested = match divergent {
                true => fb.arg(2),
                false => zero,
            };
            let c0 = icmp(fb, IntPredicate::Ult, tested, four);
            fb.cond_br(c0, a, b);
            fb.switch_to(a);
            fb.br(merge);
            fb.switch_to(b);
            fb.br(merge);
            fb.switch_to(merge);
            let k = phi(fb, i32t, &[(one, a), (two, b)]);
            fb.br(head);
            fb.switch_to(head);
            // `k`, the `br`, the phi, then its back-edge value.
            let i = phi(fb, i32t, &[(zero, merge), (after(k, 3), head)]);
            let inc = fb.bin(BinOp::Add, i, one);
            assert_eq!(inc, after(k, 3));
            let j = fb.bin(BinOp::Add, i, k);
            let at = gep(fb, f32t, fb.arg(0), &[j]);
            let w = load(fb, f32t, at);
            let c = icmp(fb, IntPredicate::Ult, inc, eight);
            fb.cond_br(c, head, exit);
            fb.switch_to(exit);
            store(fb, w, fb.arg(1));
            fb.ret(None);
        });
        let plan = on_iris(&e);
        assert_eq!(
            uniform_loop_reads(&plan, 0),
            want,
            "divergent branch: {divergent}"
        );
    }
}

/// A read at a uniform offset outside any loop is not a loop read, and a slot read at one
/// constant offset in a loop is not either.
#[test]
fn only_dynamic_reads_in_loops_count() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, read_buffer(0));
    k.param(AddrSpace::Device, read_buffer(1));
    k.param(AddrSpace::Device, buffer(2, AddrSpace::Device, None));
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let f32t = fb.module().float();
        let zero = fb.module().const_int(i32t, 0);
        let one = fb.module().const_int(i32t, 1);
        let three = fb.module().const_int(i32t, 3);
        let eight = fb.module().const_int(i32t, 8);
        let entry = BlockId::new(0);
        let (head, exit) = (fb.block(), fb.block());
        // Slot 0 outside the loop at a uniform but not constant offset.
        let n = load(fb, i32t, fb.arg(1));
        let n = fb.bin(BinOp::And, n, zero);
        let at_first = gep(fb, f32t, fb.arg(0), &[n]);
        let first = load(fb, f32t, at_first);
        fb.br(head);
        fb.switch_to(head);
        // `first`, the `br`, the phi, then its back-edge value.
        let i = phi(fb, i32t, &[(zero, entry), (after(first, 3), head)]);
        let inc = fb.bin(BinOp::Add, i, one);
        assert_eq!(inc, after(first, 3));
        // Slot 1 in the loop at one constant offset.
        let at_fixed = gep(fb, f32t, fb.arg(1), &[three]);
        let fixed = load(fb, f32t, at_fixed);
        let c = icmp(fb, IntPredicate::Ult, inc, eight);
        fb.cond_br(c, head, exit);
        fb.switch_to(exit);
        store(fb, first, fb.arg(2));
        store(fb, fixed, fb.arg(2));
        fb.ret(None);
    });
    let plan = on_iris(&e);
    assert!(read_only(&plan, 0) && read_only(&plan, 1));
    assert!(!uniform_loop_reads(&plan, 0), "outside the loop");
    assert!(!uniform_loop_reads(&plan, 1), "one constant offset");
}

/// The read-only test: a slot declared `air.read_write` that is only read is not, nor one
/// declared `air.read` that is written, nor any slot of a function where a buffer pointer has a
/// dynamic root (a pointer loaded from memory may reach the slot's memory).
#[test]
fn read_only_needs_the_declaration_the_accesses_and_no_dynamic_root() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, read_buffer(0));
    k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
    k.param(AddrSpace::Device, read_buffer(2));
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let v = load(fb, i32t, fb.arg(0));
        let u = load(fb, i32t, fb.arg(1));
        let sum = fb.bin(BinOp::Add, v, u);
        store(fb, sum, fb.arg(2));
        fb.ret(None);
    });
    let plan = ok(&e);
    assert!(!plan.needs.dynamic_roots);
    assert!(read_only(&plan, 0), "declared and accessed read only");
    assert!(!read_only(&plan, 1), "declared read_write");
    assert!(!read_only(&plan, 2), "declared read, written");

    let mut k = Kernel::new();
    k.param(AddrSpace::Device, read_buffer(0));
    k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let q = load(fb, ptr, fb.arg(0));
        let v = load(fb, i32t, q);
        store(fb, v, fb.arg(1));
        fb.ret(None);
    });
    let plan = ok(&e);
    assert!(plan.needs.dynamic_roots);
    assert!(!read_only(&plan, 0), "a dynamic root may reach it");
}
