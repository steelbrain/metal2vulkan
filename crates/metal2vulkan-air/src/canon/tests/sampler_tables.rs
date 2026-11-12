//! A pointer loaded from a constant table at an index only the run knows is the `select` chain
//! over the table's elements: a program-scope `constant constexpr array<sampler, N>` read at a
//! buffer's index, as `{ [N x { ptr }] }` of sampler state addresses.

use super::*;

/// The element a table entry holds.
#[derive(Clone, Copy)]
enum Element {
    /// The address of state global `k`.
    State(usize),
    /// An undefined pointer.
    Undef,
}

/// `kernel k(device i32 *out, device i64 *idx)`: `table[idx[0]]` of a constant table of
/// `elements` (`{ [N x { ptr }] }`, each `{ ptr }` the address of one of three sampler state
/// globals), compared with state 1's address, stored as an `i32`.
fn table_kernel(elements: &[Element]) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let dev = m.ptr(AddrSpace::Device);
    let cptr = m.ptr(AddrSpace::Constant);
    let words = m.array(i64t, 2);
    let mut states = Vec::new();
    for k in 0..3u64 {
        let w = m.const_int(i64t, 0x100 + k);
        let Value::Const(w) = w else { unreachable!() };
        let init = m.constant(words, Const::Aggregate(Box::new([w, w])));
        let name = m.sym(&format!("state{k}"));
        states.push(m.global(Global {
            name,
            value_ty: words,
            space: AddrSpace::Constant,
            constant: true,
            init: Some(init),
            externally_initialized: false,
            visibility: Default::default(),
            unnamed_addr: Default::default(),
            linkage: Linkage::Internal,
            align: None,
            section: None,
        }));
    }
    let sampler = m.struct_ty(&[cptr]);
    let arr = m.array(sampler, elements.len() as u64);
    let wrapper = m.struct_ty(&[arr]);
    let entries: Vec<_> = elements
        .iter()
        .map(|e| {
            let p = match *e {
                Element::State(k) => match m.global_addr(states[k]) {
                    Value::Const(c) => c,
                    _ => unreachable!(),
                },
                Element::Undef => m.constant(cptr, Const::Undef),
            };
            m.constant(sampler, Const::Aggregate(Box::new([p])))
        })
        .collect();
    let arr_c = m.constant(arr, Const::Aggregate(entries.into()));
    let init = m.constant(wrapper, Const::Aggregate(Box::new([arr_c])));
    let name = m.sym("table");
    let table = m.global(Global {
        name,
        value_ty: wrapper,
        space: AddrSpace::Constant,
        constant: true,
        init: Some(init),
        externally_initialized: false,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: None,
        section: None,
    });
    let table_addr = m.global_addr(table);
    let state1 = m.global_addr(states[1]);
    let (z64, z32) = (m.const_int(i64t, 0), m.const_int(i32t, 0));
    let fty = m.fn_ty(void, &[dev, dev]);
    let mut f = m.define("k", fty);
    let i = load(&mut f, i64t, arg(1));
    let at = f.push(
        cptr,
        InstKind::Gep {
            src_ty: wrapper,
            base: table_addr,
            indices: Box::new([z64, z32, i, z32]),
            inbounds: true,
        },
    );
    let s = load(&mut f, cptr, at);
    let is1 = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: s,
            b: state1,
        },
    );
    let w = f.push(
        i32t,
        InstKind::Cast {
            op: CastOp::ZExt,
            v: is1,
        },
    );
    store(&mut f, w, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0), buffer(1, 1)])], vec![]);
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// The count of lines of `text` holding `what`.
fn count(text: &str, what: &str) -> usize {
    text.lines().filter(|l| l.contains(what)).count()
}

/// Three addresses at an index the run reads are two compares and two `select`s, the first
/// choosing element 0 at index 0, the last element every other index's; the table's
/// `getelementptr` and load are gone.
#[test]
fn a_table_load_at_a_dynamic_index_is_a_select_chain() {
    use Element::State;
    let e = table_kernel(&[State(0), State(1), State(2)]).unwrap();
    let text = body_text(&e);
    assert_eq!(count(&text, "getelementptr"), 0, "{text}");
    assert_eq!(count(&text, " = select "), 2, "{text}");
    // The chain's two compares, and the kernel's own.
    assert_eq!(count(&text, " = icmp eq "), 3, "{text}");
    assert_eq!(
        count(&text, "\"table\""),
        0,
        "the table is no longer read: {text}"
    );
    // The chain's first `select` falls back to the last element, state 2.
    let state2 = text
        .lines()
        .find(|l| l.contains("\"state2\""))
        .and_then(|l| l.split_whitespace().next())
        .unwrap_or_else(|| panic!("state 2 is read: {text}"));
    let first = text.lines().find(|l| l.contains(" = select ")).unwrap();
    assert!(
        first.ends_with(state2) || first.contains(&format!("{state2},")),
        "{first} falls back to {state2}"
    );
    // Only the index's load is left.
    assert_eq!(count(&text, " = load "), 1, "{text}");
}

/// A table of one element is that element; a table holding an element that is no constant
/// address, and one past the pass's length, keep their load.
#[test]
fn a_table_load_is_read_only_where_every_element_is_known() {
    use Element::{State, Undef};
    let one = body_text(&table_kernel(&[State(1)]).unwrap());
    assert_eq!(count(&one, " = select "), 0, "{one}");
    assert_eq!(count(&one, "getelementptr"), 0, "{one}");
    assert_eq!(count(&one, " = load "), 1, "{one}");
    let undefined = body_text(&table_kernel(&[State(0), Undef, State(2)]).unwrap());
    assert_eq!(count(&undefined, "getelementptr"), 1, "{undefined}");
    assert_eq!(count(&undefined, " = select "), 0, "{undefined}");
    let long = vec![State(0); 1 + crate::canon::const_tables::ELEMENTS as usize];
    let long = body_text(&table_kernel(&long).unwrap());
    assert_eq!(count(&long, "getelementptr"), 1, "{long}");
    assert_eq!(count(&long, " = select "), 0, "{long}");
}
