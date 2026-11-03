//! The static initializers' integer conversions: `air.convert` of an integer or `bool`, a scalar or
//! each lane of a vector, to an integer; and of a scalar to a float. A vector converted to floats
//! is the device's to compute, and its initializer is refused as before.

use super::*;

/// The type of `(element bits, lanes)`: an integer of those bits (a `float` for 0 bits), or a
/// vector of `lanes` of them.
fn shape(m: &mut ModuleBuilder, (bits, lanes): (u8, u32)) -> TypeId {
    let elem = if bits == 0 { m.float() } else { m.int(bits) };
    if lanes == 1 {
        elem
    } else {
        m.vector(elem, lanes)
    }
}

/// A static initializer stores `<name>` of function constant 0 (of shape `src`, given `bytes`
/// as `data`) into a constant global of shape `dst`; the kernel stores the global. The
/// specialization.
fn specialized(
    name: &str,
    src: (u8, u32),
    dst: (u8, u32),
    data: DataType,
    bytes: &[u8],
) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let dev = m.ptr(AddrSpace::Device);
    let (src, dst) = (shape(&mut m, src), shape(&mut m, dst));
    let fc = fc_global(&mut m, "x", src);
    let undef = m.constant(dst, Const::Undef);
    let g = m.simple_global("g", dst, AddrSpace::Constant, Some(undef));
    let (fc_addr, g_addr) = (m.global_addr(fc), m.global_addr(g));
    let op = AirIntrinsic::parse(name).unwrap_or_else(|| panic!("{name}"));
    let decl_ty = m.fn_ty(dst, &[src]);
    let decl = m.declare(name, decl_ty);
    static_init(&mut m, "init", |f| {
        let v = load(f, src, fc_addr);
        let w = call(f, dst, Callee::Air { decl, op }, &[v]);
        store(f, w, g_addr);
    });
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = load(&mut f, dst, g_addr);
    store(&mut f, v, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0)])],
        vec![fc_decl(0, fc, src)],
    );
    let value = ConstValue {
        ty: data,
        bytes: bytes.into(),
    };
    specialize(module, &abi, &request(&[(0, value)]), &mut meter())
}

/// The lanes [`specialized`] stores.
fn converted(name: &str, src: (u8, u32), dst: (u8, u32), data: DataType, bytes: &[u8]) -> Vec<u64> {
    let e = specialized(name, src, dst, data, bytes).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    stored_lanes(&e)
}

#[test]
fn a_static_initializer_converts_integers_lane_by_lane() {
    let float = |x: f32| u64::from(x.to_bits());
    let cases: [(&str, (u8, u32), (u8, u32), DataType, &[u8], Vec<u64>); 6] = [
        // Widening, by the source's signedness.
        (
            "air.convert.u.v4i32.u.v4i8",
            (8, 4),
            (32, 4),
            DataType::UChar4,
            &[1, 200, 3, 255],
            vec![1, 200, 3, 255],
        ),
        (
            "air.convert.s.v4i32.s.v4i8",
            (8, 4),
            (32, 4),
            DataType::UChar4,
            &[1, 200, 3, 255],
            vec![1, 0xffff_ffc8, 3, 0xffff_ffff],
        ),
        // Narrowing truncates.
        (
            "air.convert.u.v2i8.u.v2i32",
            (32, 2),
            (8, 2),
            DataType::UInt2,
            &[0xff, 0x01, 0, 0, 0x07, 0, 1, 0],
            vec![0xff, 0x07],
        ),
        // To `bool`: `x != 0`.
        (
            "air.convert.u.v4i1.u.v4i8",
            (8, 4),
            (1, 4),
            DataType::UChar4,
            &[0, 2, 0, 128],
            vec![0, 1, 0, 1],
        ),
        // A scalar, sign-extended.
        (
            "air.convert.u.i32.s.i8",
            (8, 1),
            (32, 1),
            DataType::Char,
            &[0xf0],
            vec![0xffff_fff0],
        ),
        // A scalar to a float, rounded to nearest even.
        (
            "air.convert.f.f32.s.i8",
            (8, 1),
            (0, 1),
            DataType::Char,
            &[0xc8],
            vec![float(-56.0)],
        ),
    ];
    for (name, src, dst, data, bytes, want) in cases {
        assert_eq!(converted(name, src, dst, data, bytes), want, "{name}");
    }
}

#[test]
fn a_vector_converted_to_floats_is_not_evaluated() {
    let e = specialized(
        "air.convert.f.v2f32.s.v2i8",
        (8, 2),
        (0, 2),
        DataType::Char2,
        &[0xc8, 3],
    );
    assert!(
        matches!(&e, Err(Error::Unsupported(u)) if u.feature == Feature::StaticInitializer),
        "{e:?}"
    );
}
