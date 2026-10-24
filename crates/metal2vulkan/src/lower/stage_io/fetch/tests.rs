//! The pulled fetch on the CPU interpreter: the element offset of every step, the decode of every
//! Metal vertex format against [`reference`], unaligned offsets, and containment, through a storage
//! view and through a device address.
//!
//! A compute harness stands in for the vertex stage (the interpreter runs compute entries): it
//! reads the element inputs (index, base instance, stride, bind residual) from a parameter buffer
//! where the vertex stage reads builtins and push fields, then runs the same `divided`,
//! `displacement`, `access::pulled::{view_read, psb_read}`, and `fetch_from` the stage emits.

use super::*;
use crate::lower::arith::legal::{f16_to_f64, f64_to_f16};
use crate::memory::views::{self, ViewKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{ExecMode, ExecModel, Type, TypedModule};

/// How the harness computes the element index from its parameters.
#[derive(Clone, Copy, Debug)]
enum Elem {
    /// The index itself (per vertex, or per instance at rate 1).
    Direct,
    /// `base + (index - base) / rate`.
    Divided(u32),
    /// No element (constant step).
    Constant,
}

/// Where the harness reads the data: the storage view `kslot(0)` at the residual parameter plus
/// the displacement, or `[[buffer(PSB_BUFFER)]]` by device address through the root table at
/// [`TABLE`] (`access::pulled::psb_read`), contained or not.
#[derive(Clone, Copy, Debug)]
enum Source {
    View(Check),
    Psb { contained: bool },
}

/// The device addresses of the root table, the data, and the sink.
const TABLE: u64 = 0x1000_0000;
const DATA: u64 = 0x2000_0000;
const SINK: u64 = 0x3000_0000;
/// The Metal buffer index the device-address harness reads (root `2 + 5`).
const PSB_BUFFER: u32 = 5;

fn kslot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// A compute module fetching `encoding` into `form` from buffer 0 with the parameters of
/// buffer 1 (`index`, `base`, `stride`, `residual`), writing each lane's bits to buffer 2.
fn module(encoding: Encoding, form: Form, elem: Elem, offset: u32, check: Check) -> TypedModule {
    module_from(Source::View(check), encoding, form, elem, offset)
}

/// [`module`] reading from `source`.
fn module_from(
    source: Source,
    encoding: Encoding,
    form: Form,
    elem: Elem,
    offset: u32,
) -> TypedModule {
    let mut mb = ModuleBuilder::new();
    let data = views::declare_at(&mut mb, kslot(0), ViewKind::Storage).expect("data");
    let params = views::declare_at(&mut mb, kslot(1), ViewKind::Storage).expect("params");
    let out = views::declare_at(&mut mb, kslot(2), ViewKind::Storage).expect("out");
    let void = mb.ty(Type::Void).expect("void");
    let mut f = mb.function(void).expect("function");
    let fb = &mut f;
    let p = |fb: &mut FunctionBuilder<'_>, i: u32| -> SValue {
        let w = k32(fb, i).expect("k");
        views::read_word(fb, params, w).expect("param")
    };
    let (index, base, stride, residual) = (p(fb, 0), p(fb, 1), p(fb, 2), p(fb, 3));
    let element = match elem {
        Elem::Direct => Some(index),
        Elem::Divided(rate) => Some(divided(fb, index, base, rate).expect("divided")),
        Elem::Constant => None,
    };
    let m = displacement(fb, element.map(|e| (e, stride)), offset).expect("displacement");
    let v = match source {
        Source::View(check) => {
            let read = pulled::view_read(fb, data, check, residual, m, encoding.bytes())
                .expect("view read");
            fetch_from(fb, read, encoding, form).expect("fetch")
        }
        Source::Psb { contained } => {
            let u64_ = fb
                .module()
                .ty(Type::Int {
                    width: 64,
                    signed: false,
                })
                .expect("u64");
            let table = fb
                .module()
                .constant(crate::spirv::ir::Constant::Scalar {
                    ty: u64_,
                    bits: crate::spirv::ir::ConstBits(TABLE),
                })
                .expect("table");
            let table = fb.module().const_value(table).expect("table");
            let read = pulled::psb_read(fb, table, PSB_BUFFER, m, 0, encoding.bytes(), contained)
                .expect("psb read");
            fetch_from(fb, read, encoding, form).expect("fetch")
        }
    };
    let u32_ = ops::ty(fb, u32_scalar()).expect("u32");
    for i in 0..u32::from(form.lanes) {
        let x = if form.lanes == 1 {
            v
        } else {
            fb.composite_extract(v, &[i]).expect("lane")
        };
        let x = if form.is_float() {
            fb.convert(ConvOp::Bitcast, x, u32_).expect("bits")
        } else {
            x
        };
        let w = k32(fb, i).expect("k");
        let ptr = views::word_ptr(fb, out, w).expect("ptr").expect("storage");
        fb.store(ptr, x, MemAccess::default()).expect("store");
    }
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    mb.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    mb.finalize().expect("finalize")
}

/// Run the module over `data` with parameters `params`; the output lanes' bits.
fn run(m: &TypedModule, data: &[u8], params: [u32; 4], lanes: u8) -> Vec<u32> {
    run_with(m, data, params, lanes, Vec::new())
}

/// The root table of the device-address harness: the sink at root 1 and the data at
/// `2 + PSB_BUFFER`, bound at `bind_offset` with its allocation size.
fn root_table(data_len: usize, bind_offset: u64) -> Vec<u8> {
    let mut t = vec![0u8; 32 * (PSB_BUFFER as usize + 3)];
    let mut entry = |root: usize, fields: [u64; 4]| {
        for (k, f) in fields.iter().enumerate() {
            let at = 32 * root + 8 * k;
            t[at..at + 8].copy_from_slice(&f.to_le_bytes());
        }
    };
    entry(1, [SINK, 0, 64, 0]);
    entry(
        PSB_BUFFER as usize + 2,
        [DATA, 0, data_len as u64, bind_offset],
    );
    t
}

/// Run a device-address module over `data` bound at `bind_offset`.
fn run_psb(
    m: &TypedModule,
    data: &[u8],
    bind_offset: u64,
    params: [u32; 4],
    lanes: u8,
) -> Vec<u32> {
    let extra = vec![
        Buffer {
            slot: None,
            address: Some(TABLE),
            bytes: root_table(data.len(), bind_offset),
        },
        Buffer {
            slot: None,
            address: Some(DATA),
            bytes: data.to_vec(),
        },
        Buffer {
            slot: None,
            address: Some(SINK),
            bytes: vec![0xA5; 64],
        },
    ];
    run_with(m, &[0; 16], params, lanes, extra)
}

fn run_with(
    m: &TypedModule,
    data: &[u8],
    params: [u32; 4],
    lanes: u8,
    extra: Vec<Buffer>,
) -> Vec<u32> {
    let mut buffers = vec![
        Buffer {
            slot: Some(kslot(0)),
            address: None,
            bytes: data.to_vec(),
        },
        Buffer {
            slot: Some(kslot(1)),
            address: None,
            bytes: params.iter().flat_map(|w| w.to_le_bytes()).collect(),
        },
        Buffer {
            slot: Some(kslot(2)),
            address: None,
            bytes: vec![0; 16],
        },
    ];
    buffers.extend(extra);
    let res = Resources {
        buffers,
        push: Vec::new(),
        spec: Default::default(),
    };
    let o = interp::run(m, &Config::default(), res).expect("run");
    o.buffers[2]
        .as_chunks::<4>()
        .0
        .iter()
        .take(usize::from(lanes))
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn float(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes,
    }
}

fn half_form(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::Half),
        lanes,
    }
}

fn int(bits: u8, lanes: u8) -> Form {
    Form {
        held: Held::Int {
            bits,
            container: 32,
        },
        lanes,
    }
}

/// The bits a lane of `form` holds for reference component `x`: an `f32` (a `half` rounded as
/// `OpQuantizeToF16` does on the interpreter), or an integer masked to its width.
fn expected_bits(form: Form, x: f64) -> u32 {
    match form.held {
        Held::Float(FloatFmt::Half) => {
            let h = f64_to_f16(x);
            let h = if h & 0x7C00 == 0 { h & 0x8000 } else { h };
            (f16_to_f64(h) as f32).to_bits()
        }
        Held::Float(_) => (x as f32).to_bits(),
        _ => (x as u32) & (ops::ones(form.bits()) as u32),
    }
}

fn check_lanes(what: &str, form: Form, got: &[u32], want: [f64; 4]) {
    for (i, g) in got.iter().enumerate() {
        let w = want[i];
        if form.is_float() && w.is_nan() {
            assert!(
                f32::from_bits(*g).is_nan(),
                "{what} lane {i}: {g:#x} is not NaN"
            );
        } else {
            assert_eq!(*g, expected_bits(form, w), "{what} lane {i} (want {w})");
        }
    }
}

/// A deterministic byte pattern.
fn pattern(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 7) as u8
        })
        .collect()
}

/// Every Metal vertex format decodes exactly as the reference, at an unaligned byte offset
/// (index 3, stride 18, residual 5, attribute offset 2: byte 61), into four lanes of its class
/// (the missing components are `(0, 0, 0, 1)`) and into fewer, narrower lanes.
#[test]
fn every_format_decodes_exactly_at_an_unaligned_offset() {
    let at = 5 + 3 * 18 + 2;
    for (n, &format) in VertexFormat::ALL.iter().enumerate() {
        if format == VertexFormat::Invalid {
            assert!(encoding(format).is_err());
            continue;
        }
        let e = encoding(format).expect("encoding");
        let data = pattern(96, n as u32 + 1);
        let want = reference(e, &data[at..]);
        let forms = if e.is_float() {
            [float(4), float(2)]
        } else {
            [int(32, 4), int(8, 3)]
        };
        for form in forms {
            for check in [Check::None, Check::InShader] {
                let m = module(e, form, Elem::Direct, 2, check);
                let got = run(&m, &data, [3, 0, 18, 5], form.lanes);
                check_lanes(&format!("{format:?} {form:?} {check:?}"), form, &got, want);
            }
        }
    }
}

/// A float format into a `half` parameter rounds (except a `half` format, which is exact).
#[test]
fn a_half_parameter_rounds_float_formats() {
    let mut data = vec![0u8; 32];
    // float2 (1/3, 65504.5), half2 (a denormal, -2), uchar4n (1, 128, 255, 7).
    data[0..4].copy_from_slice(&(1.0f32 / 3.0).to_bits().to_le_bytes());
    data[4..8].copy_from_slice(&65504.5f32.to_bits().to_le_bytes());
    data[8..10].copy_from_slice(&0x0001u16.to_le_bytes());
    data[10..12].copy_from_slice(&0xC000u16.to_le_bytes());
    data[12..16].copy_from_slice(&[1, 128, 255, 7]);
    for (format, at) in [
        (VertexFormat::Float2, 0usize),
        (VertexFormat::Half2, 8),
        (VertexFormat::UChar4Normalized, 12),
    ] {
        let e = encoding(format).expect("encoding");
        let form = half_form(2);
        let m = module(e, form, Elem::Constant, at as u32, Check::InShader);
        let got = run(&m, &data, [0, 0, 0, 0], 2);
        let want = reference(e, &data[at..]);
        if format == VertexFormat::Half2 {
            // Exact, the denormal included.
            assert_eq!(got[0], (f16_to_f64(1) as f32).to_bits());
            assert_eq!(got[1], (-2.0f32).to_bits());
        } else {
            check_lanes(&format!("{format:?}"), form, &got, want);
        }
    }
}

/// An unnormalized 8- or 16-bit integer format read into a float parameter reads each component's
/// integer value, zero- or sign-extended, as a float, and the missing ones as `(0, 0, 0, 1)`; a
/// `half` parameter rounds it to nearest even, 65520 and above to infinity. The rows are Metal's
/// measured ones on an M4 (row `r`'s component `c` the width's value `(r + 3 c) %
/// 8`; 65519 read as 65504, 65520 and 65535 as infinity, 2051 as 2052), through a view, contained
/// or not. A 32-bit integer format into a float parameter is an invalid descriptor, as Metal
/// refuses its pipeline.
#[test]
fn integer_formats_read_into_float_parameters_take_their_values() {
    let u8s = [0i64, 1, 2, 127, 128, 200, 254, 255];
    let i8s = [0i64, 1, -1, 127, -128, -2, 100, -100];
    let u16s = [0i64, 1, 2049, 2051, 65504, 65519, 65520, 65535];
    let i16s = [0i64, -1, 32767, -32768, 2049, -2051, 1000, -1000];
    let mut converted = 0;
    for &format in VertexFormat::ALL {
        let Ok(
            e @ Encoding::Int {
                bits,
                count,
                signed,
                normalized: false,
            },
        ) = encoding(format)
        else {
            continue;
        };
        assert_eq!(class_match(e, false, 3), Ok(ClassMatch::Same), "{format:?}");
        if bits == 32 {
            assert_eq!(
                class_match(e, true, 3),
                Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                    attribute: 3
                })),
                "{format:?}"
            );
            continue;
        }
        assert_eq!(
            class_match(e, true, 3),
            Ok(ClassMatch::IntToFloat),
            "{format:?}"
        );
        converted += 1;
        let values = match (bits, signed) {
            (8, false) => u8s,
            (8, true) => i8s,
            (_, false) => u16s,
            (_, true) => i16s,
        };
        let width = usize::from(bits / 8);
        let stride = width * usize::from(count);
        let mut data = Vec::new();
        let mut rows = Vec::new();
        for r in 0..8 {
            let mut want = [0.0, 0.0, 0.0, 1.0];
            for (c, w) in want.iter_mut().enumerate().take(usize::from(count)) {
                let v = values[(r + 3 * c) % 8];
                data.extend_from_slice(&v.to_le_bytes()[..width]);
                *w = v as f64;
            }
            rows.push(want);
        }
        data.resize(data.len() + 16, 0xEE);
        for form in [float(4), half_form(4), float(2)] {
            for check in [Check::None, Check::InShader] {
                let m = module(e, form, Elem::Direct, 0, check);
                for (r, want) in rows.iter().enumerate() {
                    let got = run(&m, &data, [r as u32, 0, stride as u32, 0], form.lanes);
                    check_lanes(
                        &format!("{format:?} {form:?} {check:?} row {r}"),
                        form,
                        &got,
                        *want,
                    );
                }
            }
        }
    }
    assert_eq!(converted, 16);
    // Metal's half answers, exactly.
    let e = encoding(VertexFormat::UShort4).expect("encoding");
    let data: Vec<u8> = [65519u16, 65520, 2051, 2049]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let m = module(e, half_form(4), Elem::Constant, 0, Check::InShader);
    let got = run(&m, &data, [0, 0, 0, 0], 4);
    assert_eq!(
        got,
        [65504.0f32, f32::INFINITY, 2052.0, 2048.0].map(f32::to_bits)
    );
}

/// The element of each step: per vertex (and per instance at rate 1) the index itself; per
/// instance at rate `n` with a nonzero base, `base + (instance - base) / n`; constant, the
/// attribute offset alone whatever the index.
#[test]
fn each_step_fetches_its_element() {
    let data: Vec<u8> = (0..64u32).flat_map(|w| (1000 + w).to_le_bytes()).collect();
    let e = encoding(VertexFormat::UInt).expect("encoding");
    let form = int(32, 1);
    let word = |byte: u32| 1000 + byte / 4;
    // [index, base, stride, residual], attribute offset, expected byte.
    let direct = module(e, form, Elem::Direct, 4, Check::InShader);
    assert_eq!(
        run(&direct, &data, [5, 0, 12, 8], 1),
        [word(8 + 5 * 12 + 4)]
    );
    assert_eq!(run(&direct, &data, [0, 7, 12, 0], 1), [word(4)]);
    for (rate, instance, base, element) in [
        (3u32, 9u32, 2u32, 4u32),
        (3, 2, 2, 2),
        (3, 4, 2, 2),
        (3, 5, 2, 3),
        (2, 7, 0, 3),
        (4, 13, 1, 4),
    ] {
        let m = module(e, form, Elem::Divided(rate), 0, Check::InShader);
        assert_eq!(
            run(&m, &data, [instance, base, 8, 0], 1),
            [word(element * 8)],
            "rate {rate} instance {instance} base {base}"
        );
    }
    let constant = module(e, form, Elem::Constant, 20, Check::InShader);
    for index in [0, 3, 1000] {
        assert_eq!(run(&constant, &data, [index, 5, 8, 4], 1), [word(24)]);
    }
}

/// An element past the end of the bound range is read at offset 0 and yields zero bytes, so the
/// decoded value is `(0, 0, 0, 1)`-filled zero, never a trap or other memory; offsets that wrap
/// `u32` are contained too.
#[test]
fn an_out_of_bounds_element_reads_zero() {
    let data = pattern(32, 9);
    let e = encoding(VertexFormat::Float3).expect("encoding");
    let form = float(4);
    let m = module(e, form, Elem::Direct, 0, Check::InShader);
    let zero = reference(e, &[0; 12]);
    for (index, stride, residual) in [
        (2u32, 12u32, 0u32),
        (1, 21, 0),
        (0, 0, 21),
        (0x1555_5556, 12, 0),
        (1, 0xFFFF_FFF8, 0),
    ] {
        let got = run(&m, &data, [index, 0, stride, residual], 4);
        let at = residual.wrapping_add(index.wrapping_mul(stride)) as usize;
        let want = if at <= 32 - 12 {
            reference(e, &data[at..])
        } else {
            zero
        };
        check_lanes(&format!("{index} {stride} {residual}"), form, &got, want);
    }
    // The last in-bounds element.
    let got = run(&m, &data, [1, 0, 20, 0], 4);
    check_lanes("last", form, &got, reference(e, &data[20..]));
}

/// Through a device address (a source buffer in a `BufferRangeLimit` variant, the descriptor-limit
/// fallback, or a shared argument's device address) every format decodes as the reference at an
/// unaligned bind offset and element, without and with containment, and the displacement starts at
/// the root entry's bind offset (no residual is added).
#[test]
fn every_format_decodes_through_a_device_address() {
    let (bind, at) = (5u64, 5 + 3 * 18 + 2);
    for (n, &format) in VertexFormat::ALL.iter().enumerate() {
        let Ok(e) = encoding(format) else { continue };
        let data = pattern(96, n as u32 + 101);
        let want = reference(e, &data[at..]);
        let form = if e.is_float() { float(4) } else { int(32, 4) };
        for contained in [false, true] {
            let m = module_from(Source::Psb { contained }, e, form, Elem::Direct, 2);
            // The residual parameter is not read.
            let got = run_psb(&m, &data, bind, [3, 0, 18, 0xDEAD], form.lanes);
            check_lanes(
                &format!("{format:?} contained={contained}"),
                form,
                &got,
                want,
            );
        }
    }
}

/// For device addresses: a read past the allocation is redirected to the sink and yields zero,
/// including a displacement that wraps `u32` and one starting inside the allocation's last bytes.
#[test]
fn an_out_of_bounds_device_address_read_is_zero() {
    let data = pattern(40, 17);
    let e = encoding(VertexFormat::Float3).expect("encoding");
    let form = float(4);
    let m = module_from(Source::Psb { contained: true }, e, form, Elem::Direct, 0);
    let zero = reference(e, &[0; 12]);
    for (index, stride, bind) in [
        (2u32, 12u32, 8u64),
        (1, 21, 8),
        (0, 0, 29),
        (0x1555_5556, 12, 0),
        (1, 0xFFFF_FFF8, 0),
    ] {
        let got = run_psb(&m, &data, bind, [index, 0, stride, 0], 4);
        let at = bind + u64::from(index.wrapping_mul(stride));
        let want = if at + 12 <= 40 {
            reference(e, &data[at as usize..])
        } else {
            zero
        };
        check_lanes(&format!("{index} {stride} {bind}"), form, &got, want);
    }
    let got = run_psb(&m, &data, 4, [2, 0, 12, 0], 4);
    check_lanes("last", form, &got, reference(e, &data[28..]));
}

/// The encodings' sizes and classes agree with the Vulkan formats the fixed-function path maps
/// the same Metal formats to.
#[test]
fn encodings_agree_with_the_vulkan_vertex_formats() {
    for &format in VertexFormat::ALL {
        let Ok(e) = encoding(format) else { continue };
        let vk = crate::metal::vertex::vk_vertex_format(format).expect("vk format");
        let name = vk.name().expect("name");
        assert_eq!(
            e.is_float(),
            !(name.ends_with("_UINT") || name.ends_with("_SINT")),
            "{format:?} {name}"
        );
        let letters = name
            .trim_start_matches("VK_FORMAT_")
            .split('_')
            .next()
            .expect("components")
            .chars()
            .filter(|c| matches!(c, 'R' | 'G' | 'B' | 'A'))
            .count();
        assert_eq!(usize::from(e.components()), letters, "{format:?} {name}");
        let bits: u32 = name
            .trim_start_matches("VK_FORMAT_")
            .split('_')
            .next()
            .expect("components")
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|d| d.parse::<u32>().ok())
            .sum();
        assert_eq!(e.bytes() * 8, bits, "{format:?} {name}");
    }
}

/// The opcode of every instruction in a module's words.
fn opcodes(m: TypedModule) -> Vec<u32> {
    let words = crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .words;
    let mut at = 5;
    let mut ops = Vec::new();
    while let Some(w) = words.get(at) {
        ops.push(w & 0xFFFF);
        at += (w >> 16).max(1) as usize;
    }
    ops
}

/// The normalized encodings: integer fields over `2^b - 1` (or `2^(b-1) - 1`).
fn normalized_encodings() -> Vec<(VertexFormat, Encoding)> {
    VertexFormat::ALL
        .iter()
        .filter_map(|&f| Some((f, encoding(f).ok()?)))
        .filter(|(_, e)| {
            matches!(
                e,
                Encoding::Int {
                    normalized: true,
                    ..
                } | Encoding::Bgra8Unorm
                    | Encoding::Rgb10A2 { .. }
            )
        })
        .collect()
}

/// A normalized decode divides with integer operations, never a device `OpFDiv`, which may be an
/// ULP off where Metal's conversion is correctly rounded.
#[test]
fn a_normalized_decode_has_no_float_division() {
    let formats = normalized_encodings();
    assert_eq!(formats.len(), 19, "{formats:?}");
    for (format, e) in formats {
        for check in [Check::None, Check::InShader] {
            let m = module(e, float(4), Elem::Direct, 0, check);
            let ops = opcodes(m);
            assert!(ops.contains(&(crate::spv::Op::UDiv as u32)), "{format:?}");
            assert!(
                !ops.contains(&(crate::spv::Op::FDiv as u32)),
                "{format:?} {check:?} decodes with OpFDiv"
            );
        }
    }
}

/// `c / d` correctly rounded to `f32` (`f64` holds the quotient to 53 bits, so its rounding to
/// `f32` is the correctly rounded one), signed codes clamped to `-1`.
fn exact_quotient(c: i64, d: u32) -> f64 {
    f64::from(((c as f64 / f64::from(d)).max(-1.0)) as f32)
}

/// Every byte decodes as UNORM8 / SNORM8 (and BGRA8's reordered UNORM8) to the correctly rounded
/// quotient.
#[test]
fn every_normalized_byte_is_correctly_rounded() {
    let data: Vec<u8> = (0..=255u8).collect();
    for (format, signed, order) in [
        (VertexFormat::UChar4Normalized, false, [0, 1, 2, 3]),
        (VertexFormat::Char4Normalized, true, [0, 1, 2, 3]),
        (VertexFormat::UChar4NormalizedBGRA, false, [2, 1, 0, 3]),
    ] {
        let e = encoding(format).expect("encoding");
        let m = module(e, float(4), Elem::Direct, 0, Check::InShader);
        for i in 0..64u32 {
            let got = run(&m, &data, [i, 0, 4, 0], 4);
            let want = order.map(|k| {
                let b = data[4 * i as usize + k];
                let c = if signed {
                    i64::from(b as i8)
                } else {
                    i64::from(b)
                };
                exact_quotient(c, if signed { 127 } else { 255 })
            });
            assert_eq!(want, reference(e, &data[4 * i as usize..]), "{format:?}");
            check_lanes(&format!("{format:?} element {i}"), float(4), &got, want);
        }
    }
}

/// Every 10-bit and 2-bit code of the packed 10:10:10:2 formats decodes to the correctly rounded
/// quotient, signed codes (the most negative one `-1`) and unsigned.
#[test]
fn every_ten_and_two_bit_code_is_correctly_rounded() {
    let n = 1024u32.div_ceil(3);
    let words: Vec<u32> = (0..n)
        .map(|i| {
            let code = |k: u32| (3 * i + k) % 1024;
            code(0) | (code(1) << 10) | (code(2) << 20) | (i << 30)
        })
        .collect();
    let data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    for (format, signed) in [
        (VertexFormat::UInt1010102Normalized, false),
        (VertexFormat::Int1010102Normalized, true),
    ] {
        let e = encoding(format).expect("encoding");
        let m = module(e, float(4), Elem::Direct, 0, Check::InShader);
        for (i, w) in words.iter().enumerate() {
            let got = run(&m, &data, [i as u32, 0, 4, 0], 4);
            let want = [(0, 10u32), (10, 10), (20, 10), (30, 2)].map(|(at, bits)| {
                let c = (w >> at) & ((1 << bits) - 1);
                if signed {
                    let s = 32 - bits;
                    let c = i64::from(((c << s) as i32) >> s);
                    exact_quotient(c, (1 << (bits - 1)) - 1)
                } else {
                    exact_quotient(i64::from(c), (1 << bits) - 1)
                }
            });
            assert_eq!(want, reference(e, &data[4 * i..]), "{format:?}");
            check_lanes(&format!("{format:?} element {i}"), float(4), &got, want);
        }
    }
}

/// A narrow float's widening to a `float` parameter answers Metal's canonical NaN, sign and
/// payload dropped, while a `half` parameter keeps the fetched `half`'s own bits.
///
/// Metal's answer, from `vertex-format-float4`'s and `attribute-format-fetches`' recordings of
/// plain Metal on an M3 Max under macOS 27.0.1: the `half` codes `0x7E01` and `0xFE01` both reach
/// a `float` attribute as `0x7FC00000`, and so does a `FloatRG11B10` field whose 5-bit exponent
/// is all ones; the same `half` codes reach a `half` attribute as themselves
/// (`vertex-format-half4`). Every finite, zero, subnormal and infinite code is bit-exact either
/// way, which `every_format_decodes_exactly` holds.
#[test]
fn a_narrow_floats_widening_answers_metals_canonical_nan() {
    const CANONICAL: u32 = 0x7FC0_0000;
    /// The `f32` bits of a `half` whose exponent field is all ones, by the IEEE widening: the
    /// sign, an all-ones `f32` exponent, and the 10-bit mantissa in the top of the 23-bit one.
    fn widened(h: u16) -> u32 {
        (u32::from(h) & 0x8000) << 16 | 0x7F80_0000 | (u32::from(h) & 0x3FF) << 13
    }
    // half4: a quiet payload NaN, its negative, a signalling NaN, and +inf.
    let halves = [0x7E01u16, 0xFE01, 0x7C01, 0x7C00];
    let mut data = vec![0u8; 32];
    for (i, h) in halves.iter().enumerate() {
        data[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
    }
    // The `FloatRG11B10` word whose three fields each hold an all-ones exponent and a non-zero
    // mantissa: two 11-bit fields of six mantissa bits and a 10-bit one of five.
    let rg11b10 = 0x7FFu32 | 0x7FF << 11 | 0x3FF << 22;
    data[8..12].copy_from_slice(&rg11b10.to_le_bytes());
    for check in [Check::None, Check::InShader] {
        let e = encoding(VertexFormat::Half4).expect("encoding");
        let m = module(e, float(4), Elem::Constant, 0, check);
        let got = run(&m, &data, [0, 0, 0, 0], 4);
        assert_eq!(
            got,
            [CANONICAL, CANONICAL, CANONICAL, f32::INFINITY.to_bits()],
            "Half4 into float4 {check:?}"
        );
        // The same codes into a `half` parameter, held widened: their own bits, as Metal's
        // `half` attribute keeps them.
        let m = module(e, half_form(4), Elem::Constant, 0, check);
        let got = run(&m, &data, [0, 0, 0, 0], 4);
        assert_eq!(got, halves.map(widened), "Half4 into half4 {check:?}");
        let e = encoding(VertexFormat::FloatRG11B10).expect("encoding");
        let m = module(e, float(4), Elem::Constant, 8, check);
        let got = run(&m, &data, [0, 0, 0, 0], 4);
        assert_eq!(
            got,
            [CANONICAL, CANONICAL, CANONICAL, 1.0f32.to_bits()],
            "FloatRG11B10 into float4 {check:?}"
        );
    }
}
