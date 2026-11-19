//! The control stage's factor rules run in the interpreter against a model of Metal's stated rules:
//! the half decode over every `half`, the triangle and quad layouts, the cull, the scale, each
//! partition's clamp, `pow2`'s rounding, the step functions' addresses, the pushed residual, and
//! containment; and its patch index and control-point elements against Metal's (Metal probes on an
//! M4).

use super::*;
use crate::device::requirements::VkFeature;
use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::pipeline::sym::ResourceKind;
use crate::spirv::interp::{run, Buffer, Config, Resources};

fn out_slot() -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index: 0,
    }
}

/// A `Block { uint[] }` storage buffer at the output slot.
fn out_buffer(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let words = mb.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[words])?;
    let block = mb.ty(Type::Struct {
        members: [words].into(),
        layout: Some(layout),
    })?;
    mb.resource(out_slot(), StorageClass::StorageBuffer, block)
}

/// Store the bits of the `float` `v` as word `index` of `out`.
fn store(fb: &mut FunctionBuilder<'_>, out: GlobalId, index: Value, v: Value) -> BResult<()> {
    let u32_ = uint(fb)?;
    let bits = fb.convert(ConvOp::Bitcast, v, u32_)?;
    let base = fb.module().global_value(out)?;
    let zero = ku(fb, 0)?;
    let ptr = fb.access_chain(base, &[zero, index])?;
    fb.store(ptr, bits, MemAccess::default())
}

/// `GlobalInvocationId` component `lane`.
fn global_id(fb: &mut FunctionBuilder<'_>, lane: u32) -> BResult<Value> {
    let u32_ = uint(fb)?;
    let v3 = fb.module().ty(Type::Vector { elem: u32_, len: 3 })?;
    let g = fb
        .module()
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, v3)?;
    let ptr = fb.module().global_value(g)?;
    let v = fb.load(ptr, MemAccess::default())?;
    fb.composite_extract(v, &[lane])
}

/// A compute module of one invocation per workgroup running `body`, which declares its globals
/// and returns the entry's body.
fn kernel(
    body: impl FnOnce(
        &mut ModuleBuilder,
    ) -> BResult<Box<dyn FnOnce(&mut FunctionBuilder<'_>) -> BResult<()>>>,
) -> TypedModule {
    let mut mb = ModuleBuilder::new();
    let run_body = body(&mut mb).expect("declarations");
    let void = mb.ty(Type::Void).expect("void");
    let mut fb = mb.function(void).expect("function");
    run_body(&mut fb).expect("body");
    fb.ret(None).expect("ret");
    let f = fb.finish().expect("finish");
    mb.entry(
        ExecModel::GLCompute,
        f,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    mb.finalize().expect("finalize")
}

fn le_words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Every finite non-negative `half` decodes to its exact `float` (subnormals included), and the
/// bits of an infinity or a NaN, which the control stage never decodes, still decode to a finite
/// `float`.
#[test]
fn every_finite_positive_half_decodes_exactly() {
    let m = kernel(|mb| {
        let input = {
            let u32_ = mb.ty(Type::Int {
                width: 32,
                signed: false,
            })?;
            let words = mb.ty(Type::RuntimeArray {
                elem: u32_,
                stride: Some(4),
            })?;
            let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[words])?;
            let block = mb.ty(Type::Struct {
                members: [words].into(),
                layout: Some(layout),
            })?;
            mb.resource(FACTOR_SLOT, StorageClass::StorageBuffer, block)?
        };
        let out = out_buffer(mb)?;
        Ok(Box::new(move |fb: &mut FunctionBuilder<'_>| {
            let i = global_id(fb, 0)?;
            let w = word(fb, input, i)?;
            let mask = ku(fb, 0xffff)?;
            let lo = fb.binary(BinOp::BitwiseAnd, w, mask)?;
            let sixteen = ku(fb, 16)?;
            let hi = fb.binary(BinOp::ShiftRightLogical, w, sixteen)?;
            let two = ku(fb, 2)?;
            let at = fb.binary(BinOp::IMul, i, two)?;
            let lo = half_value(fb, lo)?;
            store(fb, out, at, lo)?;
            let one = ku(fb, 1)?;
            let at = fb.binary(BinOp::IAdd, at, one)?;
            let hi = half_value(fb, hi)?;
            store(fb, out, at, hi)
        }))
    });
    let input: Vec<u32> = (0..0x4000u32)
        .map(|i| (2 * i) | ((2 * i + 1) << 16))
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(FACTOR_SLOT),
                address: None,
                bytes: le_words(&input),
            },
            Buffer {
                slot: Some(out_slot()),
                address: None,
                bytes: vec![0; 0x8000 * 4],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [0x4000, 1, 1],
        step_limit: u64::MAX,
        ..Config::default()
    };
    let o = run(&m, &cfg, res).expect("run");
    for (h, got) in words_of(&o.buffers[1]).into_iter().enumerate() {
        let want = f16_bits_to_f32(h as u16);
        if want.is_finite() {
            assert_eq!(got, want.to_bits(), "half {h:#06x}");
        } else {
            assert!(f32::from_bits(got).is_finite(), "half {h:#06x}: {got:#x}");
        }
    }
}

/// Metal's stated rules for one patch's `halves` (edges then insides) under `c` and `scale`:
/// the outer and inner levels Vulkan receives.
fn model(c: &Control, halves: &[u16], scale: f32) -> (Vec<f32>, Vec<f32>) {
    let (edges, _) = c.counts();
    let values: Vec<f32> = halves.iter().map(|&h| f16_bits_to_f32(h)).collect();
    let culled = values[..edges].iter().any(|v| v.is_nan() || *v <= 0.0);
    let (lo, hi) = c.range();
    let levels: Vec<f32> = values
        .iter()
        .map(|&v| {
            let v = if c.scale_enabled { v * scale } else { v };
            let v = if v >= lo { v } else { lo };
            let v = if v <= hi { v } else { hi };
            match c.partition {
                Partition::Pow2 => (v.ceil() as u32).next_power_of_two() as f32,
                Partition::Integer if c.scale_enabled => v.ceil(),
                _ => v,
            }
        })
        .collect();
    let outer = if culled {
        vec![0.0; edges]
    } else {
        levels[..edges].to_vec()
    };
    (outer, levels[edges..].to_vec())
}

fn control(domain: PatchDomain, partition: Partition, max_factor: u32) -> Control {
    Control {
        domain,
        partition,
        winding: Winding::Clockwise,
        max_factor,
        step: Step::PerPatch,
        scale_enabled: false,
        push_offset: 0,
        control_points: 0,
        index_type: IndexType::None,
        addressed: false,
    }
}

/// The device address of the factors in the address form's runs: past 2^32, so a read that
/// dropped the high word would miss them.
const FACTORS: u64 = 0x0000_0012_3450_0000;

/// The factor buffer's words for `patches` patches of `halves` each, packed from `offset`
/// bytes, each patch `Control::factor_bytes` apart.
fn pack(domain: PatchDomain, offset: u32, patches: &[Vec<u16>]) -> Vec<u32> {
    let size = Control::factor_bytes(domain) as usize;
    let mut bytes = vec![0u8; offset as usize + size * patches.len()];
    for (p, halves) in patches.iter().enumerate() {
        for (k, h) in halves.iter().enumerate() {
            let at = offset as usize + p * size + 2 * k;
            bytes[at..at + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    words_of(&bytes)
}

/// The levels [`levels`] computes for patches `0..patches` of instances `0..instances` with
/// `factors` bound and `push` (residual offset, instance stride, scale) pushed: per invocation,
/// the outer levels then the inner ones, as `float`s. In the address form the factors are at
/// [`FACTORS`], and the pushed address is it plus the offset.
fn run_levels(
    c: &Control,
    factors: &[u32],
    push: (u32, u32, f32),
    patches: u32,
    instances: u32,
) -> Vec<Vec<f32>> {
    let (edges, insides) = c.counts();
    let per = (edges + insides) as u32;
    let cc = *c;
    let m = kernel(move |mb| {
        let src = declare_sources(mb, &cc)?;
        let out = out_buffer(mb)?;
        Ok(Box::new(move |fb: &mut FunctionBuilder<'_>| {
            let patch = global_id(fb, 0)?;
            let instance = global_id(fb, 1)?;
            let (outer, inner) = levels(fb, &cc, &src, patch, instance)?;
            let count = ku(fb, patches)?;
            let row = fb.binary(BinOp::IMul, instance, count)?;
            let row = fb.binary(BinOp::IAdd, row, patch)?;
            let width = ku(fb, per)?;
            let first = fb.binary(BinOp::IMul, row, width)?;
            for (k, v) in outer.into_iter().chain(inner).enumerate() {
                let offset = ku(fb, k as u32)?;
                let at = fb.binary(BinOp::IAdd, first, offset)?;
                store(fb, out, at, v)?;
            }
            Ok(())
        }))
    });
    let mut push_bytes = vec![0u8; c.push_offset as usize];
    let (slot, address) = match c.addressed {
        false => {
            push_bytes.extend(push.0.to_le_bytes());
            push_bytes.extend(push.1.to_le_bytes());
            push_bytes.extend(push.2.to_bits().to_le_bytes());
            push_bytes.extend([0; 12]);
            (Some(FACTOR_SLOT), None)
        }
        true => {
            push_bytes.extend((FACTORS + u64::from(push.0)).to_le_bytes());
            push_bytes.extend(push.1.to_le_bytes());
            push_bytes.extend(push.2.to_bits().to_le_bytes());
            (None, Some(FACTORS))
        }
    };
    let res = Resources {
        buffers: vec![
            Buffer {
                slot,
                address,
                bytes: le_words(factors),
            },
            Buffer {
                slot: Some(out_slot()),
                address: None,
                bytes: vec![0; (patches * instances * per * 4) as usize],
            },
            Buffer {
                slot: Some(PATCH_INDEX_SLOT),
                address: None,
                bytes: vec![0; 16],
            },
        ],
        push: push_bytes,
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [patches, instances, 1],
        step_limit: u64::MAX,
        ..Config::default()
    };
    let o = run(&m, &cfg, res).expect("run");
    words_of(&o.buffers[1])
        .chunks(per as usize)
        .map(|w| w.iter().map(|b| f32::from_bits(*b)).collect())
        .collect()
}

fn expect(c: &Control, halves: &[u16], scale: f32) -> Vec<f32> {
    let (outer, inner) = model(c, halves, scale);
    outer.into_iter().chain(inner).collect()
}

fn h(v: f32) -> u16 {
    f32_to_f16_bits(v)
}

/// Each domain's layout: edge `k` is outer level `k`, inside `k` inner level `k`, halves low
/// first; values inside the range pass unchanged (integer partition, max 64).
#[test]
fn factor_layouts_map_lane_for_lane() {
    for (domain, halves) in [
        (
            PatchDomain::Triangle,
            vec![h(2.0), h(3.5), h(5.25), h(7.75)],
        ),
        (
            PatchDomain::Quad,
            vec![h(2.0), h(3.5), h(5.25), h(7.75), h(9.5), h(11.0)],
        ),
    ] {
        let c = control(domain, Partition::Integer, 64);
        let got = run_levels(
            &c,
            &pack(domain, 0, std::slice::from_ref(&halves)),
            (0, 0, 1.0),
            1,
            1,
        );
        let values: Vec<f32> = halves.iter().map(|&x| f16_bits_to_f32(x)).collect();
        assert_eq!(got, vec![values], "{domain:?}");
    }
}

/// A zero, negative, negative-zero, or NaN edge factor culls (every outer level 0), an inside
/// factor never does, and a subnormal positive edge factor does not.
#[test]
fn a_non_positive_or_nan_edge_factor_culls() {
    let nan = 0x7e00;
    let cases: Vec<Vec<u16>> = vec![
        vec![h(0.0), h(4.0), h(4.0), h(4.0)],
        vec![h(4.0), h(-1.0), h(4.0), h(4.0)],
        vec![h(4.0), h(4.0), 0x8000, h(4.0)],
        vec![h(4.0), nan, h(4.0), h(4.0)],
        vec![h(4.0), h(4.0), h(4.0), nan],
        vec![h(4.0), h(4.0), h(4.0), h(-3.0)],
        vec![0x0001, h(4.0), h(4.0), h(4.0)],
        vec![h(4.0), h(4.0), h(4.0), 0xfc00],
    ];
    let c = control(PatchDomain::Triangle, Partition::Integer, 16);
    let got = run_levels(
        &c,
        &pack(PatchDomain::Triangle, 0, &cases),
        (0, 0, 1.0),
        cases.len() as u32,
        1,
    );
    let culled: Vec<bool> = got.iter().map(|l| l[..3] == [0.0; 3]).collect();
    assert_eq!(culled, [true, true, true, true, false, false, false, false]);
    for (halves, got) in cases.iter().zip(&got) {
        assert_eq!(got, &expect(&c, halves, 1.0), "{halves:x?}");
    }
    // A NaN or negative inside factor takes the range's lower end; a subnormal edge factor
    // clamps up to it.
    assert_eq!(got[4][3], 1.0);
    assert_eq!(got[5][3], 1.0);
    assert_eq!(got[6][0], 1.0);
}

/// Each partition's range and `pow2`'s rounding, for a quad, at several maxima.
#[test]
fn each_partition_clamps_and_rounds_as_metal_states() {
    let values = [
        0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.5, 7.0, 9.0, 15.0, 16.0, 17.0, 31.5, 33.0, 64.0,
        100.0, 65504.0,
    ];
    let infinity = 0x7c00;
    let mut halves: Vec<u16> = values.iter().map(|&v| h(v)).collect();
    halves.push(infinity);
    // Six lanes a patch: pad to whole patches with 1.0.
    while !halves.len().is_multiple_of(6) {
        halves.push(h(1.0));
    }
    let patches: Vec<Vec<u16>> = halves.chunks(6).map(<[u16]>::to_vec).collect();
    for partition in [
        Partition::Pow2,
        Partition::Integer,
        Partition::FractionalOdd,
        Partition::FractionalEven,
    ] {
        for max in [2, 16, 64] {
            let c = control(PatchDomain::Quad, partition, max);
            let got = run_levels(
                &c,
                &pack(PatchDomain::Quad, 0, &patches),
                (0, 0, 1.0),
                patches.len() as u32,
                1,
            );
            for (p, row) in patches.iter().zip(&got) {
                assert_eq!(row, &expect(&c, p, 1.0), "{partition:?} max {max}: {p:x?}");
            }
        }
    }
    // Spot values: `pow2` rounds 5.5 up to 8 and 17 to 32; fractional odd's top is max - 1.
    let c = control(PatchDomain::Quad, Partition::Pow2, 64);
    let row = run_levels(
        &c,
        &pack(
            PatchDomain::Quad,
            0,
            &[vec![h(5.5), h(17.0), h(1.0), h(3.0), h(0.5), h(64.0)]],
        ),
        (0, 0, 1.0),
        1,
        1,
    );
    assert_eq!(row[0], [8.0, 32.0, 1.0, 4.0, 1.0, 64.0]);
    let c = control(PatchDomain::Quad, Partition::FractionalOdd, 16);
    let row = run_levels(
        &c,
        &pack(
            PatchDomain::Quad,
            0,
            &[vec![h(20.0), h(0.5), h(15.5), h(3.0), h(1.0), h(16.0)]],
        ),
        (0, 0, 1.0),
        1,
        1,
    );
    assert_eq!(row[0], [15.0, 1.0, 15.0, 3.0, 1.0, 15.0]);
    let c = control(PatchDomain::Quad, Partition::FractionalEven, 16);
    let row = run_levels(
        &c,
        &pack(
            PatchDomain::Quad,
            0,
            &[vec![h(1.0), h(0.5), h(15.5), h(3.0), h(1.5), h(99.0)]],
        ),
        (0, 0, 1.0),
        1,
        1,
    );
    assert_eq!(row[0], [2.0, 2.0, 15.5, 3.0, 2.0, 16.0]);
}

/// With the scale enabled every factor is multiplied before the clamp, and a culled patch stays
/// culled whatever the scale; without it the pushed scale is not read.
#[test]
fn the_scale_multiplies_before_the_clamp() {
    let patches = vec![
        vec![h(2.0), h(3.0), h(4.0), h(1.5)],
        vec![h(0.0), h(3.0), h(4.0), h(1.5)],
        vec![h(40.0), h(0.25), h(4.0), h(0.5)],
    ];
    for (enabled, scale) in [(true, 0.5), (true, 3.0), (false, 3.0)] {
        let mut c = control(PatchDomain::Triangle, Partition::FractionalEven, 64);
        c.scale_enabled = enabled;
        let got = run_levels(
            &c,
            &pack(PatchDomain::Triangle, 0, &patches),
            (0, 0, scale),
            3,
            1,
        );
        for (p, row) in patches.iter().zip(&got) {
            assert_eq!(
                row,
                &expect(&c, p, scale),
                "scale {enabled} {scale}: {p:x?}"
            );
        }
    }
}

/// A scaled `integer` level is rounded up to an integer by the stage, so a driver that holds the
/// levels at `half` precision (MoltenVK) takes it exactly: factor `0x47F3` by the `half` scale
/// `0x3007` is 1.000445 in `float`, which `half` rounds to 1.0 (one segment), where Metal draws
/// two; the stage writes 2. Unscaled, a level is the factor, a `half` already; a scaled fractional
/// level stays the product.
#[test]
fn a_scaled_integer_level_is_rounded_up_to_a_half() {
    let scale = f16_bits_to_f32(0x3007);
    let factor = f16_bits_to_f32(0x47f3);
    assert_eq!(f16_bits_to_f32(f32_to_f16_bits(factor * scale)), 1.0);
    let patch = vec![vec![0x47f3, 0x47f3, 0x47f3, 0x47f3]];
    for (partition, enabled, level) in [
        (Partition::Integer, true, 2.0),
        (Partition::Integer, false, factor),
        (Partition::FractionalOdd, true, factor * scale),
    ] {
        let mut c = control(PatchDomain::Triangle, partition, 64);
        c.scale_enabled = enabled;
        let got = run_levels(
            &c,
            &pack(PatchDomain::Triangle, 0, &patch),
            (0, 0, scale),
            1,
            1,
        );
        assert_eq!(got[0], [level; 4], "{partition:?} scaled {enabled}");
    }
}

/// Each step function's address: the pushed residual, `+ patch · size` per patch,
/// `+ instance · stride` per instance; and a factor past the bound range reads 0, which culls.
#[test]
fn each_step_function_addresses_its_factors() {
    let domain = PatchDomain::Quad;
    let (patches, instances) = (3u32, 2u32);
    let (offset, stride) = (8u32, 40u32);
    // Patch `p` of instance `i` at `offset + p · 12 + i · stride` holds factors `1 + 2p + 7i`.
    let value = |p: u32, i: u32| h(1.0 + 2.0 * p as f32 + 7.0 * i as f32);
    let mut bytes = vec![0u8; (offset + stride * instances + 12 * patches) as usize];
    for i in 0..instances {
        for p in 0..patches {
            for k in 0..6 {
                let at = (offset + stride * i + 12 * p + 2 * k) as usize;
                bytes[at..at + 2].copy_from_slice(&value(p, i).to_le_bytes());
            }
        }
    }
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let factors = words_of(&bytes);
    for step in [
        Step::Constant,
        Step::PerPatch,
        Step::PerInstance,
        Step::PerPatchAndPerInstance,
    ] {
        let mut c = control(domain, Partition::FractionalEven, 64);
        c.step = step;
        c.push_offset = 20;
        let got = run_levels(&c, &factors, (offset, stride, 1.0), patches, instances);
        for i in 0..instances {
            for p in 0..patches {
                let (pp, ii) = match step {
                    Step::Constant => (0, 0),
                    Step::PerPatch => (p, 0),
                    Step::PerInstance => (0, i),
                    _ => (p, i),
                };
                let want = expect(&c, &[value(pp, ii); 6], 1.0);
                assert_eq!(got[(i * patches + p) as usize], want, "{step:?} {p} {i}");
            }
        }
    }
    // Only the first patch's factors bound: the second reads 0 and culls.
    let c = control(domain, Partition::Integer, 64);
    let factors = pack(domain, 0, &[vec![h(3.0); 6]]);
    let got = run_levels(&c, &factors, (0, 0, 1.0), 2, 1);
    assert_eq!(got[0], [3.0; 6]);
    assert_eq!(got[1], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
}

/// Factors at any even byte are read from that byte, each `half` its own: an offset or an instance
/// stride 2 past a multiple of 4, which plain Metal reads at that byte (Metal probe case
/// `factor_offset_2` on an M4), as one at a multiple of 4, for each step function and both domains.
/// Each `half` of the buffer is a distinct factor, so a read one `half` off answers other levels.
#[test]
fn factors_at_an_even_byte_are_read_from_it() {
    let (patches, instances) = (2u32, 2u32);
    // `half` `j` of the buffer is the factor `1 + j`, up to 48 of them.
    let bytes: Vec<u8> = (0..48u16)
        .flat_map(|j| h(1.0 + f32::from(j)).to_le_bytes())
        .collect();
    let factors = words_of(&bytes);
    for domain in [PatchDomain::Triangle, PatchDomain::Quad] {
        let size = Control::factor_bytes(domain);
        let (edges, insides) = control(domain, Partition::Integer, 64).counts();
        for (offset, stride) in [(2u32, 2u32), (6, 30), (0, 26), (4, 2), (10, 0)] {
            for step in [
                Step::Constant,
                Step::PerPatch,
                Step::PerInstance,
                Step::PerPatchAndPerInstance,
            ] {
                let mut c = control(domain, Partition::Integer, 64);
                c.step = step;
                let got = run_levels(&c, &factors, (offset, stride, 1.0), patches, instances);
                for i in 0..instances {
                    for p in 0..patches {
                        let mut at = offset;
                        if matches!(step, Step::PerPatch | Step::PerPatchAndPerInstance) {
                            at += p * size;
                        }
                        if matches!(step, Step::PerInstance | Step::PerPatchAndPerInstance) {
                            at += i * stride;
                        }
                        let first = (at / 2) as usize;
                        let halves: Vec<u16> = (first..first + edges + insides)
                            .map(|j| h(1.0 + j as f32))
                            .collect();
                        let want = expect(&c, &halves, 1.0);
                        assert_eq!(
                            got[(i * patches + p) as usize],
                            want,
                            "{domain:?} {step:?} offset {offset} stride {stride} patch {p} \
                             instance {i}"
                        );
                    }
                }
            }
        }
    }
}

/// The levels of `c`'s address form (its push offset rounded up to 8, the address's alignment)
/// for the same factors and push as [`run_levels`] gives its bound form, which they must equal.
fn run_both(
    c: &Control,
    factors: &[u32],
    push: (u32, u32, f32),
    patches: u32,
    instances: u32,
) -> Vec<Vec<f32>> {
    let bound = run_levels(c, factors, push, patches, instances);
    let mut a = *c;
    a.addressed = true;
    a.push_offset = c.push_offset.next_multiple_of(8);
    let addressed = run_levels(&a, factors, push, patches, instances);
    assert_eq!(addressed, bound, "{c:?}");
    addressed
}

/// The address form reads the factors the bound form reads, at the same bytes, under the same
/// rules: every factor class under each partition and a scale, each step function at offsets and
/// instance strides at a multiple of 4 and 2 past one, both domains, from an address past 2^32.
#[test]
fn the_address_form_reads_what_the_bound_form_reads() {
    let mut values: Vec<u16> = (0..32u16)
        .flat_map(|e| [0, 1, 0x200, 0x3ff].map(move |m| (e << 10) | m))
        .chain((0..0x8000).step_by(97))
        .collect();
    values.sort_unstable();
    values.dedup();
    let patches: Vec<Vec<u16>> = values
        .iter()
        .map(|&x| vec![x, h(4.0), h(4.0), h(4.0), x, x | 0x8000])
        .collect();
    let factors = pack(PatchDomain::Quad, 0, &patches);
    for (partition, max, scale) in [
        (Partition::Pow2, 16, Some(0.5)),
        (Partition::Integer, 64, None),
        (Partition::FractionalOdd, 64, None),
        (Partition::FractionalOdd, 64, Some(3.0)),
        (Partition::FractionalEven, 8, Some(1024.0)),
    ] {
        let mut c = control(PatchDomain::Quad, partition, max);
        c.scale_enabled = scale.is_some();
        let scale = scale.unwrap_or(1.0);
        let got = run_both(&c, &factors, (0, 0, scale), patches.len() as u32, 1);
        for (p, row) in patches.iter().zip(&got) {
            assert_eq!(row, &expect(&c, p, scale), "{partition:?}: {:#06x}", p[0]);
        }
    }
    let bytes: Vec<u8> = (0..48u16)
        .flat_map(|j| h(1.0 + f32::from(j)).to_le_bytes())
        .collect();
    let factors = words_of(&bytes);
    for domain in [PatchDomain::Triangle, PatchDomain::Quad] {
        for (offset, stride) in [(2u32, 2u32), (6, 30), (0, 26), (4, 2), (10, 0)] {
            for step in [
                Step::Constant,
                Step::PerPatch,
                Step::PerInstance,
                Step::PerPatchAndPerInstance,
            ] {
                let mut c = control(domain, Partition::FractionalOdd, 64);
                c.step = step;
                c.push_offset = 20;
                run_both(&c, &factors, (offset, stride, 1.0), 2, 2);
            }
        }
    }
}

/// The address form's push fields: the address at the stage's offset, a `u64`, and the stride and
/// the scale where the stage reads them; its push range ends at the last.
#[test]
fn the_address_form_pushes_the_address_and_what_it_reads() {
    use crate::pipeline::sym::PushField as F;
    for (step, scale, fields, bytes) in [
        (
            Step::PerPatch,
            false,
            vec![(F::TessellationFactorAddress, 0)],
            8,
        ),
        (
            Step::Constant,
            true,
            vec![
                (F::TessellationFactorAddress, 0),
                (F::TessellationFactorScale, 12),
            ],
            16,
        ),
        (
            Step::PerInstance,
            false,
            vec![
                (F::TessellationFactorAddress, 0),
                (F::TessellationFactorInstanceStride, 8),
            ],
            12,
        ),
    ] {
        assert_eq!(address_fields(step, scale), fields, "{step:?} {scale}");
        assert_eq!(Control::push_bytes(true, step, scale), (bytes, 8));
        assert_eq!(Control::push_bytes(false, step, scale), (PUSH_BYTES, 4));
    }
}

/// In the address form the patch index is the patch's position: the form binds no patch index
/// buffer.
#[test]
fn the_address_form_s_patch_index_is_the_position() {
    let mut c = control(PatchDomain::Triangle, Partition::Integer, 64);
    c.addressed = true;
    let m = kernel(move |mb| {
        let src = declare_sources(mb, &c)?;
        assert!(src.factors.is_none() && src.patch_indices.is_none());
        let out = out_buffer(mb)?;
        Ok(Box::new(move |fb: &mut FunctionBuilder<'_>| {
            let position = global_id(fb, 0)?;
            let patch = patch_index(fb, &src, position)?;
            let f32_ = float(fb)?;
            let as_float = fb.convert(ConvOp::Bitcast, patch, f32_)?;
            store(fb, out, position, as_float)
        }))
    });
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(out_slot()),
            address: None,
            bytes: vec![0; 16],
        }],
        push: vec![0; 8],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [4, 1, 1],
        step_limit: u64::MAX,
        ..Config::default()
    };
    let o = run(&m, &cfg, res).expect("run");
    assert_eq!(words_of(&o.buffers[0]), [0, 1, 2, 3]);
}

/// The patch index of each position `0..positions` with `indices` bound as the patch index
/// buffer and `push` (its residual offset, whether the draw binds one) pushed.
fn run_patch_index(indices: &[u32], push: (u32, u32), positions: u32) -> Vec<u32> {
    let c = control(PatchDomain::Triangle, Partition::Integer, 64);
    let m = kernel(move |mb| {
        let src = declare_sources(mb, &c)?;
        let out = out_buffer(mb)?;
        Ok(Box::new(move |fb: &mut FunctionBuilder<'_>| {
            let position = global_id(fb, 0)?;
            let patch = patch_index(fb, &src, position)?;
            let base = fb.module().global_value(out)?;
            let zero = ku(fb, 0)?;
            let ptr = fb.access_chain(base, &[zero, position])?;
            fb.store(ptr, patch, MemAccess::default())
        }))
    });
    let mut push_bytes = vec![0u8; c.push_offset as usize + 12];
    push_bytes.extend(push.0.to_le_bytes());
    push_bytes.extend(push.1.to_le_bytes());
    push_bytes.extend([0; 4]);
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(FACTOR_SLOT),
                address: None,
                bytes: vec![0; 16],
            },
            Buffer {
                slot: Some(out_slot()),
                address: None,
                bytes: vec![0; (positions * 4) as usize],
            },
            Buffer {
                slot: Some(PATCH_INDEX_SLOT),
                address: None,
                bytes: le_words(indices),
            },
        ],
        push: push_bytes,
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [positions, 1, 1],
        step_limit: u64::MAX,
        ..Config::default()
    };
    let o = run(&m, &cfg, res).expect("run");
    words_of(&o.buffers[1])
}

/// Metal's patch index (a Metal probe on an M4): the position without a patch index buffer, and
/// with one the word at the position from the buffer's residual offset, 0 outside the bound range.
/// A residual below the range's start (a direct draw's from its patch start) wraps.
#[test]
fn the_patch_index_is_the_bound_word_at_the_position() {
    let indices = [7, 7, 1, 3, 0, 2];
    assert_eq!(run_patch_index(&indices, (8, 0), 5), [0, 1, 2, 3, 4]);
    assert_eq!(run_patch_index(&indices, (8, 1), 5), [1, 3, 0, 2, 0]);
    assert_eq!(run_patch_index(&indices, (0, 1), 6), indices);
    assert_eq!(
        run_patch_index(&indices, (8u32.wrapping_neg(), 1), 5),
        [0, 0, 7, 7, 1]
    );
}

/// The elements [`elements`] computes for patch indices `0..patches` of `n` control points under
/// `index_type`, with `indices` (bytes) bound as the control-point index buffer and `residual`
/// pushed: per patch, its `n` elements.
fn run_elements(
    index_type: IndexType,
    n: u32,
    indices: &[u8],
    residual: u32,
    patches: u32,
) -> Vec<Vec<u32>> {
    let mut c = control(PatchDomain::Triangle, Partition::Integer, 64);
    c.control_points = n;
    c.index_type = index_type;
    let m = kernel(move |mb| {
        let src = declare_sources(mb, &c)?;
        let out = out_buffer(mb)?;
        Ok(Box::new(move |fb: &mut FunctionBuilder<'_>| {
            let patch = global_id(fb, 0)?;
            let count = ku(fb, n)?;
            let first = fb.binary(BinOp::IMul, patch, count)?;
            let base = fb.module().global_value(out)?;
            let zero = ku(fb, 0)?;
            for (k, v) in elements(fb, &c, &src, patch)?.into_iter().enumerate() {
                let k = ku(fb, k as u32)?;
                let at = fb.binary(BinOp::IAdd, first, k)?;
                let ptr = fb.access_chain(base, &[zero, at])?;
                fb.store(ptr, v, MemAccess::default())?;
            }
            Ok(())
        }))
    });
    let mut push_bytes = vec![0u8; c.push_offset as usize + 20];
    push_bytes.extend(residual.to_le_bytes());
    let mut buffers = vec![
        Buffer {
            slot: Some(FACTOR_SLOT),
            address: None,
            bytes: vec![0; 16],
        },
        Buffer {
            slot: Some(out_slot()),
            address: None,
            bytes: vec![0; (patches * n * 4) as usize],
        },
        Buffer {
            slot: Some(PATCH_INDEX_SLOT),
            address: None,
            bytes: vec![0; 16],
        },
    ];
    if index_type != IndexType::None {
        buffers.push(Buffer {
            slot: Some(CONTROL_POINT_INDEX_SLOT),
            address: None,
            bytes: indices.to_vec(),
        });
    }
    let res = Resources {
        buffers,
        push: push_bytes,
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [patches, 1, 1],
        step_limit: u64::MAX,
        ..Config::default()
    };
    let o = run(&m, &cfg, res).expect("run");
    words_of(&o.buffers[1])
        .chunks(n as usize)
        .map(<[u32]>::to_vec)
        .collect()
}

/// Metal's control points (Metal probes on an M4): control point `k` of patch `p` is element `p ·
/// N + k` without a control-point index type, and with one the `uint` or `ushort` at that position
/// from the buffer's residual offset (a `ushort` in either half of a word), 0 outside the bound
/// range. A residual below the range's start (a direct draw's from its patch start) wraps.
#[test]
fn control_points_are_the_indices_at_the_patch_positions() {
    assert_eq!(
        run_elements(IndexType::None, 3, &[], 0, 3),
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    );
    let words: Vec<u32> = (0..8).rev().collect();
    let bytes = le_words(&words);
    assert_eq!(
        run_elements(IndexType::UInt32, 3, &bytes, 0, 3),
        [[7, 6, 5], [4, 3, 2], [1, 0, 0]]
    );
    assert_eq!(
        run_elements(IndexType::UInt32, 3, &bytes, 4, 2),
        [[6, 5, 4], [3, 2, 1]]
    );
    assert_eq!(
        run_elements(IndexType::UInt32, 3, &bytes, 12u32.wrapping_neg(), 3),
        [[0, 0, 0], [7, 6, 5], [4, 3, 2]]
    );
    let halves: Vec<u8> = (0..8u16)
        .rev()
        .map(|i| 0x100 * i + 0x10)
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(
        run_elements(IndexType::UInt16, 3, &halves, 0, 3),
        [
            [0x710, 0x610, 0x510],
            [0x410, 0x310, 0x210],
            [0x110, 0x10, 0]
        ]
    );
    assert_eq!(
        run_elements(IndexType::UInt16, 3, &halves, 2, 2),
        [[0x610, 0x510, 0x410], [0x310, 0x210, 0x110]]
    );
    assert_eq!(
        run_elements(IndexType::UInt16, 3, &halves, 6u32.wrapping_neg(), 2),
        [[0, 0, 0], [0x710, 0x610, 0x510]]
    );
}

/// Every class of `half` as an edge factor and as an inside factor follows Metal's stated rules:
/// the cull, the scale, the clamp, and `pow2`'s rounding for NaNs, infinities, zeros, negatives,
/// subnormals, and normals. A quad patch per value `x`: edge 0 and inside 0 are `x`, inside 1 is
/// `x` negated; `x` is every exponent with the mantissas at and around each class boundary, and
/// every 31st value besides. ([`every_finite_positive_half_decodes_exactly`] covers every value's
/// decoding.)
#[test]
fn every_factor_class_follows_metals_rules() {
    let mut values: Vec<u16> = (0..32u16)
        .flat_map(|e| [0, 1, 2, 0x155, 0x200, 0x2aa, 0x3fe, 0x3ff].map(move |m| (e << 10) | m))
        .chain((0..0x8000).step_by(31))
        .collect();
    values.sort_unstable();
    values.dedup();
    let patches: Vec<Vec<u16>> = values
        .iter()
        .map(|&x| vec![x, h(4.0), h(4.0), h(4.0), x, x | 0x8000])
        .collect();
    let factors = pack(PatchDomain::Quad, 0, &patches);
    for (partition, max, scale) in [
        (Partition::Pow2, 16, Some(0.5)),
        (Partition::Integer, 64, None),
        (Partition::FractionalOdd, 64, Some(3.0)),
        (Partition::FractionalEven, 8, Some(1024.0)),
    ] {
        let mut c = control(PatchDomain::Quad, partition, max);
        c.scale_enabled = scale.is_some();
        let scale = scale.unwrap_or(1.0);
        let got = run_levels(&c, &factors, (0, 0, scale), patches.len() as u32, 1);
        for (p, row) in patches.iter().zip(&got) {
            assert_eq!(row, &expect(&c, p, scale), "{partition:?}: {:#06x}", p[0]);
        }
    }
}

/// No float instruction of the control stage can see a NaN or an infinity (the module docs), so
/// Vulkan's default float controls, under which a device may assume neither, cannot change a
/// level: every `float` it computes is a constant, an integer converted (`ConvertUToF`), an
/// `Ldexp` or a `Ceil` of one, a product, a select, or the pushed scale (a positive normal
/// `half`, which the binder checks). None is a bitcast of a factor's bits or loaded from the
/// factor buffer, and no float comparison is unordered.
#[test]
fn no_float_instruction_of_the_control_stage_can_see_a_nan_or_an_infinity() {
    use crate::spirv::ir::{Inst, TypedModule};
    use std::collections::HashMap;
    let float = |m: &TypedModule, v: Value| {
        let ty = m.value(v).expect("value").ty;
        matches!(m.types().get(ty).expect("type"), Type::Float { .. })
    };
    for partition in [
        Partition::Pow2,
        Partition::Integer,
        Partition::FractionalOdd,
        Partition::FractionalEven,
    ] {
        for scale_enabled in [false, true] {
            let mut c = control(PatchDomain::Quad, partition, 16);
            c.scale_enabled = scale_enabled;
            c.step = Step::PerPatchAndPerInstance;
            let m = control_module(&c).expect("control");
            let push = m
                .globals()
                .iter()
                .find(|g| g.class == StorageClass::PushConstant)
                .map(|g| g.value);
            let defs: HashMap<Value, &Inst> = m
                .blocks()
                .iter()
                .flat_map(|b| &b.insts)
                .filter_map(|n| n.result.map(|r| (r, &n.inst)))
                .collect();
            let mut floats = 0;
            for node in m.blocks().iter().flat_map(|b| &b.insts) {
                let produces = node.result.is_some_and(|r| float(&m, r));
                floats += usize::from(produces);
                let fine = match &node.inst {
                    Inst::Binary { op, a, .. } if float(&m, *a) => matches!(
                        op,
                        BinOp::FMul | BinOp::FOrdGreaterThanEqual | BinOp::FOrdLessThanEqual
                    ),
                    Inst::Binary { .. } => !produces,
                    Inst::Convert { op, a, .. } if float(&m, *a) => *op == ConvOp::ConvertFToU,
                    Inst::Convert { op, .. } => !produces || *op == ConvOp::ConvertUToF,
                    Inst::Glsl { op, args, .. }
                        if produces || args.iter().any(|&a| float(&m, a)) =>
                    {
                        matches!(op, GlslOp::Ldexp | GlslOp::Ceil)
                    }
                    Inst::Load { ptr, .. } if produces => matches!(
                        defs.get(ptr),
                        Some(Inst::AccessChain { base, .. }) if Some(*base) == push
                    ),
                    Inst::Unary { a, .. } => !float(&m, *a) && !produces,
                    Inst::Select { .. } | Inst::Store { .. } => true,
                    _ => !produces,
                };
                assert!(fine, "{partition:?} scale {scale_enabled}: {:?}", node.inst);
            }
            assert!(floats > 0);
        }
    }
}

/// The control stage declares Metal's partition, domain, and output winding (the winding
/// reversed under the upper-left domain origin; an A-track reading) with one output vertex.
#[test]
fn the_control_stage_declares_the_descriptor_modes() {
    let mut c = control(PatchDomain::Triangle, Partition::FractionalOdd, 16);
    assert_eq!(
        modes(&c),
        [
            ExecMode::Triangles,
            ExecMode::SpacingFractionalOdd,
            ExecMode::VertexOrderCcw,
            ExecMode::OutputVertices { n: 1 }
        ]
    );
    c.domain = PatchDomain::Quad;
    c.partition = Partition::Pow2;
    c.winding = Winding::CounterClockwise;
    assert_eq!(
        modes(&c),
        [
            ExecMode::Quads,
            ExecMode::SpacingEqual,
            ExecMode::VertexOrderCw,
            ExecMode::OutputVertices { n: 1 }
        ]
    );
}

/// The synthesized modules pass L3 for every domain, partition, winding, and step function, with
/// and without control points and under each control-point index type, on the committed profiles
/// with `tessellationShader`: the tessellation levels are `Patch` outputs
/// (`VUID-StandaloneSpirv-TessLevelInner-10880`).
#[test]
fn the_synthesized_stages_pass_l3() {
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    for (stem, p) in profiles
        .iter()
        .filter(|(_, p)| p.fields().fragment.tessellation_shader)
    {
        crate::spirv::testing::check_valid(vertex_module().expect("vertex"), p)
            .unwrap_or_else(|e| panic!("{stem} vertex: {e}"));
        for domain in [PatchDomain::Triangle, PatchDomain::Quad] {
            for (partition, step, winding) in [
                (Partition::Pow2, Step::Constant, Winding::Clockwise),
                (
                    Partition::Integer,
                    Step::PerPatch,
                    Winding::CounterClockwise,
                ),
                (
                    Partition::FractionalOdd,
                    Step::PerInstance,
                    Winding::Clockwise,
                ),
                (
                    Partition::FractionalEven,
                    Step::PerPatchAndPerInstance,
                    Winding::CounterClockwise,
                ),
            ] {
                for (control_points, index_type) in [
                    (0, IndexType::None),
                    (3, IndexType::UInt16),
                    (16, IndexType::None),
                    (32, IndexType::UInt32),
                ] {
                    let c = Control {
                        domain,
                        partition,
                        winding,
                        max_factor: 16,
                        step,
                        scale_enabled: step == Step::PerInstance,
                        push_offset: 64,
                        control_points,
                        index_type,
                        addressed: false,
                    };
                    crate::spirv::testing::check_valid(control_module(&c).expect("control"), p)
                        .unwrap_or_else(|e| panic!("{stem} {c:?}: {e}"));
                    // The address form, on a profile with device addresses and 64-bit integers,
                    // without a control-point index type, which it does not implement.
                    let enabled = &p.fields().enabled.features;
                    if index_type == IndexType::None
                        && enabled.contains(&VkFeature::BufferDeviceAddress)
                        && enabled.contains(&VkFeature::ShaderInt64)
                    {
                        let c = Control {
                            addressed: true,
                            ..c
                        };
                        crate::spirv::testing::check_valid(control_module(&c).expect("control"), p)
                            .unwrap_or_else(|e| panic!("{stem} {c:?}: {e}"));
                    }
                }
            }
        }
    }
}
