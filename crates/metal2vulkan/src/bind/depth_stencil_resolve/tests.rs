//! The depth/stencil resolve draws: what each needs, valid on every committed profile (L3,
//! `G-val`), numbered and refused as the module docs say, and the pick exact on the CPU
//! interpreter: run in a kernel over every depth pattern of two and four samples from five
//! values (ties of every shape, and `+0` beside `-0`), it takes the last sample of the smallest
//! depth and the first of the largest.

use super::*;
use crate::device::requirements::{VkExtension, VkFeature};
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::MemAccess;
use crate::spirv::testing::{assert_valid, committed_profiles};
use std::path::PathBuf;

/// Every draw at `samples`: each pick with each write.
fn every_draw(samples: u32) -> Vec<DepthStencilResolveDraw> {
    let mut writes = vec![ResolveWrite::Depth, ResolveWrite::StencilExport];
    writes.extend(StencilBit::ALL.map(ResolveWrite::StencilPlane));
    let mut out = Vec::new();
    for pick in [DepthPick::Min, DepthPick::Max] {
        for &write in &writes {
            out.push(DepthStencilResolveDraw::new(pick, samples, write));
        }
    }
    out
}

/// The requirements of `m` as serialized.
fn needs(m: TypedModule) -> Requirements {
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .requirements
}

/// The export needs the extension and nothing else does; no draw runs at sample rate.
#[test]
fn each_draw_needs_what_its_docs_say() {
    for draw in every_draw(4) {
        let r = needs(fragment_typed_module(draw).expect("fragment"));
        assert_eq!(
            r.extensions.contains(&VkExtension::ExtShaderStencilExport),
            draw.write == ResolveWrite::StencilExport,
            "{draw:?}: {r:?}"
        );
        assert!(
            !r.features.contains(&VkFeature::SampleRateShading),
            "{draw:?}: {r:?}"
        );
        assert_eq!(
            draw.write.reads_stencil(),
            draw.write != ResolveWrite::Depth
        );
    }
}

/// Every committed profile that has what a draw needs validates it, at every sample count.
#[test]
fn the_modules_are_valid_on_every_committed_profile() {
    for (name, profile) in committed_profiles().expect("profiles") {
        for samples in [2, 4, 8, 16, 32, 64] {
            for draw in every_draw(samples) {
                let m = fragment_typed_module(draw).expect("fragment");
                if profile.check(&needs(m.clone())).is_ok() {
                    let _ = assert_valid(m, &profile);
                } else {
                    assert_eq!(draw.write, ResolveWrite::StencilExport, "{name} {draw:?}");
                }
            }
        }
    }
}

/// On a committed device the modules come numbered with the fixed bindings, and a sample count
/// that is no resolve's is refused.
#[test]
fn the_modules_come_numbered_and_refuse_another_count() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(&path).expect("read");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device = Device::new(desc, DevicePolicy::default()).expect("device");
    for draw in every_draw(4) {
        let modules = depth_stencil_resolve_modules(&device, draw, Feature::Format);
        match modules {
            Ok(m) => assert!(!m.vertex.is_empty() && !m.fragment.is_empty()),
            Err(Error::Unsupported(u)) => {
                assert_eq!(draw.write, ResolveWrite::StencilExport, "{draw:?}");
                assert_eq!(u.feature, Feature::Format);
            }
            Err(other) => panic!("{draw:?}: {other}"),
        }
    }
    for samples in [0, 1, 3, 128] {
        let draw = DepthStencilResolveDraw::new(DepthPick::Min, samples, ResolveWrite::Depth);
        assert!(
            matches!(
                depth_stencil_resolve_modules(&device, draw, Feature::Format),
                Err(Error::Internal(_))
            ),
            "{samples}"
        );
        assert!(fragment_typed_module(draw).is_err(), "{samples}");
    }
}

// ---- The pick on the interpreter ----

/// The kernel's buffers: each invocation's depths' bits (one word a sample), and its two output
/// words, the picked depth's bits and its sample.
const DEPTHS_IN: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
const PICKED_OUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// A kernel running the fragment stage's pick ([`pick_sample`]) over `samples` depths an
/// invocation.
fn pick_module(pick: DepthPick, samples: u32) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = super::super::depth_stencil_copy::u32_ty(&mut b).expect("u32");
    let f32_ = b.ty(Type::Float { width: 32 }).expect("f32");
    let words = super::super::visibility_combine::words_buffer;
    let input = words(&mut b, DEPTHS_IN, u32_).expect("in");
    let output = words(&mut b, PICKED_OUT, u32_).expect("out");
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 }).expect("uvec3");
    let invocation = b
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)
        .expect("id");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let build = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let zero = uint(f, 0)?;
        let invocation = f.module().global_value(invocation)?;
        let invocation = f.load(invocation, MemAccess::default())?;
        let i = f.composite_extract(invocation, &[0])?;
        let input = f.module().global_value(input)?;
        let output = f.module().global_value(output)?;
        let per = uint(f, samples)?;
        let base = f.binary(BinOp::IMul, i, per)?;
        let mut depths = Vec::new();
        for s in 0..samples {
            let k = uint(f, s)?;
            let at = f.binary(BinOp::IAdd, base, k)?;
            let p = f.access_chain(input, &[zero, at])?;
            let bits = f.load(p, MemAccess::default())?;
            depths.push(f.convert(ConvOp::Bitcast, bits, f32_)?);
        }
        let (depth, index) = pick_sample(f, pick, &depths)?;
        let two = uint(f, 2)?;
        let out = f.binary(BinOp::IMul, i, two)?;
        let bits = f.convert(ConvOp::Bitcast, depth, u32_)?;
        let p = f.access_chain(output, &[zero, out])?;
        f.store(p, bits, MemAccess::default())?;
        let one = uint(f, 1)?;
        let out = f.binary(BinOp::IAdd, out, one)?;
        let p = f.access_chain(output, &[zero, out])?;
        f.store(p, index, MemAccess::default())?;
        Ok(())
    };
    build(&mut f).expect("body");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// Metal's pick of `depths` (the module docs), written independently of the modules: the last
/// sample whose depth no other sample's is below (`Min`), or the first whose depth no other
/// sample's is above (`Max`).
fn metal(pick: DepthPick, depths: &[f32]) -> (u32, u32) {
    let mut at = 0;
    for (s, &d) in depths.iter().enumerate() {
        let better = match pick {
            DepthPick::Min => d <= depths[at],
            DepthPick::Max => d > depths[at],
        };
        if better {
            at = s;
        }
    }
    (depths[at].to_bits(), at as u32)
}

/// Every pattern of `samples` depths from five values, `-0` and `+0` among them.
fn patterns(samples: u32) -> Vec<Vec<f32>> {
    let values = [-0.0f32, 0.0, 0.25, 0.5, 1.0];
    let mut out: Vec<Vec<f32>> = vec![Vec::new()];
    for _ in 0..samples {
        out = out
            .into_iter()
            .flat_map(|p| {
                values.iter().map(move |&v| {
                    let mut q = p.clone();
                    q.push(v);
                    q
                })
            })
            .collect();
    }
    out
}

#[test]
fn the_pick_takes_metals_sample_of_the_extreme_depth() {
    for samples in [2, 4] {
        let all = patterns(samples);
        for pick in [DepthPick::Min, DepthPick::Max] {
            let m = pick_module(pick, samples);
            let input: Vec<u8> = all
                .iter()
                .flatten()
                .flat_map(|d| d.to_bits().to_le_bytes())
                .collect();
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: Some(DEPTHS_IN),
                        address: None,
                        bytes: input,
                    },
                    Buffer {
                        slot: Some(PICKED_OUT),
                        address: None,
                        bytes: vec![0; all.len() * 8],
                    },
                ],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order: Order::Forward,
                workgroups: [all.len() as u32, 1, 1],
                ..Config::default()
            };
            let out = run(&m, &cfg, res).expect("run").buffers.remove(1);
            for (i, depths) in all.iter().enumerate() {
                let word = |k: usize| {
                    let at = (i * 2 + k) * 4;
                    u32::from_le_bytes(out[at..at + 4].try_into().expect("word"))
                };
                assert_eq!(
                    (word(0), word(1)),
                    metal(pick, depths),
                    "{pick:?} of {depths:?}"
                );
            }
        }
    }
}
