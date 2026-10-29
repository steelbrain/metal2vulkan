//! Mipmapped and array samples of an extended-range texture: the level Metal's rule reads, a
//! linear mip filter's trilinear sum, a 2D array's layer, and gathers at the base level.
//!
//! - On the CPU interpreter, Metal's samples and gathers of the XR mip probe (an Apple M3 Max on
//!   macOS 27.0.1, the fixture `xr_levels.tsv`) are Metal's bytes, through a runtime sampler and
//!   through a constexpr one of the same state: `BGRA10_XR`, `BGR10_XR` and their `_sRGB` twins,
//!   2D and 2D array textures of five levels, nearest and linear filters under nearest and linear
//!   mip filters, `float` and `half` results, at explicit levels; and under a level clamp.
//! - A level from gradients is Vulkan's level of detail by Metal's rule, and records
//!   `DeviceSampleLevel`; a runtime sampler's records it where its word states a mip filter. Its
//!   texels are not executed (the interpreter models no `Log2`).
//! - What is not built is refused: a linear mip filter of a form whose levels' blend is not
//!   measured, by the lowering for a constexpr sampler and by the binder for a runtime one, and a
//!   runtime sampler with a linear mip filter whose clamp's maximum the function does not read.

use super::samples::{form_of, raw_of, sampled_state, Through};
use super::*;
use crate::bind::SamplerDescriptor;
use crate::error::Unsupported;
use crate::metal::sampler::SamplerWord;
use crate::metal::{SamplerMinMagFilter, SamplerMipFilter};
use crate::pipeline::artifact::PrecisionNote;

/// The probe's textures' side and levels.
const PROBE_SIDE: u32 = 16;
const PROBE_LEVELS: u32 = 5;

/// The shape of a probe's texture: 2D, a 2D array of three layers, or 3D.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    Plane,
    Array,
    Volume,
}

impl Shape {
    /// A row's shape: a 2D array's or a 2D texture's.
    fn of(array: bool) -> Shape {
        match array {
            true => Shape::Array,
            false => Shape::Plane,
        }
    }

    fn layers(self) -> u32 {
        match self {
            Shape::Array => 3,
            Shape::Plane | Shape::Volume => 1,
        }
    }

    /// The depth of a level `w` texels wide.
    fn depth(self, w: u32) -> u32 {
        match self {
            Shape::Volume => w,
            Shape::Plane | Shape::Array => 1,
        }
    }
}

/// One line of `xr_levels.tsv` (the XR mip probe's kernels' samples):
/// a sample (`S`) or a gather of each component (`G`); the raw format; `2d` or `2da`; the result
/// (`f`, `h`); the min and mag filter and the mip filter of the runtime sampler (clamped to the
/// edge, normalized coordinates); its level clamp (`-` the default, else `min,max`); the point's
/// `u`, `v` and level as `f32` bits and its layer; and the result's lanes as bits (a gather's four
/// texels, components x, y, z, w).
struct Row {
    gather: bool,
    form: ExtendedRange,
    array: bool,
    half: bool,
    linear: bool,
    mip: SamplerMipFilter,
    clamp: Option<(f32, f32)>,
    point: [f32; 4],
    bits: Vec<u32>,
}

fn rows() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/xr_levels.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let f = |s: &str| f32::from_bits(u32::from_str_radix(s, 16).expect("f32 bits"));
    text.lines()
        .map(|l| {
            let r: Vec<&str> = l.split(' ').collect();
            let clamp = match r[6] {
                "-" => None,
                c => {
                    let (a, b) = c.split_once(',').expect("clamp");
                    Some((a.parse().expect("min"), b.parse().expect("max")))
                }
            };
            Row {
                gather: r[0] == "G",
                form: form_of(r[1]),
                array: r[2] == "2da",
                half: r[3] == "h",
                linear: r[4] == "linear",
                mip: match r[5] {
                    "mipnearest" => SamplerMipFilter::Nearest,
                    _ => SamplerMipFilter::Linear,
                },
                clamp,
                point: [
                    f(r[7]),
                    f(r[8]),
                    f(r[9]),
                    r[10].parse::<u32>().expect("layer") as f32,
                ],
                bits: r[11]
                    .split(',')
                    .map(|x| u32::from_str_radix(x, 16).expect("bits"))
                    .collect(),
            }
        })
        .collect()
}

/// The probe's texture of `form` in `shape`: its generator's codes (an LCG seeded with `seed`: 7
/// for the samples, 9 for the gathers, 11 and 13 under the clamp, 17 for the 3D samples under
/// mirrored repeat and repeat), per level per layer, row by row and a 3D level's slice by slice,
/// red, green, blue, alpha (`BGR10_XR`'s alpha 0).
pub(super) fn probe_texture(
    form: ExtendedRange,
    shape: Shape,
    seed: u64,
) -> Vec<Vec<Vec<[u32; 4]>>> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let four = matches!(form, ExtendedRange::Bgra10 { .. });
    (0..PROBE_LEVELS)
        .map(|l| {
            let w = (PROBE_SIDE >> l).max(1);
            (0..shape.layers())
                .map(|_| {
                    (0..w * w * shape.depth(w))
                        .map(|_| match four {
                            // The bytes of blue, green, red and alpha, in that order.
                            true => {
                                let c: [u32; 4] = std::array::from_fn(|_| rnd() % 1024);
                                [c[2], c[1], c[0], c[3]]
                            }
                            false => {
                                let b = rnd() % 1024;
                                let g = rnd() % 1024;
                                let r = rnd() % 1024;
                                let _ = rnd();
                                [r, g, b, 0]
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// A kernel sampling `texture2d<T, sample>` (or `texture2d_array`) slot 0 through `through` at
/// `float4` buffer 0's lane `tid` (`u`, `v`, the level, the layer), at that level, storing the
/// texel to buffer 1; or with `gather`, gathering each component there, four texels a lane.
pub(super) fn level_kernel(through: Through, half: bool, array: bool, gather: bool) -> String {
    let (texel, vec4, size, suffix, out_ty) = if half {
        ("half", "<4 x half>", 8, "v4f16", "half4")
    } else {
        ("float", "<4 x float>", 16, "v4f32", "float4")
    };
    let (state, states, state_md, s, params, sampler_md) = match through {
        Through::Constexpr(w) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n\n"
            ),
            "!air.sampler_states = !{!6}\n",
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
            "@__air_sampler_state",
            "",
            "",
        ),
        Through::Runtime => (
            String::new(),
            "",
            "",
            "%s",
            ", ptr addrspace(2) %s",
            "!7 = !{i32 3, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n",
        ),
    };
    let (tid_index, args_md) = match through {
        Through::Constexpr(_) => (3, "!3, !4, !5, !8"),
        Through::Runtime => (4, "!3, !4, !5, !7, !8"),
    };
    let (dim, layer_arg, layer_ty) = match array {
        true => ("2d_array", ", i32 %layer", ", i32"),
        false => ("2d", "", ""),
    };
    let ty_name = match array {
        true => format!("texture2d_array<{texel}, sample>"),
        false => format!("texture2d<{texel}, sample>"),
    };
    let body = match gather {
        false => format!(
            r#"  %r = call {{ {vec4}, i8 }} @air.sample_texture_{dim}.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv{layer_arg}, i1 true, <2 x i32> zeroinitializer, i1 true, float %lod, float 0.0, i32 0)
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
"#
        ),
        true => {
            let mut b = String::from("  %i4 = shl i64 %i, 2\n");
            for c in 0..4 {
                b.push_str(&format!(
                    r#"  %r{c} = call {{ {vec4}, i8 }} @air.gather_texture_{dim}.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv{layer_arg}, i1 true, <2 x i32> zeroinitializer, i32 {c}, i32 0)
  %v{c} = extractvalue {{ {vec4}, i8 }} %r{c}, 0
  %n{c} = add i64 %i4, {c}
  %q{c} = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %n{c}
  store {vec4} %v{c}, ptr addrspace(1) %q{c}, align {size}
"#
                ));
            }
            b
        }
    };
    let decl = match gather {
        false => format!(
            "declare {{ {vec4}, i8 }} @air.sample_texture_{dim}.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>{layer_ty}, i1, <2 x i32>, i1, float, float, i32)"
        ),
        true => format!(
            "declare {{ {vec4}, i8 }} @air.gather_texture_{dim}.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>{layer_ty}, i1, <2 x i32>, i32, i32)"
        ),
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{params}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %pt = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %pt, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %lod = extractelement <4 x float> %pt, i64 2
  %sl = extractelement <4 x float> %pt, i64 3
  %layer = fptoui float %sl to i32
{body}  ret void
}}

{decl}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args_md}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{ty_name}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
{state_md}{sampler_md}!8 = !{{i32 {tid_index}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// The runtime sampler of a row's state: its filters, mip filter and clamp, clamped to the edge.
pub(super) fn descriptor(
    linear: bool,
    mip: SamplerMipFilter,
    clamp: Option<(f32, f32)>,
) -> SamplerDescriptor {
    let mut d = SamplerDescriptor::new();
    let f = match linear {
        true => SamplerMinMagFilter::Linear,
        false => SamplerMinMagFilter::Nearest,
    };
    d.min_filter = f;
    d.mag_filter = f;
    d.mip_filter = mip;
    d.s_address_mode = crate::metal::SamplerAddressMode::ClampToEdge;
    d.t_address_mode = crate::metal::SamplerAddressMode::ClampToEdge;
    if let Some((min, max)) = clamp {
        d.lod_min_clamp = min;
        d.lod_max_clamp = max;
    }
    d
}

/// The AIR state words of the constexpr sampler of a row's state (`half` clamp bounds, the
/// default maximum 65504).
pub(super) fn constexpr_words(
    linear: bool,
    mip: SamplerMipFilter,
    clamp: Option<(f32, f32)>,
) -> u64 {
    let base = super::samples::words(linear, 1, false, 0);
    let mip = u64::from(mip.as_raw()) << 13;
    let (min, max) = clamp.unwrap_or((0.0, 65504.0));
    let half = |x: f32| u64::from(crate::float16::f32_to_f16_bits(x));
    let cleared = base & !(0xffff_ffff << 24);
    cleared | mip | half(min) << 24 | half(max) << 40
}

/// The texels the module `m` of `plan` reads at `points` of the probe's texture `levels` in
/// `form` on the CPU interpreter (one thread a point; four texels a gather's), through a runtime
/// sampler of `desc` where it has one: each texel's lanes as bits. Both scheduling orders agree.
pub(super) fn run_levels(
    plan: &Plan,
    m: &TypedModule,
    (form, levels, shape): (ExtendedRange, &[Vec<Vec<[u32; 4]>>], Shape),
    (points, half, gather): (&[[f32; 4]], bool, bool),
    desc: Option<SamplerDescriptor>,
    what: &str,
) -> Vec<[u32; 4]> {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    if let Some(desc) = desc {
        let s = plan
            .families
            .sampler
            .slots
            .get(&0)
            .expect("the runtime sampler");
        let fields = s.fields();
        let word = SamplerWord::of(&desc).0;
        spec.insert(
            SpecKey::SamplerState {
                slot: s.slot,
                fields,
            },
            u64::from(word & fields),
        );
        if s.min_level {
            let min = crate::metal::sampler::query_min_level(&desc);
            spec.insert(SpecKey::SamplerMinLevel(s.slot), u64::from(min));
        }
    }
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let texels = if gather { 4 } else { 1 };
    let size = if half { 8 } else { 16 } * texels;
    let buffer = |index: u32, bytes: Vec<u8>| Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    };
    let input = points
        .iter()
        .flat_map(|c| c.iter().flat_map(|x| x.to_le_bytes()))
        .collect();
    let res = Resources {
        buffers: vec![buffer(0, input), buffer(1, vec![0xab; size * points.len()])],
        push,
        spec,
    };
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index: 0,
    };
    let extent = [PROBE_SIDE, PROBE_SIDE, shape.depth(PROBE_SIDE)];
    let mut img = Image::new(slot, extent, shape.layers(), PROBE_LEVELS, 1);
    for (l, level) in (0u32..).zip(levels) {
        let w = (PROBE_SIDE >> l).max(1);
        for (layer, codes) in (0u32..).zip(level) {
            for (n, c) in (0u32..).zip(codes) {
                let xyz = [n % w, n / w % w, n / (w * w)];
                let at = img.index(l, layer, 0, xyz).expect("texel");
                img.texels[at] = raw_of(form, *c);
            }
        }
    }
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [points.len() as u32, 1, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), vec![img.clone()])
            .unwrap_or_else(|e| panic!("{what}: interpreter: {e:?}"));
        outs.push(o.buffers.into_iter().nth(1).expect("out"));
    }
    assert_eq!(outs[0], outs[1], "{what}: the scheduling orders disagree");
    (0..(points.len() * texels) as u32)
        .map(|n| std::array::from_fn(|c| lane(&outs[0], half, n, c)))
        .collect()
}

/// Every sample and gather of `xr_levels.tsv` is Metal's bytes, through the runtime sampler of
/// its state and through the constexpr sampler of that state: the level Metal's rule reads (the
/// sampler's clamp included), a linear mip filter's trilinear sum, a nearest one's level, the
/// array's layer clamped into it, and a gather's base level whatever the mip filter and clamp, its
/// `half` taps rounded as nearest samples.
#[test]
fn level_samples_are_metals_bytes() {
    let rows = rows();
    assert!(rows.len() > 600, "{} rows", rows.len());
    // The rows of each kernel and sampler state, in the fixture's order.
    let mut groups: Vec<(String, Vec<&Row>)> = Vec::new();
    for r in &rows {
        let key = format!(
            "{:?} {} {} {} {} {:?} {:?}",
            r.form, r.gather, r.array, r.half, r.linear, r.mip, r.clamp
        );
        match groups.last_mut() {
            Some((k, g)) if *k == key => g.push(r),
            _ => groups.push((key, vec![r])),
        }
    }
    let mut checked = 0;
    for (key, group) in &groups {
        let r0 = group[0];
        let seed = match (r0.gather, r0.clamp.is_some()) {
            (false, false) => 7,
            (true, false) => 9,
            (false, true) => 11,
            (true, true) => 13,
        };
        let levels = probe_texture(r0.form, Shape::of(r0.array), seed);
        let points: Vec<[f32; 4]> = group.iter().map(|r| r.point).collect();
        let desc = descriptor(r0.linear, r0.mip, r0.clamp);
        let words = constexpr_words(r0.linear, r0.mip, r0.clamp);
        // A runtime sampler's linear mip filter under a clamp whose maximum a nearest mip filter
        // reads below the last level is the binder's refusal
        // (`unmeasured_linear_mip_filters_are_refused`): its constexpr twin alone runs.
        let bound = r0.mip != SamplerMipFilter::Linear
            || SamplerWord::of(&desc).0 & SamplerWord::LEVEL_MAX == 0;
        let throughs = [
            (Through::Runtime, Some(desc)),
            (Through::Constexpr(words), None),
        ];
        for (through, runtime) in throughs.into_iter().filter(|(_, d)| bound || d.is_none()) {
            let kernel = level_kernel(through, r0.half, r0.array, r0.gather);
            let Some(bc) = assemble("levels", &kernel) else {
                return;
            };
            let mut inputs = LowerInputs::new();
            inputs.extended_range.insert(SOURCE, r0.form);
            for (stem, p) in profiles() {
                let what = format!("{key} {through:?} on {stem}");
                validate(&bc, &inputs, &p, &what);
                let (plan, m) = typed_module_with(&bc, "k", &p, &inputs)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let got = run_levels(
                    &plan,
                    &m,
                    (r0.form, &levels, Shape::of(r0.array)),
                    (&points, r0.half, r0.gather),
                    runtime,
                    &what,
                );
                let per = if r0.gather { 4 } else { 1 };
                for (k, r) in group.iter().enumerate() {
                    let lanes: Vec<u32> = got[k * per..(k + 1) * per]
                        .iter()
                        .flatten()
                        .copied()
                        .collect();
                    // A linear filter's sum, and a linear mip filter's blend by the clamped level's
                    // `f32` fraction where Metal takes sixty-fourths, are the device's arithmetic.
                    let blends = r.mip == SamplerMipFilter::Linear;
                    let weight = if blends { 1.0 / 64.0 } else { 0.0 };
                    match (r.linear || blends) && !r.gather {
                        true => assert!(
                            super::samples::filtered_close(&lanes, &r.bits, r.half, weight),
                            "{what}: point {:?}: {lanes:x?} against {:x?}",
                            r.point,
                            r.bits
                        ),
                        false => assert_eq!(lanes, r.bits, "{what}: point {:?}", r.point),
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0);
}

/// A kernel's sample with explicit gradients takes Vulkan's level of detail, computed in the
/// shader, by Metal's rule, and records `DeviceSampleLevel`: through a constexpr sampler always,
/// through a runtime one where its word states a mip filter. The module is valid on every profile;
/// the CPU interpreter models no `Log2`, so its texels are not executed here.
#[test]
fn gradient_levels_record_the_device_level() {
    let form = ExtendedRange::Bgra10 { srgb: false };
    let nearest = Through::Constexpr(constexpr_words(false, SamplerMipFilter::Nearest, None));
    for through in [nearest, Through::Runtime] {
        let kernel = grad_kernel(through);
        let Some(bc) = assemble("gradient levels", &kernel) else {
            return;
        };
        let mut inputs = LowerInputs::new();
        inputs.extended_range.insert(SOURCE, form);
        for (stem, p) in profiles() {
            let what = format!("{through:?} on {stem}");
            let a = validate_entry(&bc, "k", &inputs, &p, &what);
            let level = PrecisionNote::DeviceSampleLevel;
            let specialized = a.specialized.iter().any(|s| s.note == level);
            match through {
                Through::Constexpr(_) => assert!(
                    a.precision.contains(&level) && !specialized,
                    "{what}: {:?}",
                    a.precision
                ),
                Through::Runtime => assert!(specialized, "{what}: {:?}", a.specialized),
            }
        }
    }
}

/// A kernel sampling `texture2d<float, sample>` slot 0 through `through` at buffer 0's lane `tid`
/// (`u`, `v`, the `u` gradient in normalized units, unused) with explicit gradients `(g, 0)` and
/// `(0, 0)`.
fn grad_kernel(through: Through) -> String {
    level_kernel(through, false, false, false)
        .replace(
            "@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float %lod, float 0.0, i32 0)",
            "@air.sample_texture_2d_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, <2 x float> %gx, <2 x float> zeroinitializer, float 0.0, i1 true, <2 x i32> zeroinitializer, i32 0)",
        )
        .replace(
            "@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float %lod, float 0.0, i32 0)",
            "@air.sample_texture_2d_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, <2 x float> %gx, <2 x float> zeroinitializer, float 0.0, i1 true, <2 x i32> zeroinitializer, i32 0)",
        )
        .replace(
            "  %layer = fptoui float %sl to i32\n",
            "  %gx = insertelement <2 x float> zeroinitializer, float %lod, i64 0\n",
        )
        .replace(
            "declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
            "declare { <4 x float>, i8 } @air.sample_texture_2d_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, <2 x float>, <2 x float>, float, i1, <2 x i32>, i32)",
        )
}

/// A linear mip filter is built for the forms whose levels' blend is measured: the lowering
/// refuses a constexpr sampler's of another form (a two-plane or padded one),
/// `NotImplemented(Format)`, and builds its nearest mip filter; the binder refuses a runtime
/// sampler's of such a form, and one whose clamp's maximum a nearest mip filter would read below
/// the last level, and plans the rest into the variant.
#[test]
fn unmeasured_linear_mip_filters_are_refused() {
    let forms = [
        (PixelFormat::BGRA10XR, ExtendedRange::Bgra10 { srgb: false }),
        (
            PixelFormat::BGR10XRSRGB,
            ExtendedRange::Bgr10 { srgb: true },
        ),
        (
            PixelFormat::RGB10A82PXR10,
            ExtendedRange::Planes { srgb: false },
        ),
        (PixelFormat::R10UnormX6, ExtendedRange::R10X6),
    ];
    for mip in [SamplerMipFilter::Nearest, SamplerMipFilter::Linear] {
        let words = constexpr_words(true, mip, None);
        let Some(bc) = assemble(
            "constexpr mip",
            &level_kernel(Through::Constexpr(words), false, false, false),
        ) else {
            return;
        };
        for (stem, p) in profiles() {
            for (_, form) in forms {
                let mut i = LowerInputs::new();
                i.extended_range.insert(SOURCE, form);
                let built = mip == SamplerMipFilter::Nearest || form.blends_levels();
                match (typed_module_with(&bc, "k", &p, &i), built) {
                    (Ok(_), true) => {}
                    (Err(crate::error::Error::Unsupported(u)), false) => {
                        let want = Unsupported::not_implemented(Feature::Format);
                        assert_eq!((u.feature, u.needs), (want.feature, want.needs));
                    }
                    (other, _) => {
                        panic!("{mip:?} {form:?} on {stem}: {:?}", other.map(|_| ()))
                    }
                }
            }
        }
    }
    let Some(bc) = assemble(
        "runtime mip",
        &level_kernel(Through::Runtime, false, false, false),
    ) else {
        return;
    };
    for (stem, dev) in devices() {
        let plain = pipeline(&dev, &bc, &LowerInputs::new()).expect("a sampling kernel");
        for (pf, form) in forms {
            let short = descriptor(true, SamplerMipFilter::Linear, Some((0.0, 3.0)));
            for (name, desc, planned) in [
                (
                    "nearest",
                    descriptor(true, SamplerMipFilter::Nearest, None),
                    true,
                ),
                (
                    "clamped nearest",
                    descriptor(false, SamplerMipFilter::Nearest, Some((0.75, 2.5))),
                    true,
                ),
                (
                    "linear",
                    descriptor(true, SamplerMipFilter::Linear, None),
                    form.blends_levels(),
                ),
                ("short linear", short, false),
            ] {
                let what = format!("{name} {pf:?} on {stem}");
                match (plain.plan(&dev, &sampled_state(pf, desc).shape()), planned) {
                    (Ok(plan), true) => assert_eq!(
                        plan.variant.extended_range,
                        BTreeMap::from([(SOURCE, form)]),
                        "{what}"
                    ),
                    (Err(crate::error::BindError::Unsupported(u)), false) => {
                        assert_eq!(u, Unsupported::not_implemented(Feature::Format), "{what}")
                    }
                    (other, _) => panic!("{what}: {:?}", other.map(|_| ())),
                }
            }
        }
    }
}
