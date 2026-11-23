//! A two-plane extended-range 3D texture, read, written and queried as Metal answers
//! (`lower::texture::emit::planes_volume`, measured on an Apple M3 Max under macOS 27.0.1).
//!
//! - On the CPU interpreter, every thread of a 37 x 5 x 3 grid reads texel `(0, 0, 0)` of the
//!   texture through a sampled and a storage slot, whatever its coordinate, and its write at
//!   `(x, y, z)` stores texel `(x, y)` of the target's first slice, its other slices untouched.
//! - Its width query answers its width, its depth query `max(1, ⌈w / 4⌉ >> l)` at each level `l`,
//!   and its level count `1 - ⌊(w - 1) / 16⌋` of its one level.

use super::*;

/// The texture's extent: a width whose depth and level answers differ from a 3D texture's.
const EXTENT: [u32; 3] = [37, 5, 3];

/// A kernel reading `texture3d<float, access>` slot 0 at its thread's `gid`, storing the texel to
/// `out` (buffer 0), writing `(x / 64, y / 64, 0.5, 1)` to `texture3d<float, write>` slot 1 at
/// `gid`, and, where `queries`, storing slot 0's width, its depth at level `x & 3`, its level
/// count and its height to `q` (buffer 1).
fn volume_kernel(source: Source, queries: bool) -> String {
    let (access, code) = match source {
        Source::Read => ("read", 1),
        Source::ReadWrite => ("read_write", 3),
    };
    let [w, h, _] = EXTENT;
    let query_body = match queries {
        true => {
            r#"  %l = and i32 %x, 3
  %qw = call i32 @air.get_width_texture_3d(ptr addrspace(1) %t, i32 0)
  %qd = call i32 @air.get_depth_texture_3d(ptr addrspace(1) %t, i32 %l)
  %qn = call i32 @air.get_num_mip_levels_texture_3d(ptr addrspace(1) %t)
  %qh = call i32 @air.get_height_texture_3d(ptr addrspace(1) %t, i32 0)
  %q0 = insertelement <4 x i32> zeroinitializer, i32 %qw, i64 0
  %q1 = insertelement <4 x i32> %q0, i32 %qd, i64 1
  %q2 = insertelement <4 x i32> %q1, i32 %qn, i64 2
  %q3 = insertelement <4 x i32> %q2, i32 %qh, i64 3
  %qp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %q, i64 %i
  store <4 x i32> %q3, ptr addrspace(1) %qp, align 16
"#
        }
        false => "",
    };
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %w, ptr addrspace(1) %out, ptr addrspace(1) %q, <3 x i32> %gid) {{
  %x = extractelement <3 x i32> %gid, i64 0
  %y = extractelement <3 x i32> %gid, i64 1
  %z = extractelement <3 x i32> %gid, i64 2
  %zh = mul i32 %z, {h}
  %zy = add i32 %zh, %y
  %row = mul i32 %zy, {w}
  %n = add i32 %row, %x
  %i = zext i32 %n to i64
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call {{ <4 x float>, i8 }} @air.read_texture_3d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <3 x i32> %gid, <3 x i32> zeroinitializer, i32 0, i32 {code})
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  %fx = uitofp i32 %x to float
  %fy = uitofp i32 %y to float
  %vx = fmul float %fx, 1.562500e-02
  %vy = fmul float %fy, 1.562500e-02
  %w0 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 5.000000e-01, float 1.000000e+00>, float %vx, i64 0
  %w1 = insertelement <4 x float> %w0, float %vy, i64 1
  call void @air.write_texture_3d.v4f32(ptr addrspace(1) %w, <3 x i32> %gid, <4 x float> %w1, i32 0, i32 2)
{query_body}  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} @air.read_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x i32>, <3 x i32>, i32, i32)
declare void @air.write_texture_3d.v4f32(ptr addrspace(1), <3 x i32>, <4 x float>, i32, i32)
declare i32 @air.get_width_texture_3d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_3d(ptr addrspace(1), i32)
declare i32 @air.get_depth_texture_3d(ptr addrspace(1), i32)
declare i32 @air.get_num_mip_levels_texture_3d(ptr addrspace(1))

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6, !7}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"texture3d<float, {access}>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture3d<float, write>", !"air.arg_name", !"w"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"q"}}
!7 = !{{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}}
"#
    )
}

/// The 3D image of slot `index` of `EXTENT`, texel `n` (in `x`, then `y`, then `z` order) of it
/// `texels(n)`.
fn volume(index: u32, texels: impl Fn(u32) -> [u32; 4]) -> Image {
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    };
    let mut img = Image::new(slot, EXTENT, 1, 1, 1);
    let [w, h, d] = EXTENT;
    for n in 0..w * h * d {
        let at = img
            .index(0, 0, 0, [n % w, n / w % h, n / (w * h)])
            .expect("texel");
        img.texels[at] = texels(n);
    }
    img
}

/// `m` over the grid, one thread a texel, with `images` bound, in both scheduling orders, which
/// must agree: `out`, `q` and the images.
fn run_volume(m: &TypedModule, plan: &Plan, images: &[Image]) -> (Vec<u8>, Vec<u8>, Vec<Image>) {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([1, 1, 1]));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let [w, h, d] = EXTENT;
    let texels = (w * h * d) as usize;
    let buffer = |index| Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes: vec![0xab; 16 * texels],
    };
    let res = Resources {
        buffers: vec![buffer(0), buffer(1)],
        push,
        spec,
    };
    let mut runs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: EXTENT,
            order,
            ..Config::default()
        };
        let (o, imgs) = run_with_images(m, &cfg, res.clone(), images.to_vec())
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        let mut buffers = o.buffers.into_iter();
        let (out, q) = (buffers.next().expect("out"), buffers.next().expect("q"));
        let target = texels_of(&imgs, 1).to_vec();
        runs.push((out, q, target, imgs));
    }
    let second = runs.pop().expect("second");
    let first = runs.pop().expect("first");
    assert_eq!(
        (&first.0, &first.1, &first.2),
        (&second.0, &second.1, &second.2),
        "the scheduling orders disagree"
    );
    (first.0, first.1, first.3)
}

/// The code Metal writes for `k / 64` (`0 <= k < 64`): the exact product with 510 rounded to
/// nearest even, offset by 384.
fn code_of(k: u32) -> u32 {
    (f64::from(k) * 510.0 / 64.0).round_ties_even() as u32 + 384
}

/// Every thread's read takes texel `(0, 0, 0)`; each write lands at `(x, y)` of the first slice;
/// the queries answer Metal's depth and level count.
#[test]
fn a_two_plane_volume_reads_its_first_texel_and_writes_its_first_slice() {
    let form = ExtendedRange::Planes { srgb: false };
    for (source, queries) in [(Source::Read, true), (Source::ReadWrite, false)] {
        let what = format!("{source:?}");
        let Some(bc) = assemble(&what, &volume_kernel(source, queries)) else {
            return;
        };
        for (stem, p) in profiles() {
            let what = format!("{what} on {stem}");
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(form, source))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&bc, &inputs(form, source), &p, &what);
            let first = |n: u32| (n * 37 + 11) % 1024;
            let images = [
                volume(0, |n| raw_texel(form, first(n))),
                volume(1, |_| [0; 4]),
            ];
            let (out, q, images) = run_volume(&m, &plan, &images);
            let [w, h, d] = EXTENT;
            let want: Vec<u32> = codes_in(form, first(0))
                .into_iter()
                .enumerate()
                .map(|(c, code)| value(form, c, code).to_bits())
                .collect();
            for n in 0..w * h * d {
                let got: Vec<u32> = (0..4).map(|c| lane(&out, false, n, c)).collect();
                assert_eq!(got, want, "{what}: thread {n} read");
            }
            let target = texels_of(&images, 1);
            let layout = volume(1, |_| [0; 4]);
            for n in 0..w * h * d {
                let [x, y, z] = [n % w, n / w % h, n / (w * h)];
                let at = layout.index(0, 0, 0, [x, y, z]).expect("texel");
                let want = match z {
                    0 => [code_of(x) << 20 | code_of(y) << 10 | 639, 255],
                    _ => [0, 0],
                };
                assert_eq!(
                    target[at][..2],
                    want,
                    "{what}: texel ({x}, {y}, {z}) written"
                );
            }
            if !queries {
                continue;
            }
            for n in 0..w * h * d {
                let word = |c: usize| {
                    let i = n as usize * 16 + 4 * c;
                    u32::from_le_bytes([q[i], q[i + 1], q[i + 2], q[i + 3]])
                };
                let level = (n % w) & 3;
                let depth = (w.div_ceil(4) >> level).max(1);
                let levels = 1u32.wrapping_sub((w - 1) / 16);
                assert_eq!(
                    [word(0), word(1), word(2), word(3)],
                    [w, depth, levels, h],
                    "{what}: thread {n}'s queries"
                );
            }
        }
    }
}
