//! The families: declarations as Apple's compiler emits them, immediates, handle seeds, table
//! totality, and the known families each decoded or listed.

use super::layout::Layout;
use super::simd::OPS as GROUP_OPS;
use super::tests::{air, classify_self, fn_ty, meter, sym, Consts};
use super::texture::OPS as TEX_OPS;
use super::*;
use crate::ir::{AddrSpace, Type};

/// One type token: `void`, `i1`..`i64`, `f16`, `bf16`, `f32`, `v<N><scalar>`, `p<space>`, or
/// `{<token>}` for the literal struct `{ T, i8 }`.
fn ty(t: &mut TypeTable, token: &str) -> TypeId {
    let m = &mut meter();
    if let Some(inner) = token.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        let texel = ty(t, inner);
        let status = t.intern(Type::Int(8), m).unwrap();
        return t
            .intern(
                Type::Struct {
                    fields: Box::new([texel, status]),
                    packed: false,
                    name: None,
                },
                m,
            )
            .unwrap();
    }
    if let Some(space) = token.strip_prefix('p') {
        let space = AddrSpace::from_raw(space.parse().unwrap()).unwrap();
        return t.intern(Type::Ptr { space }, m).unwrap();
    }
    if token == "void" {
        return t.intern(Type::Void, m).unwrap();
    }
    if let Some(rest) = token.strip_prefix('v') {
        let n = rest.bytes().take_while(u8::is_ascii_digit).count();
        let elem = ty(t, &rest[n..]);
        let len = rest[..n].parse().unwrap();
        return t.intern(Type::Vector { elem, len }, m).unwrap();
    }
    let scalar = match token {
        "i1" => Type::Int(1),
        "i8" => Type::Int(8),
        "i16" => Type::Int(16),
        "i32" => Type::Int(32),
        "i64" => Type::Int(64),
        "f16" => Type::Half,
        "bf16" => Type::BFloat,
        "f32" => Type::Float,
        other => panic!("type token {other}"),
    };
    t.intern(scalar, m).unwrap()
}

/// Classify `name` declared as `decl`: `"<ret> (<param>, ...)"` in [`ty`] tokens.
fn declared(name: &str, decl: &str) -> DeclClass {
    let mut t = TypeTable::new();
    let (ret, params) = decl.split_once(" (").unwrap();
    let params = params.strip_suffix(')').unwrap();
    let ret = ty(&mut t, ret);
    let params: Vec<TypeId> = params
        .split(", ")
        .filter(|p| !p.is_empty())
        .map(|p| ty(&mut t, p))
        .collect();
    let f = t
        .intern(
            Type::Function {
                ret,
                params: params.into(),
                vararg: false,
            },
            &mut meter(),
        )
        .unwrap();
    classify(name.as_bytes(), sym(), f, &t).unwrap()
}

fn declared_air(name: &str, decl: &str) -> AirIntrinsic {
    match declared(name, decl) {
        DeclClass::Air(op) => op,
        other => panic!("{name} {decl}: {other:?}"),
    }
}

/// Declarations exactly as they appear in the public fixtures and the local corpus census, one or
/// more per family and operand form. Each must decode.
const OBSERVED: &[(&str, &str)] = &[
    // Sampling.
    (
        "air.sample_texture_2d.v4f32",
        "{v4f32} (p1, p2, v2f32, i1, v2i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_texture_2d.s.v4i16",
        "{v4i16} (p1, p2, v2f32, i1, v2i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_texture_1d.v4f32",
        "{v4f32} (p1, p2, f32, i1, i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_texture_1d_array.v4f32",
        "{v4f32} (p1, p2, f32, i32, i1, i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_texture_2d_array.v4f16",
        "{v4f16} (p1, p2, v2f32, i32, i1, v2i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_texture_3d.u.v4i32",
        "{v4i32} (p1, p2, v3f32, i1, v3i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_texture_cube.v4f32",
        "{v4f32} (p1, p2, v3f32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_texture_cube_array.v4f16",
        "{v4f16} (p1, p2, v3f32, i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_texture_2d_grad.v4f16",
        "{v4f16} (p1, p2, v2f32, v2f32, v2f32, f32, i1, v2i32, i32)",
    ),
    (
        "air.sample_depth_2d.f32",
        "{f32} (p1, p2, i32, v2f32, i1, v2i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_depth_2d_array.f32",
        "{f32} (p1, p2, i32, v2f32, i32, i1, v2i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_depth_cube.f32",
        "{f32} (p1, p2, i32, v3f32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_compare_depth_2d.f32",
        "{f32} (p1, p2, i32, v2f32, f32, i1, v2i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_compare_depth_2d_array.f32",
        "{f32} (p1, p2, i32, v2f32, i32, f32, i1, v2i32, i1, f32, f32, i32)",
    ),
    (
        "air.gather_texture_2d.v4f32",
        "{v4f32} (p1, p2, v2f32, i1, v2i32, i32, i32)",
    ),
    (
        "air.gather_texture_2d_array.s.v4i32",
        "{v4i32} (p1, p2, v2f32, i32, i1, v2i32, i32, i32)",
    ),
    (
        "air.gather_depth_2d.v4f32",
        "{v4f32} (p1, p2, i32, v2f32, i1, v2i32, i32)",
    ),
    (
        "air.gather_depth_2d_array.v4f32",
        "{v4f32} (p1, p2, i32, v2f32, i32, i1, v2i32, i32)",
    ),
    (
        "air.calculate_clamped_lod_texture_2d",
        "f32 (p1, p2, v2f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_texture_2d",
        "f32 (p1, p2, v2f32, i32)",
    ),
    // Sampling forms decoded (Apple's declarations).
    (
        "air.sample_compare_depth_2d_array_grad.f32",
        "{f32} (p1, p2, i32, v2f32, i32, f32, v2f32, v2f32, f32, i1, v2i32, i32)",
    ),
    (
        "air.sample_compare_depth_2d_grad.f32",
        "{f32} (p1, p2, i32, v2f32, f32, v2f32, v2f32, f32, i1, v2i32, i32)",
    ),
    (
        "air.sample_compare_depth_cube.f32",
        "{f32} (p1, p2, i32, v3f32, f32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_compare_depth_cube_array.f32",
        "{f32} (p1, p2, i32, v3f32, i32, f32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_compare_depth_cube_array_grad.f32",
        "{f32} (p1, p2, i32, v3f32, i32, f32, v3f32, v3f32, f32, i32)",
    ),
    (
        "air.sample_compare_depth_cube_grad.f32",
        "{f32} (p1, p2, i32, v3f32, f32, v3f32, v3f32, f32, i32)",
    ),
    (
        "air.sample_depth_2d_array_grad.f32",
        "{f32} (p1, p2, i32, v2f32, i32, v2f32, v2f32, f32, i1, v2i32, i32)",
    ),
    (
        "air.sample_depth_2d_grad.f32",
        "{f32} (p1, p2, i32, v2f32, v2f32, v2f32, f32, i1, v2i32, i32)",
    ),
    (
        "air.sample_depth_cube_array.f32",
        "{f32} (p1, p2, i32, v3f32, i32, i1, f32, f32, i32)",
    ),
    (
        "air.sample_depth_cube_array_grad.f32",
        "{f32} (p1, p2, i32, v3f32, i32, v3f32, v3f32, f32, i32)",
    ),
    (
        "air.sample_depth_cube_grad.f32",
        "{f32} (p1, p2, i32, v3f32, v3f32, v3f32, f32, i32)",
    ),
    (
        "air.sample_texture_3d_grad.v4f32",
        "{v4f32} (p1, p2, v3f32, v3f32, v3f32, f32, i1, v3i32, i32)",
    ),
    (
        "air.sample_texture_cube_array_grad.v4f32",
        "{v4f32} (p1, p2, v3f32, i32, v3f32, v3f32, f32, i32)",
    ),
    (
        "air.sample_texture_cube_grad.v4f32",
        "{v4f32} (p1, p2, v3f32, v3f32, v3f32, f32, i32)",
    ),
    (
        "air.gather_compare_depth_2d.f32",
        "{v4f32} (p1, p2, i32, v2f32, f32, i1, v2i32, i32)",
    ),
    (
        "air.gather_compare_depth_2d_array.f32",
        "{v4f32} (p1, p2, i32, v2f32, i32, f32, i1, v2i32, i32)",
    ),
    (
        "air.gather_compare_depth_cube.f32",
        "{v4f32} (p1, p2, i32, v3f32, f32, i32)",
    ),
    (
        "air.gather_compare_depth_cube_array.f32",
        "{v4f32} (p1, p2, i32, v3f32, i32, f32, i32)",
    ),
    (
        "air.gather_depth_cube.v4f32",
        "{v4f32} (p1, p2, i32, v3f32, i32)",
    ),
    (
        "air.gather_depth_cube_array.v4f32",
        "{v4f32} (p1, p2, i32, v3f32, i32, i32)",
    ),
    (
        "air.gather_texture_cube.v4f32",
        "{v4f32} (p1, p2, v3f32, i32, i32)",
    ),
    (
        "air.gather_texture_cube_array.v4f32",
        "{v4f32} (p1, p2, v3f32, i32, i32, i32)",
    ),
    (
        "air.calculate_clamped_lod_depth_2d",
        "f32 (p1, p2, v2f32, i32)",
    ),
    (
        "air.calculate_clamped_lod_depth_2d_array",
        "f32 (p1, p2, v2f32, i32)",
    ),
    (
        "air.calculate_clamped_lod_depth_cube",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_clamped_lod_depth_cube_array",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_clamped_lod_texture_2d_array",
        "f32 (p1, p2, v2f32, i32)",
    ),
    (
        "air.calculate_clamped_lod_texture_3d",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_clamped_lod_texture_cube",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_clamped_lod_texture_cube_array",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_depth_2d",
        "f32 (p1, p2, v2f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_depth_2d_array",
        "f32 (p1, p2, v2f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_depth_cube",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_depth_cube_array",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_texture_2d_array",
        "f32 (p1, p2, v2f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_texture_3d",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_texture_cube",
        "f32 (p1, p2, v3f32, i32)",
    ),
    (
        "air.calculate_unclamped_lod_texture_cube_array",
        "f32 (p1, p2, v3f32, i32)",
    ),
    // Reads: the current form, the legacy form, and 16-bit coordinates.
    (
        "air.read_texture_2d.v4f32",
        "{v4f32} (p1, p2, v2i32, v2i32, i32, i32)",
    ),
    ("air.read_texture_2d.v4f32", "{v4f32} (p1, v2i32, i32, i32)"),
    (
        "air.read_texture_2d.i16.u.v4i16",
        "{v4i16} (p1, p2, v2i16, v2i16, i16, i32)",
    ),
    (
        "air.read_texture_2d.i16.u.v4i16",
        "{v4i16} (p1, v2i16, i16, i32)",
    ),
    (
        "air.read_texture_1d.v4f32",
        "{v4f32} (p1, p2, i32, i32, i32)",
    ),
    ("air.read_texture_1d.v4f32", "{v4f32} (p1, i32, i32, i32)"),
    (
        "air.read_texture_2d_array.v4f16",
        "{v4f16} (p1, p2, v2i32, i32, v2i32, i32, i32)",
    ),
    (
        "air.read_texture_2d_array.v4f16",
        "{v4f16} (p1, v2i32, i32, i32, i32)",
    ),
    (
        "air.read_texture_2d_ms.v4f32",
        "{v4f32} (p1, p2, v2i32, i32, i32)",
    ),
    (
        "air.read_texture_2d_ms.v4f32",
        "{v4f32} (p1, v2i32, i32, i32)",
    ),
    (
        "air.read_texture_2d_ms_array.u.v4i16",
        "{v4i16} (p1, p2, v2i32, i32, i32, i32)",
    ),
    (
        "air.read_texture_3d.v4f32",
        "{v4f32} (p1, p2, v3i32, v3i32, i32, i32)",
    ),
    ("air.read_texture_3d.v4f32", "{v4f32} (p1, v3i32, i32, i32)"),
    (
        "air.read_texture_cube.v4f32",
        "{v4f32} (p1, p2, v2i32, i32, i32, i32)",
    ),
    (
        "air.read_texture_cube_array.v4f32",
        "{v4f32} (p1, p2, v2i32, i32, i32, i32, i32)",
    ),
    (
        "air.read_texture_cube.i16.v4f16",
        "{v4f16} (p1, v2i16, i16, i16, i32)",
    ),
    (
        "air.read_texture_buffer_1d.v4f16",
        "{v4f16} (p1, p2, i32, i32)",
    ),
    ("air.read_texture_buffer_1d.v4f16", "{v4f16} (p1, i32, i32)"),
    (
        "air.read_depth_2d.f32",
        "{f32} (p1, p2, i32, v2i32, v2i32, i32, i32)",
    ),
    ("air.read_depth_2d.f32", "{f32} (p1, i32, v2i32, i32, i32)"),
    (
        "air.read_depth_2d_array.i16.f32",
        "{f32} (p1, p2, i32, v2i16, i16, v2i16, i16, i32)",
    ),
    (
        "air.read_depth_2d_ms.f32",
        "{f32} (p1, p2, i32, v2i32, i32, i32)",
    ),
    // The texture reads Apple's metal 32023.921 declares: a 1D array, a depth cube and cube array
    // (a cube array's face before its array index), a depth multisampled array.
    (
        "air.read_texture_1d_array.v4f32",
        "{v4f32} (p1, p2, i32, i32, i32, i32)",
    ),
    (
        "air.read_texture_1d_array.i16.u.v4i32",
        "{v4i32} (p1, p2, i16, i16, i16, i32)",
    ),
    (
        "air.read_depth_cube.f32",
        "{f32} (p1, p2, i32, v2i32, i32, i32, i32)",
    ),
    (
        "air.read_depth_cube_array.i16.f32",
        "{f32} (p1, p2, i32, v2i16, i16, i16, i16, i32)",
    ),
    (
        "air.read_depth_2d_ms_array.f32",
        "{f32} (p1, p2, i32, v2i32, i32, i32, i32)",
    ),
    // Writes, atomics, fences.
    (
        "air.write_texture_1d.v4f32",
        "void (p1, i32, v4f32, i32, i32)",
    ),
    (
        "air.write_texture_1d_array.u.v4i32",
        "void (p1, i32, i32, v4i32, i32, i32)",
    ),
    (
        "air.write_texture_2d.v4f32",
        "void (p1, v2i32, v4f32, i32, i32)",
    ),
    (
        "air.write_texture_2d.i16.s.v4i16",
        "void (p1, v2i16, v4i16, i16, i32)",
    ),
    (
        "air.write_texture_2d_array.i16.u.v4i32",
        "void (p1, v2i16, i16, v4i32, i16, i32)",
    ),
    (
        "air.write_texture_3d.i16.v4f16",
        "void (p1, v3i16, v4f16, i16, i32)",
    ),
    (
        "air.write_texture_cube.v4f32",
        "void (p1, v2i32, i32, v4f32, i32, i32)",
    ),
    (
        "air.write_texture_cube_array.v4f32",
        "void (p1, v2i32, i32, i32, v4f32, i32, i32)",
    ),
    (
        "air.write_texture_buffer_1d.u.v4i32",
        "void (p1, i32, v4i32, i32)",
    ),
    (
        "air.atomic_fetch_max_explicit_texture_2d.i16.u.v4i32",
        "v4i32 (p1, v2i16, v2i16, v4i32, i32, i32)",
    ),
    ("air.fence_texture_2d", "void (p1)"),
    (
        "air.write_imageblock_slice_to_texture_2d.v4f32",
        "void (p1, p4, i1, v2i16, v2i16, v2i32, i32, i1, i32)",
    ),
    (
        "air.write_imageblock_slice_to_texture_2d.i16.v4i16",
        "void (p1, p4, i1, v2i16, v2i16, v2i16, i16, i1, i32)",
    ),
    (
        "air.write_imageblock_slice_to_texture_2d_array.i16.f16",
        "void (p1, p4, i1, v2i16, v2i16, v2i16, i16, i16, i1, i32)",
    ),
    // Queries and null handles.
    ("air.get_width_texture_2d", "i32 (p1, i32)"),
    ("air.get_height_depth_2d_array", "i32 (p1, i32)"),
    ("air.get_depth_texture_3d", "i32 (p1, i32)"),
    ("air.get_width_texture_2d_ms", "i32 (p1)"),
    ("air.get_width_texture_cube_array", "i32 (p1, i32)"),
    ("air.get_array_size_texture_2d_array", "i32 (p1)"),
    ("air.get_num_mip_levels_depth_cube", "i32 (p1)"),
    ("air.get_num_samples_texture_2d_ms", "i32 (p1)"),
    // The size queries Apple's metal 32023.921 declares: a multisampled texture and a texture
    // buffer take no level.
    ("air.get_width_texture_1d_array", "i32 (p1, i32)"),
    ("air.get_width_texture_2d_ms_array", "i32 (p1)"),
    ("air.get_height_texture_2d_ms_array", "i32 (p1)"),
    ("air.get_width_texture_buffer_1d", "i32 (p1)"),
    ("air.get_width_depth_2d_ms", "i32 (p1)"),
    ("air.get_height_depth_2d_ms_array", "i32 (p1)"),
    ("air.get_width_depth_cube", "i32 (p1, i32)"),
    ("air.get_height_depth_cube_array", "i32 (p1, i32)"),
    ("air.is_null_texture_1d", "i1 (p1)"),
    ("air.get_null_texture_3d", "p1 ()"),
    ("air.get_read_sampler", "p2 ()"),
    // Derivatives and boolean reductions.
    ("air.dfdx.v3f16", "v3f16 (v3f16)"),
    ("air.fast_dfdy.f32", "f32 (f32)"),
    ("air.fwidth.v4f32", "v4f32 (v4f32)"),
    ("air.all.v2i1", "i1 (v2i1)"),
    ("air.any.v4i1", "i1 (v4i1)"),
    // Barriers and fences.
    ("air.wg.barrier", "void (i32, i32)"),
    ("air.simdgroup.barrier", "void (i32, i32)"),
    ("air.atomic.fence", "void (i32, i32, i32)"),
    // Before AIR 2.0 (`xcrun metal` 32023.921 at deployment target macOS 10.12, `air64_v111`):
    // a texture operation states no trailing access qualifier, and a memory atomic no trailing
    // volatile flag. One `metal2vulkan-air` test reads these back operand by operand
    // (`pre_air_2_0_states_no_access_qualifier_or_volatile_flag`).
    ("air.read_texture_2d.v4f32", "v4f32 (p1, v2i32, i32)"),
    ("air.read_depth_2d.f32", "f32 (p1, i32, v2i32, i32)"),
    ("air.write_texture_2d.v4f32", "void (p1, v2i32, v4f32, i32)"),
    (
        "air.sample_texture_2d.v4f32",
        "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32)",
    ),
    (
        "air.gather_texture_2d.v4f32",
        "v4f32 (p1, p2, v2f32, i1, v2i32, i32)",
    ),
    ("air.atomic.global.add.u.i32", "i32 (p1, i32, i32, i32)"),
    ("air.atomic.local.add.u.i32", "i32 (p3, i32, i32, i32)"),
    ("air.atomic.local.load.i32", "i32 (p3, i32, i32)"),
    ("air.atomic.local.store.i32", "void (p3, i32, i32, i32)"),
    (
        "air.atomic.global.cmpxchg.weak.i32",
        "i32 (p1, p0, i32, i32, i32, i32)",
    ),
    // AIR 2.9 spells `frexp` as an intrinsic where every earlier emitter outlines it as the
    // runtime routine `___metal_frexp_<T>_pthread<I>`.
    ("air.frexp.v4f32", "v4f32 (v4f32, p0)"),
    // AIR 2.9 (`xcrun metal` 32023.921 at deployment target macOS 27): a barrier takes its order,
    // and a memory atomic its `mem_flags` before the volatile flag.
    ("air.wg.barrier", "void (i32, i32, i32)"),
    ("air.simdgroup.barrier", "void (i32, i32, i32)"),
    (
        "air.atomic.global.add.u.i32",
        "i32 (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.sub.s.i32",
        "i32 (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.and.u.i32",
        "i32 (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.or.u.i32",
        "i32 (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.xor.u.i32",
        "i32 (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.min.u.i32",
        "i32 (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.max.s.i32",
        "i32 (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.xchg.i32",
        "i32 (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.cmpxchg.weak.i32",
        "i32 (p1, p0, i32, i32, i32, i32, i32, i1)",
    ),
    ("air.atomic.global.load.i32", "i32 (p1, i32, i32, i32, i1)"),
    (
        "air.atomic.global.store.i32",
        "void (p1, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.add.f32",
        "f32 (p1, f32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.sub.f32",
        "f32 (p1, f32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.xchg.f32",
        "f32 (p1, f32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.max.u.i64",
        "void (p1, i64, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.global.min.u.i64",
        "void (p1, i64, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.local.add.u.i32",
        "i32 (p3, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.local.sub.u.i32",
        "i32 (p3, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.local.min.s.i32",
        "i32 (p3, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.local.max.s.i32",
        "i32 (p3, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.local.xchg.i32",
        "i32 (p3, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.local.cmpxchg.weak.i32",
        "i32 (p3, p0, i32, i32, i32, i32, i32, i1)",
    ),
    ("air.atomic.local.load.i32", "i32 (p3, i32, i32, i32, i1)"),
    (
        "air.atomic.local.store.i32",
        "void (p3, i32, i32, i32, i32, i1)",
    ),
    // MSL 4.1's atomics on thread memory (AIR 2.9 only).
    (
        "air.atomic.private.store.i32",
        "void (p0, i32, i32, i32, i32, i1)",
    ),
    ("air.atomic.private.load.i32", "i32 (p0, i32, i32, i32, i1)"),
    (
        "air.atomic.private.xchg.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.cmpxchg.weak.i32",
        "i32 (p0, p0, i32, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.add.u.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.add.s.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.sub.u.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.sub.s.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.and.u.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.or.u.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.xor.u.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.min.u.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.min.s.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.max.u.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.max.s.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.store.f32",
        "void (p0, f32, i32, i32, i32, i1)",
    ),
    ("air.atomic.private.load.f32", "f32 (p0, i32, i32, i32, i1)"),
    (
        "air.atomic.private.xchg.f32",
        "f32 (p0, f32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.add.f32",
        "f32 (p0, f32, i32, i32, i32, i1)",
    ),
    (
        "air.atomic.private.sub.f32",
        "f32 (p0, f32, i32, i32, i32, i1)",
    ),
    // Simdgroup and quad.
    ("air.simd_shuffle_down.u.v4i16", "v4i16 (v4i16, i16)"),
    (
        "air.simd_shuffle_and_fill_up.f32",
        "f32 (f32, f32, i16, i16)",
    ),
    ("air.simd_sum.v4f32", "v4f32 (v4f32)"),
    ("air.simd_prefix_exclusive_sum.s.i32", "i32 (i32)"),
    ("air.simd_or.u.i8", "i8 (i8)"),
    ("air.simd_broadcast_first.u.i32", "i32 (i32)"),
    // The products and rotate up (`xcrun metal` 32023.921).
    ("air.simd_product.f32", "f32 (f32)"),
    ("air.simd_product.u.i8", "i8 (i8)"),
    ("air.simd_prefix_inclusive_product.s.i32", "i32 (i32)"),
    ("air.simd_prefix_exclusive_product.f16", "f16 (f16)"),
    ("air.simd_shuffle_rotate_up.v4f32", "v4f32 (v4f32, i16)"),
    ("air.simd_ballot.i64", "i64 (i1)"),
    ("air.simd_all", "i1 (i1)"),
    ("air.simd_is_first", "i1 ()"),
    ("air.quad_broadcast.v2f16", "v2f16 (v2f16, i16)"),
    ("air.quad_shuffle_rotate_down.f32", "f32 (f32, i16)"),
    ("air.quad_max.s.i32", "i32 (i32)"),
    ("air.quad_any", "i1 (i1)"),
    ("air.quad_active_threads_mask", "i16 ()"),
    // The quad members `xcrun metal` 32023.921 emits beside them: the fills with an explicit modulo
    // or 4 where the source spells none.
    ("air.quad_broadcast_first.v4f32", "v4f32 (v4f32)"),
    ("air.quad_shuffle_rotate_up.v2f32", "v2f32 (v2f32, i16)"),
    (
        "air.quad_shuffle_and_fill_up.f32",
        "f32 (f32, f32, i16, i16)",
    ),
    (
        "air.quad_shuffle_and_fill_down.f32",
        "f32 (f32, f32, i16, i16)",
    ),
    ("air.quad_and.s.i16", "i16 (i16)"),
    ("air.quad_or.u.i32", "i32 (i32)"),
    ("air.quad_xor.u.i32", "i32 (i32)"),
    ("air.quad_product.s.i32", "i32 (i32)"),
    ("air.quad_prefix_exclusive_sum.f32", "f32 (f32)"),
    ("air.quad_prefix_inclusive_sum.u.i32", "i32 (i32)"),
    ("air.quad_prefix_exclusive_product.f32", "f32 (f32)"),
    ("air.quad_prefix_inclusive_product.f16", "f16 (f16)"),
    // Apple's spelling (`xcrun metal` 32023.883): a quad ballot and a simdgroup's active mask.
    ("air.quad_ballot", "i16 (i1)"),
    ("air.simd_active_threads_mask.i64", "i64 ()"),
    ("air.is_uniform.v2i16", "i1 (v2i16)"),
    ("air.is_uniform.i1", "i1 (i1)"),
    // `uniform<half>`, `uniform<float4>`, and `uniform<bfloat2>` (`xcrun metal` 32023.921).
    ("air.is_uniform.f16", "i1 (f16)"),
    ("air.is_uniform.v4f32", "i1 (v4f32)"),
    ("air.is_uniform.v2bf16", "i1 (v2bf16)"),
    ("air.get_simdgroup_size.i16", "i16 ()"),
    ("air.get_simdgroup_size.i32", "i32 ()"),
    // Global built-in variables (`xcrun metal` 32023.921, Metal 3.1): a vector built-in's lane,
    // and a scalar built-in.
    ("air.get_local_size.i32", "i32 (i32)"),
    ("air.get_global_id.i16", "i16 (i16)"),
    ("air.get_dispatch_threads_per_threadgroup.i32", "i32 (i32)"),
    ("air.get_local_linear_id.i32", "i32 ()"),
    ("air.get_quadgroups_per_threadgroup.i32", "i32 ()"),
    // Simdgroup matrices.
    (
        "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
        "v64f32 (p3, v2i64, v2i64, v2i64)",
    ),
    (
        "air.simdgroup_matrix_8x8_store.v64f16.p1f16",
        "void (v64f16, p1, v2i64, v2i64, v2i64)",
    ),
    // The AIR 2.7 block load and store (`-std=metal3.0 -target air64-apple-macos15.0`).
    (
        "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
        "v64f32 (p3, i64, v2i64, i1)",
    ),
    (
        "air.simdgroup_matrix_8x8_store.v64f16.p1f16",
        "void (v64f16, p1, i64, v2i64, i1)",
    ),
    (
        "air.simdgroup_matrix_8x8_init_diag.v64bf16.bf16",
        "v64bf16 (bf16)",
    ),
    (
        "air.simdgroup_matrix_8x8_init_filled.v64f16.f16",
        "v64f16 (f16)",
    ),
    (
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f32.v64f16.v64f32",
        "v64f16 (v64f32, v64f16, v64f32)",
    ),
    (
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f8e5m2.v64f8e5m2.v64f32",
        "v64f32 (v64i8, v64i8, v64f32)",
    ),
    (
        "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8bf16.v8f16.v8f32",
        "v8f32 (v8bf16, i1, v8f16, i1, v8f32)",
    ),
    (
        "air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.u.v8i32.v8i8.v8i8.v8i32",
        "v8i32 (v8i8, i1, v8i8, i1, v8i32)",
    ),
    // Function constants, fragment, imageblocks.
    ("air.is_function_constant_defined", "i1 (p2)"),
    ("air.normalize_function_constant_predicate.i8", "i8 (i8)"),
    ("air.discard_fragment", "void ()"),
    (
        "air.map_physical_to_screen_coordinates.v2f32.p2i8.i32",
        "v2f32 (v2f32, p2, i32)",
    ),
    (
        "air.map_screen_to_physical_coordinates.v2i32.p2i8.i32",
        "v2i32 (v2i32, p2, i32)",
    ),
    ("air.get_num_samples.i32", "i32 (i32)"),
    ("air.quad_is_helper_thread", "i1 ()"),
    ("air.simd_is_helper_thread", "i1 ()"),
    // Pull-model interpolation (`pull_model_interpolants`).
    ("air.interpolate_center_perspective.v4f32", "v4f32 (p1)"),
    ("air.interpolate_center_no_perspective.f32", "f32 (p1)"),
    ("air.interpolate_centroid_perspective.v2f32", "v2f32 (p1)"),
    ("air.interpolate_centroid_no_perspective.f16", "f16 (p1)"),
    (
        "air.interpolate_sample_perspective.v4f16",
        "v4f16 (p1, i32)",
    ),
    (
        "air.interpolate_sample_no_perspective.v4f32",
        "v4f32 (p1, i32)",
    ),
    ("air.interpolate_offset_perspective.f32", "f32 (p1, v2f32)"),
    (
        "air.interpolate_offset_no_perspective.v2f32",
        "v2f32 (p1, v2f32)",
    ),
    ("air.imageblock_data", "p4 (v2i16, i32, i16)"),
    ("air.get_imageblock_width", "i16 ()"),
    (
        "air.load.implicit_imageblock.v4f16",
        "v4f16 (i32, v2i16, i32, i16)",
    ),
    (
        "air.store.implicit_imageblock.f32",
        "void (f32, i32, v2i16, i32, i16)",
    ),
    // Async copies, function tables, acceleration structures, tensors, indirect commands.
    (
        "air.simdgroup_async_copy_2d.p3i8.p1i8",
        "p0 (i64, i64, p3, i64, i64, v2i64, p1, i64, i64, v2i64, v2i64, i32)",
    ),
    ("air.get_null_simdgroup_event", "p0 ()"),
    ("air.is_null_simdgroup_event", "i1 (p0)"),
    ("air.wait_simdgroup_events", "void (i32, p0)"),
    (
        "air.get_function_pointer_visible_function_table",
        "p0 (p1, i32)",
    ),
    ("air.get_size_visible_function_table", "i32 (p1)"),
    ("air.is_null_visible_function_table", "i1 (p1)"),
    ("air.get_null_intersection_function_table", "p1 ()"),
    (
        "air.set_buffer_intersection_function_table.p1i8",
        "void (p1, p1, i32)",
    ),
    (
        "air.get_instance_count_instance_acceleration_structure",
        "i32 (p1)",
    ),
    (
        "air.get_primitive_acceleration_structure_instance_acceleration_structure",
        "p1 (p1, i32)",
    ),
    (
        "air.get_data_pointer_instance_acceleration_structure",
        "p1 (p1)",
    ),
    ("air.get_descriptor_size_tensor", "i16 (i16, i16)"),
    ("air.get_extent_private_tensor.i32", "i32 (p0, i16, i16)"),
    (
        "air.init_strided_private_tensor.i32.local",
        "void (p0, i16, p3, p0, p0, i8)",
    ),
    (
        "air.slice_private_tensor_private_tensor.s.i32",
        "void (p0, p0, i16, p0, p0)",
    ),
    (
        "air.concurrent_dispatch_threads_compute_command",
        "void (p1, i32, v3i32, v3i32)",
    ),
    (
        "air.set_pipeline_state_compute_command",
        "void (p1, i32, p1)",
    ),
    ("air.set_barrier_compute_command", "void (p1, i32)"),
    ("air.clear_barrier_compute_command", "void (p1, i32)"),
    ("air.reset_compute_command", "void (p1, i32)"),
    ("air.copy_compute_command", "void (p1, i32, p1, i32)"),
    (
        "air.set_stage_in_region_compute_command",
        "void (p1, i32, v3i32, v3i32)",
    ),
    (
        "air.set_imageblock_size_compute_command",
        "void (p1, i32, v2i16)",
    ),
    (
        "air.set_threadgroup_memory_length_compute_command",
        "void (p1, i32, i32, i32)",
    ),
    (
        "air.set_kernel_buffer_compute_command.p2i8",
        "void (p1, i32, p2, i64, i32)",
    ),
    (
        "air.set_fragment_buffer_render_command.p1i8",
        "void (p1, i32, p1, i32)",
    ),
    // AIR 2.5's kernel and vertex buffer setters (Metal compiler 31001.723, the local corpus
    // census): no attribute stride.
    (
        "air.set_kernel_buffer_compute_command.p1i8",
        "void (p1, i32, p1, i32)",
    ),
    (
        "air.set_kernel_buffer_compute_command.p2i8",
        "void (p1, i32, p2, i32)",
    ),
    (
        "air.set_vertex_buffer_render_command.p1i8",
        "void (p1, i32, p1, i32)",
    ),
    (
        "air.set_vertex_buffer_render_command.p2i8",
        "void (p1, i32, p2, i32)",
    ),
    (
        "air.draw_primitives_render_command",
        "void (p1, i32, i32, i32, i32, i32, i32)",
    ),
    // A render command's barrier and its object and mesh setters (the fixture `mesh_commands`,
    // a Metal probe).
    ("air.set_barrier_render_command", "void (p1, i32)"),
    ("air.clear_barrier_render_command", "void (p1, i32)"),
    (
        "air.set_object_buffer_render_command.p1i8",
        "void (p1, i32, p1, i32)",
    ),
    (
        "air.set_mesh_buffer_render_command.p2i8",
        "void (p1, i32, p2, i32)",
    ),
    (
        "air.draw_mesh_threadgroups_render_command",
        "void (p1, i32, v3i32, v3i32, v3i32)",
    ),
    (
        "air.draw_mesh_threads_render_command",
        "void (p1, i32, v3i32, v3i32, v3i32)",
    ),
    (
        "air.set_object_threadgroup_memory_length_render_command",
        "void (p1, i32, i32, i32)",
    ),
    // 8-bit float conversions.
    ("air.convert.f.v8f16.f.v8f8e4m3fn", "v8f16 (v8i8)"),
    ("air.convert.f.f32.f.f8e5m2", "f32 (i8)"),
];

/// Every texture atomic Apple's metal 32023.921 declares: each operation on each texture its
/// atomics take, both texel signs, and the `.i16` coordinates of each texture, as `<name>
/// <declaration>` lines.
const ATOMICS_OBSERVED: &str = "\
air.atomic_compare_exchange_weak_explicit_texture_1d.s.v4i32 i1 (p1, i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_1d.u.v4i32 i1 (p1, i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_1d_array.s.v4i32 i1 (p1, i32, i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_1d_array.u.v4i32 i1 (p1, i32, i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_2d.i16.u.v4i32 i1 (p1, v2i16, v2i16, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_2d.s.v4i32 i1 (p1, v2i32, v2i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_2d.u.v4i32 i1 (p1, v2i32, v2i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_2d_array.s.v4i32 i1 (p1, v2i32, i32, v2i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_2d_array.u.v4i32 i1 (p1, v2i32, i32, v2i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_3d.s.v4i32 i1 (p1, v3i32, v3i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_3d.u.v4i32 i1 (p1, v3i32, v3i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_buffer_1d.s.v4i32 i1 (p1, i32, p0, v4i32, i32, i32, i32)
air.atomic_compare_exchange_weak_explicit_texture_buffer_1d.u.v4i32 i1 (p1, i32, p0, v4i32, i32, i32, i32)
air.atomic_exchange_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_exchange_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_1d.i16.s.v4i32 v4i32 (p1, i16, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_1d_array.i16.u.v4i32 v4i32 (p1, i16, i16, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_2d.i16.u.v4i32 v4i32 (p1, v2i16, v2i16, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_2d_array.i16.u.v4i32 v4i32 (p1, v2i16, i16, v2i16, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_3d.i16.s.v4i32 v4i32 (p1, v3i16, v3i16, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_buffer_1d.i16.u.v4i32 v4i32 (p1, i16, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_add_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_and_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_max_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_min_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_or_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_sub_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_fetch_xor_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, v4i32, i32, i32)
air.atomic_load_explicit_texture_1d.s.v4i32 v4i32 (p1, i32, i32, i32)
air.atomic_load_explicit_texture_1d.u.v4i32 v4i32 (p1, i32, i32, i32)
air.atomic_load_explicit_texture_1d_array.s.v4i32 v4i32 (p1, i32, i32, i32, i32)
air.atomic_load_explicit_texture_1d_array.u.v4i32 v4i32 (p1, i32, i32, i32, i32)
air.atomic_load_explicit_texture_2d.i16.u.v4i32 v4i32 (p1, v2i16, v2i16, i32, i32)
air.atomic_load_explicit_texture_2d.s.v4i32 v4i32 (p1, v2i32, v2i32, i32, i32)
air.atomic_load_explicit_texture_2d.u.v4i32 v4i32 (p1, v2i32, v2i32, i32, i32)
air.atomic_load_explicit_texture_2d_array.s.v4i32 v4i32 (p1, v2i32, i32, v2i32, i32, i32)
air.atomic_load_explicit_texture_2d_array.u.v4i32 v4i32 (p1, v2i32, i32, v2i32, i32, i32)
air.atomic_load_explicit_texture_3d.s.v4i32 v4i32 (p1, v3i32, v3i32, i32, i32)
air.atomic_load_explicit_texture_3d.u.v4i32 v4i32 (p1, v3i32, v3i32, i32, i32)
air.atomic_load_explicit_texture_buffer_1d.s.v4i32 v4i32 (p1, i32, i32, i32)
air.atomic_load_explicit_texture_buffer_1d.u.v4i32 v4i32 (p1, i32, i32, i32)
air.atomic_store_explicit_texture_1d.s.v4i32 void (p1, v4i32, i32, i32, i32)
air.atomic_store_explicit_texture_1d.u.v4i32 void (p1, v4i32, i32, i32, i32)
air.atomic_store_explicit_texture_1d_array.s.v4i32 void (p1, v4i32, i32, i32, i32, i32)
air.atomic_store_explicit_texture_1d_array.u.v4i32 void (p1, v4i32, i32, i32, i32, i32)
air.atomic_store_explicit_texture_2d.i16.u.v4i32 void (p1, v4i32, v2i16, v2i16, i32, i32)
air.atomic_store_explicit_texture_2d.s.v4i32 void (p1, v4i32, v2i32, v2i32, i32, i32)
air.atomic_store_explicit_texture_2d.u.v4i32 void (p1, v4i32, v2i32, v2i32, i32, i32)
air.atomic_store_explicit_texture_2d_array.s.v4i32 void (p1, v4i32, v2i32, i32, v2i32, i32, i32)
air.atomic_store_explicit_texture_2d_array.u.v4i32 void (p1, v4i32, v2i32, i32, v2i32, i32, i32)
air.atomic_store_explicit_texture_3d.s.v4i32 void (p1, v4i32, v3i32, v3i32, i32, i32)
air.atomic_store_explicit_texture_3d.u.v4i32 void (p1, v4i32, v3i32, v3i32, i32, i32)
air.atomic_store_explicit_texture_buffer_1d.s.v4i32 void (p1, v4i32, i32, i32, i32)
air.atomic_store_explicit_texture_buffer_1d.u.v4i32 void (p1, v4i32, i32, i32, i32)
";

#[test]
fn observed_declarations_decode() {
    for &(name, decl) in OBSERVED {
        declared_air(name, decl);
    }
    for line in ATOMICS_OBSERVED.lines() {
        let (name, decl) = line.split_once(' ').unwrap();
        let AirIntrinsic::Texture(t) = declared_air(name, decl) else {
            panic!("{name}");
        };
        assert!(matches!(t.op, TexOp::Atomic(_)), "{name}");
    }
    // A known name declared any other way is malformed at its calls.
    for (name, decl) in [
        (
            "air.sample_texture_2d.v4f32",
            "{v4f32} (p1, p2, v2f32, i1, v2i32, i1, f32, f32)",
        ),
        (
            "air.sample_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32, f32, i32)",
        ),
        (
            "air.read_texture_2d.v4f32",
            "{v4f32} (p1, p2, v2i32, i32, i32)",
        ),
        (
            "air.write_texture_2d.v4f32",
            "void (p2, v2i32, v4f32, i32, i32)",
        ),
        ("air.wg.barrier", "void (i32)"),
        ("air.simd_sum.f32", "f16 (f16)"),
        ("air.get_read_sampler", "p1 ()"),
    ] {
        assert_eq!(declared(name, decl), DeclClass::Mismatch, "{name} {decl}");
    }
}

/// A kernel or vertex buffer setter is the strided form when declared with the attribute stride,
/// and AIR 2.5's form, MSL's plain setter, when declared without it; the fragment, object and mesh
/// setters have the stride-less form alone, and no other operand list binds.
#[test]
fn buffer_setter_forms_are_chosen_by_the_declaration() {
    use AddrSpace::{Constant, Device};
    use CommandStage::{Fragment, Kernel, Mesh, Object, Vertex};
    let setter = |stage, space, strided| {
        AirIntrinsic::IndirectCommand(IndirectCommand::SetBuffer {
            stage,
            space,
            strided,
        })
    };
    for (family, stage) in [
        ("set_kernel_buffer_compute_command", Kernel),
        ("set_vertex_buffer_render_command", Vertex),
    ] {
        for (space, overload) in [(Device, "p1i8"), (Constant, "p2i8")] {
            let name = format!("air.{family}.{overload}");
            let ptr = &overload[..2];
            let strided = format!("void (p1, i32, {ptr}, i64, i32)");
            let plain = format!("void (p1, i32, {ptr}, i32)");
            assert_eq!(
                declared_air(&name, &strided),
                setter(stage, space, true),
                "{name}"
            );
            assert_eq!(
                declared_air(&name, &plain),
                setter(stage, space, false),
                "{name}"
            );
            // The plain form keeps the command handle at 0 and the buffer at 2, as the strided.
            let sig = setter(stage, space, false).signature();
            assert_eq!(
                sig.handles().collect::<Vec<_>>(),
                vec![(0, HandleKind::Command)]
            );
            assert_eq!(sig.param(2), Some(Operand::Ptr(space)), "{name}");
            for decl in [
                format!("void (p1, i32, {ptr})"),
                format!("void (p1, i32, {ptr}, i32, i32)"),
                format!("void (p1, i32, {ptr}, i64)"),
            ] {
                assert_eq!(declared(&name, &decl), DeclClass::Mismatch, "{name} {decl}");
            }
        }
    }
    for (family, stage) in [
        ("set_fragment_buffer_render_command", Fragment),
        ("set_object_buffer_render_command", Object),
        ("set_mesh_buffer_render_command", Mesh),
    ] {
        let name = format!("air.{family}.p1i8");
        assert_eq!(
            declared_air(&name, "void (p1, i32, p1, i32)"),
            setter(stage, Device, false),
            "{name}"
        );
        assert_eq!(
            declared(&name, "void (p1, i32, p1, i64, i32)"),
            DeclClass::Mismatch,
            "{name}"
        );
    }
}

#[test]
fn read_forms_are_chosen_by_the_declaration() {
    let current = declared_air(
        "air.read_texture_2d.v4f32",
        "{v4f32} (p1, p2, v2i32, v2i32, i32, i32)",
    );
    let legacy = declared_air("air.read_texture_2d.v4f32", "{v4f32} (p1, v2i32, i32, i32)");
    let (AirIntrinsic::Texture(c), AirIntrinsic::Texture(l)) = (current, legacy) else {
        panic!("{current:?} {legacy:?}");
    };
    assert_eq!((c.form, l.form), (Form::Current, Form::Legacy));
    let roles = |t: Texture| t.layout().positions().map(|(r, _)| r).collect::<Vec<_>>();
    use TexRole as R;
    assert_eq!(
        roles(c),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::Offset,
            R::Lod,
            R::Access
        ]
    );
    assert_eq!(roles(l), [R::Texture, R::Coord, R::Lod, R::Access]);
}

/// Every AIR up to 2.1 states no residency status on a sample, gather, or read: the result is the
/// texel alone rather than `{T, i8}`, and a sample carries no min-LOD-clamp position. AIR 2.2
/// added both in one step. The declaration picks the form, as it does a read's or an atomic's
/// operand positions, so one case covers the range; a read of such a module has the legacy
/// operand form as well.
#[test]
fn the_older_forms_without_a_residency_status_are_chosen_by_the_declaration() {
    use TexRole as R;
    let texture = |name: &str, decl: &str| match declared_air(name, decl) {
        AirIntrinsic::Texture(t) => t,
        other => panic!("{name} {decl}: {other:?}"),
    };
    let roles = |t: Texture| t.layout().positions().map(|(r, _)| r).collect::<Vec<_>>();
    // Every declaration below is one `xcrun metal -std=macos-metal2.1
    // -mmacosx-version-min=10.14` (Apple metal 32023.921) writes, which records
    // `!air.version = !{i32 2, i32 1, i32 0}` and the triple `air64_v21`; the same shapes appear
    // at every lower deployment target the compiler accepts, down to `air64_v18`.
    let cases: &[(&str, &str)] = &[
        // Sampling: no status, and no min-LOD clamp after the bias-or-level value.
        (
            "air.sample_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32, i32)",
        ),
        (
            "air.sample_texture_1d.v4f32",
            "v4f32 (p1, p2, f32, i1, i32, i1, f32, i32)",
        ),
        (
            "air.sample_texture_2d_array.v4f32",
            "v4f32 (p1, p2, v2f32, i32, i1, v2i32, i1, f32, i32)",
        ),
        (
            "air.sample_texture_3d.v4f32",
            "v4f32 (p1, p2, v3f32, i1, v3i32, i1, f32, i32)",
        ),
        // A cube has no offset pair in either form.
        (
            "air.sample_texture_cube.v4f32",
            "v4f32 (p1, p2, v3f32, i1, f32, i32)",
        ),
        (
            "air.sample_texture_cube_array.v4f32",
            "v4f32 (p1, p2, v3f32, i32, i1, f32, i32)",
        ),
        // A gradient sample: the clamp sat between the gradients and the offset pair.
        (
            "air.sample_texture_2d_grad.v4f32",
            "v4f32 (p1, p2, v2f32, v2f32, v2f32, i1, v2i32, i32)",
        ),
        // Depth: the depth word stays before the coordinate.
        (
            "air.sample_depth_2d.f32",
            "f32 (p1, p2, i32, v2f32, i1, v2i32, i1, f32, i32)",
        ),
        (
            "air.sample_compare_depth_2d.f32",
            "f32 (p1, p2, i32, v2f32, f32, i1, v2i32, i1, f32, i32)",
        ),
        (
            "air.sample_depth_cube.f32",
            "f32 (p1, p2, i32, v3f32, i1, f32, i32)",
        ),
        // Gathers have no clamp position in either form; only the result changes.
        (
            "air.gather_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i32, i32)",
        ),
        (
            "air.gather_depth_2d.v4f32",
            "v4f32 (p1, p2, i32, v2f32, i1, v2i32, i32)",
        ),
        (
            "air.gather_compare_depth_2d.f32",
            "v4f32 (p1, p2, i32, v2f32, f32, i1, v2i32, i32)",
        ),
        // Reads: AIR 2.1 writes the legacy operand form (no sampler, no offset) and no status.
        ("air.read_texture_2d.v4f32", "v4f32 (p1, v2i32, i32, i32)"),
        ("air.read_texture_2d.u.v4i32", "v4i32 (p1, v2i32, i32, i32)"),
        ("air.read_texture_2d.v4f16", "v4f16 (p1, v2i32, i32, i32)"),
        ("air.read_texture_1d.v4f32", "v4f32 (p1, i32, i32, i32)"),
        (
            "air.read_texture_2d_array.v4f32",
            "v4f32 (p1, v2i32, i32, i32, i32)",
        ),
        ("air.read_texture_3d.v4f32", "v4f32 (p1, v3i32, i32, i32)"),
        (
            "air.read_texture_cube.v4f32",
            "v4f32 (p1, v2i32, i32, i32, i32)",
        ),
        (
            "air.read_texture_cube_array.v4f32",
            "v4f32 (p1, v2i32, i32, i32, i32, i32)",
        ),
        (
            "air.read_texture_2d_ms.v4f32",
            "v4f32 (p1, v2i32, i32, i32)",
        ),
        ("air.read_texture_buffer_1d.v4f32", "v4f32 (p1, i32, i32)"),
        ("air.read_depth_2d.f32", "f32 (p1, i32, v2i32, i32, i32)"),
    ];
    for (name, decl) in cases {
        let t = texture(name, decl);
        assert!(!t.status, "{name}");
        assert!(!roles(t).contains(&R::MinLodClamp), "{name}");
    }
    // A read of such a module has the legacy operand positions too.
    let read = texture("air.read_texture_2d.v4f32", "v4f32 (p1, v2i32, i32, i32)");
    assert_eq!(read.form, Form::Legacy);
    assert_eq!(roles(read), [R::Texture, R::Coord, R::Lod, R::Access]);
    // The sample's immediates are read at the shifted positions: the LOD mode is still the
    // position before the value, and the access qualifier the last.
    let sample = texture(
        "air.sample_texture_2d.v4f32",
        "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32, i32)",
    );
    assert_eq!(
        roles(sample),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::OffsetFlag,
            R::Offset,
            R::LodMode,
            R::LodValue,
            R::Access
        ]
    );
    // The current forms still decode, and as the status-stating form.
    for (name, decl) in [
        (
            "air.sample_texture_2d.v4f32",
            "{v4f32} (p1, p2, v2f32, i1, v2i32, i1, f32, f32, i32)",
        ),
        (
            "air.read_texture_2d.v4f32",
            "{v4f32} (p1, p2, v2i32, v2i32, i32, i32)",
        ),
        ("air.read_texture_2d.v4f32", "{v4f32} (p1, v2i32, i32, i32)"),
    ] {
        assert!(texture(name, decl).status, "{name} {decl}");
    }
    // The mixtures no emitter has written stay unknown: a status result without the clamp, a
    // clamp without the status, and a read with the sampler and offset but no status.
    for (name, decl) in [
        (
            "air.sample_texture_2d.v4f32",
            "{v4f32} (p1, p2, v2f32, i1, v2i32, i1, f32, i32)",
        ),
        (
            "air.sample_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32, f32, i32)",
        ),
        (
            "air.read_texture_2d.v4f32",
            "v4f32 (p1, p2, v2i32, v2i32, i32, i32)",
        ),
    ] {
        assert!(
            matches!(declared(name, decl), DeclClass::Mismatch),
            "{name} {decl}"
        );
    }
    // An operation that carries no status in any form has no second form from this axis: a
    // write's only alternative is the pre-AIR-2.0 one, which states no access qualifier, and
    // that one is last.
    let write = texture(
        "air.write_texture_2d.v4f32",
        "void (p1, v2i32, v4f32, i32, i32)",
    );
    assert!(write.status && write.states_access);
    let older = write.alternative().expect("the pre-AIR-2.0 form");
    assert_eq!(
        older,
        Texture {
            states_access: false,
            ..write
        }
    );
    assert!(older.alternative().is_none());
}

/// The oldest texture declarations Apple's compiler writes: no residency status, no min-LOD
/// clamp, and no trailing access qualifier either.
///
/// The emitter moved two independent axes at two steps: the trailing access qualifier arrived at
/// AIR 2.0, and the residency status with a sample's min-LOD clamp at AIR 2.2. A deployment
/// target below both writes a declaration missing every one of those positions, and each axis
/// has its own test above (`the_older_forms_without_a_residency_status_...`,
/// `pre_air_2_0_states_no_access_qualifier_or_volatile_flag`), so what is under test here is the
/// combination. Every declaration below is one Apple's compiler wrote for our own MSL.
#[test]
fn the_oldest_forms_state_no_status_no_clamp_and_no_access_qualifier() {
    use TexRole as R;
    let texture = |name: &str, decl: &str| match declared_air(name, decl) {
        AirIntrinsic::Texture(t) => t,
        other => panic!("{name} {decl}: {other:?}"),
    };
    let roles = |t: Texture| t.layout().positions().map(|(r, _)| r).collect::<Vec<_>>();
    // Read from the modules `xcrun metal -c -mmacosx-version-min=10.12` (Apple metal 32023.921)
    // wrote, which record `!air.version = !{i32 1, i32 11, i32 0}`. The same source at
    // `-mmacosx-version-min=10.11` (AIR 1.8) and, through the iOS SDK, at
    // `-mios-version-min=8.0` (AIR 1.6) declares every one of these shapes identically, so the
    // whole range below AIR 2.0 is this one set.
    let cases: &[(&str, &str)] = &[
        (
            "air.sample_texture_1d.v4f32",
            "v4f32 (p1, p2, f32, i1, i32, i1, f32)",
        ),
        (
            "air.sample_texture_1d_array.v4f32",
            "v4f32 (p1, p2, f32, i32, i1, i32, i1, f32)",
        ),
        (
            "air.sample_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32)",
        ),
        (
            "air.sample_texture_2d_grad.v4f32",
            "v4f32 (p1, p2, v2f32, v2f32, v2f32, i1, v2i32)",
        ),
        (
            "air.sample_texture_2d_array.v4f32",
            "v4f32 (p1, p2, v2f32, i32, i1, v2i32, i1, f32)",
        ),
        (
            "air.sample_texture_3d.v4f32",
            "v4f32 (p1, p2, v3f32, i1, v3i32, i1, f32)",
        ),
        (
            "air.sample_texture_cube.v4f32",
            "v4f32 (p1, p2, v3f32, i1, f32)",
        ),
        (
            "air.sample_texture_cube_array.v4f32",
            "v4f32 (p1, p2, v3f32, i32, i1, f32)",
        ),
        (
            "air.gather_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i32)",
        ),
        (
            "air.gather_texture_2d_array.v4f32",
            "v4f32 (p1, p2, v2f32, i32, i1, v2i32, i32)",
        ),
        (
            "air.gather_texture_cube.v4f32",
            "v4f32 (p1, p2, v3f32, i32)",
        ),
        (
            "air.gather_texture_cube_array.v4f32",
            "v4f32 (p1, p2, v3f32, i32, i32)",
        ),
        (
            "air.sample_depth_2d.f32",
            "f32 (p1, p2, i32, v2f32, i1, v2i32, i1, f32)",
        ),
        (
            "air.sample_compare_depth_2d.f32",
            "f32 (p1, p2, i32, v2f32, f32, i1, v2i32, i1, f32)",
        ),
        (
            "air.sample_depth_2d_array.f32",
            "f32 (p1, p2, i32, v2f32, i32, i1, v2i32, i1, f32)",
        ),
        (
            "air.sample_depth_cube.f32",
            "f32 (p1, p2, i32, v3f32, i1, f32)",
        ),
        (
            "air.gather_depth_2d.v4f32",
            "v4f32 (p1, p2, i32, v2f32, i1, v2i32)",
        ),
        (
            "air.gather_compare_depth_2d.f32",
            "v4f32 (p1, p2, i32, v2f32, f32, i1, v2i32)",
        ),
        (
            "air.gather_compare_depth_cube.f32",
            "v4f32 (p1, p2, i32, v3f32, f32)",
        ),
        ("air.read_texture_1d.v4f32", "v4f32 (p1, i32, i32)"),
        (
            "air.read_texture_1d_array.v4f32",
            "v4f32 (p1, i32, i32, i32)",
        ),
        ("air.read_texture_2d.v4f32", "v4f32 (p1, v2i32, i32)"),
        ("air.read_texture_2d.s.v4i32", "v4i32 (p1, v2i32, i32)"),
        ("air.read_texture_2d.u.v4i32", "v4i32 (p1, v2i32, i32)"),
        (
            "air.read_texture_2d_array.v4f32",
            "v4f32 (p1, v2i32, i32, i32)",
        ),
        ("air.read_texture_3d.v4f32", "v4f32 (p1, v3i32, i32)"),
        ("air.read_texture_cube.v4f32", "v4f32 (p1, v2i32, i32, i32)"),
        (
            "air.read_texture_cube_array.v4f32",
            "v4f32 (p1, v2i32, i32, i32, i32)",
        ),
        ("air.read_texture_2d_ms.v4f32", "v4f32 (p1, v2i32, i32)"),
        ("air.read_depth_2d.f32", "f32 (p1, i32, v2i32, i32)"),
        (
            "air.read_depth_2d_array.f32",
            "f32 (p1, i32, v2i32, i32, i32)",
        ),
    ];
    for (name, decl) in cases {
        let t = texture(name, decl);
        let roles = roles(t);
        assert!(!t.status, "{name} {decl}: a residency status");
        assert!(!t.states_access, "{name} {decl}: an access qualifier");
        assert!(!roles.contains(&R::MinLodClamp), "{name} {decl}: a clamp");
        assert!(!roles.contains(&R::Access), "{name} {decl}: a qualifier");
        // A read of such a module takes the legacy operand positions as well: the sampler and
        // the offset arrived at AIR 2.8, later than either of the other two.
        if t.op == TexOp::Read {
            assert_eq!(t.form, Form::Legacy, "{name} {decl}");
            assert!(!roles.contains(&R::Sampler), "{name} {decl}: a sampler");
            assert!(!roles.contains(&R::Offset), "{name} {decl}: an offset");
        }
    }
    // The positions that are there keep their meaning and their order: a sample ends at the
    // bias-or-level value, a gather at its component, a read at its level.
    assert_eq!(
        roles(texture(
            "air.sample_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32)"
        )),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::OffsetFlag,
            R::Offset,
            R::LodMode,
            R::LodValue
        ]
    );
    assert_eq!(
        roles(texture(
            "air.gather_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i32)"
        )),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::OffsetFlag,
            R::Offset,
            R::Component
        ]
    );
    assert_eq!(
        roles(texture(
            "air.read_texture_2d.v4f32",
            "v4f32 (p1, v2i32, i32)"
        )),
        [R::Texture, R::Coord, R::Lod]
    );
    // A gradient sample's clamp sat between the gradients and the offset pair, so dropping it
    // leaves the offset pair last.
    assert_eq!(
        roles(texture(
            "air.sample_texture_2d_grad.v4f32",
            "v4f32 (p1, p2, v2f32, v2f32, v2f32, i1, v2i32)"
        )),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::Dpdx,
            R::Dpdy,
            R::OffsetFlag,
            R::Offset
        ]
    );
    // Dropping one axis and not the other is a declaration no emitter wrote, and no form of
    // these names: a status with no qualifier, and a qualifier with no status but a clamp.
    for (name, decl) in [
        (
            "air.sample_texture_2d.v4f32",
            "{v4f32} (p1, p2, v2f32, i1, v2i32, i1, f32, f32)",
        ),
        ("air.read_texture_2d.v4f32", "{v4f32} (p1, v2i32, i32)"),
        (
            "air.sample_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32, f32)",
        ),
    ] {
        assert_eq!(declared(name, decl), DeclClass::Mismatch, "{name} {decl}");
    }
}

/// A 2D or 3D texture atomic's two forms: Xcode's AIR passes an offset after the coordinate (and
/// layer), the AIR macOS 26.6's Metal on x86-64 hands its device omits it. The declaration picks
/// the form; a 1D, 1D array or buffer atomic has one form, without an offset.
#[test]
fn atomic_forms_are_chosen_by_the_declaration() {
    use TexRole as R;
    let roles = |t: Texture| t.layout().positions().map(|(r, _)| r).collect::<Vec<_>>();
    let texture = |name: &str, decl: &str| match declared_air(name, decl) {
        AirIntrinsic::Texture(t) => t,
        other => panic!("{name}: {other:?}"),
    };
    let fetch = "air.atomic_fetch_add_explicit_texture_2d.u.v4i32";
    let current = texture(fetch, "v4i32 (p1, v2i32, v2i32, v4i32, i32, i32)");
    let legacy = texture(fetch, "v4i32 (p1, v2i32, v4i32, i32, i32)");
    assert_eq!((current.form, legacy.form), (Form::Current, Form::Legacy));
    assert_eq!(
        roles(legacy),
        [R::Texture, R::Coord, R::Value, R::Order, R::Access]
    );
    let legacy_cases = [
        (
            "air.atomic_fetch_xor_explicit_texture_2d.s.v4i32",
            "v4i32 (p1, v2i32, v4i32, i32, i32)",
        ),
        (
            "air.atomic_fetch_min_explicit_texture_2d_array.u.v4i32",
            "v4i32 (p1, v2i32, i32, v4i32, i32, i32)",
        ),
        (
            "air.atomic_exchange_explicit_texture_3d.s.v4i32",
            "v4i32 (p1, v3i32, v4i32, i32, i32)",
        ),
        (
            "air.atomic_load_explicit_texture_2d.u.v4i32",
            "v4i32 (p1, v2i32, i32, i32)",
        ),
        (
            "air.atomic_store_explicit_texture_2d_array.s.v4i32",
            "void (p1, v4i32, v2i32, i32, i32, i32)",
        ),
        (
            "air.atomic_compare_exchange_weak_explicit_texture_2d.u.v4i32",
            "i1 (p1, v2i32, p0, v4i32, i32, i32, i32)",
        ),
        (
            "air.atomic_max_explicit_texture_2d.u.v4i64",
            "void (p1, v2i32, v4i64, i32, i32)",
        ),
        (
            "air.atomic_fetch_max_explicit_texture_2d.i16.u.v4i32",
            "v4i32 (p1, v2i16, v4i32, i32, i32)",
        ),
    ];
    for (name, decl) in legacy_cases {
        let t = texture(name, decl);
        assert_eq!(t.form, Form::Legacy, "{name}");
        assert!(!roles(t).contains(&R::Offset), "{name}");
    }
    // One form where the shape has no offset: a 1D atomic declared with one more position is not
    // a form of it.
    let one_d = "air.atomic_fetch_add_explicit_texture_1d.u.v4i32";
    assert_eq!(
        texture(one_d, "v4i32 (p1, i32, v4i32, i32, i32)").form,
        Form::Current
    );
    assert!(matches!(
        declared(one_d, "v4i32 (p1, i32, i32, v4i32, i32, i32)"),
        DeclClass::Mismatch
    ));
}

#[test]
fn texture_layouts_name_every_position() {
    use TexRole as R;
    let roles = |name: &str| {
        let AirIntrinsic::Texture(t) = air(name) else {
            panic!("{name}");
        };
        t.layout().positions().map(|(r, _)| r).collect::<Vec<_>>()
    };
    assert_eq!(
        roles("air.sample_compare_depth_2d_array.f32"),
        [
            R::Texture,
            R::Sampler,
            R::DepthWord,
            R::Coord,
            R::Layer,
            R::CompareRef,
            R::OffsetFlag,
            R::Offset,
            R::LodMode,
            R::LodValue,
            R::MinLodClamp,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.sample_texture_2d_array_grad.v4f32"),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::Layer,
            R::Dpdx,
            R::Dpdy,
            R::MinLodClamp,
            R::OffsetFlag,
            R::Offset,
            R::Access
        ]
    );
    // MSL `read(coord, face, array, lod)`: the face comes first (declaration observed from
    // `texturecube_array::read`).
    assert_eq!(
        roles("air.read_texture_cube_array.v4f32"),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::Face,
            R::Layer,
            R::Lod,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.write_texture_cube.v4f16"),
        [R::Texture, R::Coord, R::Face, R::Value, R::Lod, R::Access]
    );
    // MSL `write(color, coord, face, array, lod)`: the face comes first, as in a read
    // (declaration observed from `texturecube_array::write`).
    assert_eq!(
        roles("air.write_texture_cube_array.v4f32"),
        [
            R::Texture,
            R::Coord,
            R::Face,
            R::Layer,
            R::Value,
            R::Lod,
            R::Access
        ]
    );
    // A 1D array's layer follows its coordinate and it has no offset; a depth cube array's face
    // precedes its array index after the depth word; a texture buffer's width takes no level.
    assert_eq!(
        roles("air.read_texture_1d_array.v4f32"),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::Layer,
            R::Lod,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.read_depth_cube_array.f32"),
        [
            R::Texture,
            R::Sampler,
            R::DepthWord,
            R::Coord,
            R::Face,
            R::Layer,
            R::Lod,
            R::Access
        ]
    );
    assert_eq!(roles("air.get_width_texture_buffer_1d"), [R::Texture]);
    assert_eq!(roles("air.get_width_depth_cube"), [R::Texture, R::Lod]);
    assert_eq!(
        roles("air.gather_texture_2d.u.v4i32"),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::OffsetFlag,
            R::Offset,
            R::Component,
            R::Access
        ]
    );
    // A cube takes no offset, in a gradient sample or a gather.
    assert_eq!(
        roles("air.sample_compare_depth_cube_array_grad.f32"),
        [
            R::Texture,
            R::Sampler,
            R::DepthWord,
            R::Coord,
            R::Layer,
            R::CompareRef,
            R::Dpdx,
            R::Dpdy,
            R::MinLodClamp,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.sample_compare_depth_2d_grad.f32"),
        [
            R::Texture,
            R::Sampler,
            R::DepthWord,
            R::Coord,
            R::CompareRef,
            R::Dpdx,
            R::Dpdy,
            R::MinLodClamp,
            R::OffsetFlag,
            R::Offset,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.gather_texture_cube_array.v4f32"),
        [
            R::Texture,
            R::Sampler,
            R::Coord,
            R::Layer,
            R::Component,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.gather_compare_depth_2d_array.f32"),
        [
            R::Texture,
            R::Sampler,
            R::DepthWord,
            R::Coord,
            R::Layer,
            R::CompareRef,
            R::OffsetFlag,
            R::Offset,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.calculate_clamped_lod_depth_cube_array"),
        [R::Texture, R::Sampler, R::Coord, R::Access]
    );
    // A texture atomic's layer follows its coordinate, a 2D or 3D one passes an offset, a store
    // passes its value first, and a compare-exchange passes `expected` before its desired texel and
    // then two orders.
    assert_eq!(
        roles("air.atomic_fetch_add_explicit_texture_2d_array.u.v4i32"),
        [
            R::Texture,
            R::Coord,
            R::Layer,
            R::Offset,
            R::Value,
            R::Order,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.atomic_store_explicit_texture_1d_array.s.v4i32"),
        [
            R::Texture,
            R::Value,
            R::Coord,
            R::Layer,
            R::Order,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.atomic_compare_exchange_weak_explicit_texture_3d.u.v4i32"),
        [
            R::Texture,
            R::Coord,
            R::Offset,
            R::Expected,
            R::Value,
            R::Order,
            R::FailureOrder,
            R::Access
        ]
    );
    assert_eq!(
        roles("air.atomic_load_explicit_texture_buffer_1d.u.v4i32"),
        [R::Texture, R::Coord, R::Order, R::Access]
    );
}

/// The reads and writes of Metal's `ulong` textures, as Apple's metal 32023.921 declares them: `u`
/// `v4i64` texels of 1D, 1D array, 2D, 2D array, 3D, and buffer textures, at 32- and 16-bit
/// coordinates, in the layouts of every other texel.
const ULONG_OBSERVED: &str = "\
air.read_texture_1d.u.v4i64 {v4i64} (p1, p2, i32, i32, i32)
air.read_texture_1d_array.u.v4i64 {v4i64} (p1, p2, i32, i32, i32, i32)
air.read_texture_2d.u.v4i64 {v4i64} (p1, p2, v2i32, v2i32, i32, i32)
air.read_texture_2d.i16.u.v4i64 {v4i64} (p1, p2, v2i16, v2i16, i16, i32)
air.read_texture_2d_array.u.v4i64 {v4i64} (p1, p2, v2i32, i32, v2i32, i32, i32)
air.read_texture_3d.u.v4i64 {v4i64} (p1, p2, v3i32, v3i32, i32, i32)
air.read_texture_buffer_1d.u.v4i64 {v4i64} (p1, p2, i32, i32)
air.write_texture_1d.u.v4i64 void (p1, i32, v4i64, i32, i32)
air.write_texture_1d_array.u.v4i64 void (p1, i32, i32, v4i64, i32, i32)
air.write_texture_2d.u.v4i64 void (p1, v2i32, v4i64, i32, i32)
air.write_texture_2d.i16.u.v4i64 void (p1, v2i16, v4i64, i16, i32)
air.write_texture_2d_array.u.v4i64 void (p1, v2i32, i32, v4i64, i32, i32)
air.write_texture_3d.u.v4i64 void (p1, v3i32, v4i64, i32, i32)
air.write_texture_buffer_1d.u.v4i64 void (p1, i32, v4i64, i32)
air.atomic_max_explicit_texture_1d.u.v4i64 void (p1, i32, v4i64, i32, i32)
air.atomic_max_explicit_texture_1d_array.u.v4i64 void (p1, i32, i32, v4i64, i32, i32)
air.atomic_max_explicit_texture_2d.u.v4i64 void (p1, v2i32, v2i32, v4i64, i32, i32)
air.atomic_max_explicit_texture_2d.i16.u.v4i64 void (p1, v2i16, v2i16, v4i64, i32, i32)
air.atomic_max_explicit_texture_2d_array.u.v4i64 void (p1, v2i32, i32, v2i32, v4i64, i32, i32)
air.atomic_max_explicit_texture_3d.u.v4i64 void (p1, v3i32, v3i32, v4i64, i32, i32)
air.atomic_max_explicit_texture_buffer_1d.u.v4i64 void (p1, i32, v4i64, i32, i32)
air.atomic_min_explicit_texture_1d.u.v4i64 void (p1, i32, v4i64, i32, i32)
air.atomic_min_explicit_texture_1d_array.u.v4i64 void (p1, i32, i32, v4i64, i32, i32)
air.atomic_min_explicit_texture_2d.u.v4i64 void (p1, v2i32, v2i32, v4i64, i32, i32)
air.atomic_min_explicit_texture_2d.i16.u.v4i64 void (p1, v2i16, v2i16, v4i64, i32, i32)
air.atomic_min_explicit_texture_2d_array.u.v4i64 void (p1, v2i32, i32, v2i32, v4i64, i32, i32)
air.atomic_min_explicit_texture_3d.u.v4i64 void (p1, v3i32, v3i32, v4i64, i32, i32)
air.atomic_min_explicit_texture_buffer_1d.u.v4i64 void (p1, i32, v4i64, i32, i32)
";

/// Apple's compiler's `ulong` texture declarations (metal 32023.921): reads, writes, and the
/// result-less `atomic_{min,max}`.
#[test]
fn ulong_texture_reads_and_writes_decode() {
    for line in ULONG_OBSERVED.lines() {
        let (name, decl) = line.split_once(' ').unwrap();
        let AirIntrinsic::Texture(t) = declared_air(name, decl) else {
            panic!("{name}");
        };
        assert!(
            matches!(
                t.op,
                TexOp::Read | TexOp::Write | TexOp::Atomic(AtomicOp::Min | AtomicOp::Max)
            ),
            "{name}"
        );
        assert!(t.ulong(), "{name}");
        let texel = t.texel.unwrap();
        assert_eq!(
            (texel.ty.elem, texel.ty.lanes, texel.sign),
            (super::ty::Elem::I64, 4, Some(super::ty::Sign::Unsigned)),
            "{name}"
        );
    }
    // Apple's compiler builds no other `ulong` texture or operation, and no `long` texture.
    for name in [
        "air.read_texture_cube.u.v4i64",
        "air.read_texture_cube_array.u.v4i64",
        "air.read_texture_2d_ms.u.v4i64",
        "air.read_depth_2d.u.v4i64",
        "air.write_texture_cube.u.v4i64",
        "air.read_texture_2d.s.v4i64",
        "air.write_texture_2d.s.v4i64",
        "air.read_texture_2d.v4i64",
        "air.read_texture_2d.u.v2i64",
        "air.sample_texture_2d.u.v4i64",
        "air.gather_texture_2d.u.v4i64",
        "air.atomic_fetch_max_explicit_texture_2d.u.v4i64",
        "air.atomic_exchange_explicit_texture_2d.u.v4i64",
        "air.atomic_store_explicit_texture_2d.u.v4i64",
        // The result-less stems are the `ulong` texel's alone.
        "air.atomic_min_explicit_texture_2d.u.v4i32",
        "air.atomic_max_explicit_texture_2d.s.v4i32",
        "air.atomic_max_explicit_texture_2d.s.v4i64",
        "air.atomic_min_explicit_texture_cube.u.v4i64",
        "air.atomic_max_explicit_texture_2d_ms.u.v4i64",
    ] {
        assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
    }
}

#[test]
fn texture_grammar_admits_only_observed_shapes() {
    for name in [
        "air.sample_texture_2d_ms.v4f32", // no sampling of multisampled textures
        "air.sample_texture_buffer_1d.v4f32", // nor of texture buffers
        "air.sample_compare_texture_2d.v4f32", // compare is depth-only
        "air.sample_texture_2d.i16.v4f32", // sampling takes float coordinates
        "air.sample_texture_2d.v4i32",    // an integer texel spells its sign
        "air.sample_texture_2d.s.v4f32",  // a float texel spells none
        "air.sample_texture_2d.v3f32",    // a color texel has four lanes
        "air.sample_depth_2d.v4f32",      // a depth sample is a scalar
        "air.read_texture_2d.v4bf16",     // no bfloat texels
        "air.write_depth_2d.f32",         // depth writes are not observed
        "air.gather_texture_1d.v4f32",    // nor 1D gathers
        "air.gather_compare_texture_2d.f32", // compare is depth-only
        "air.calculate_clamped_lod_texture_1d", // no LOD query of a 1D texture
        "air.atomic_fetch_add_explicit_texture_cube.u.v4i32", // no cube atomics
        "air.atomic_fetch_add_explicit_texture_2d_ms.u.v4i32", // nor multisampled ones
        "air.atomic_fetch_add_explicit_depth_2d.f32", // nor depth ones
        "air.atomic_fetch_add_explicit_texture_2d.v4f32", // atomics take integer texels
        "air.atomic_load_explicit_texture_2d.u.v4i16", // of 32 bits
        "air.get_depth_texture_2d",       // a 2D texture has no depth
        "air.get_num_samples_texture_2d", // nor samples
        "air.fence_texture_2d.v4f32",     // a fence spells no texel
        "air.sample_texture_2d.v4f32.extra", // trailing token
        "air.sample_surface_2d.v4f32",    // unknown object kind
    ] {
        assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
    }
}

/// The declaration is what chooses a texture operation's operand form, over the whole grammar.
///
/// Each name's forms are a chain ([`Texture::alternative`]) walked newest first: the current
/// form, then a read's or a 2D or 3D atomic's legacy operand positions, then the form that states
/// no residency status, then the one that states no trailing access qualifier. The emitter added
/// them in the opposite order and each shape implies the ones before it -- AIR 2.8's sampler and
/// offset only ever appear with AIR 2.2's status, which only ever appears with AIR 2.0's
/// qualifier -- so the chain is exactly the set of shapes some emitter wrote, and the chain's
/// length is the number of axes the operation has.
///
/// Reading each form back is what makes the selection honest and version-free: declared by
/// itself, a form must classify as *itself*. If two forms of one name ever shared a function
/// type, the first in the chain would answer for both and the choice would be a guess. This holds
/// for every spelling the grammar admits, not only for the declarations that have been observed.
#[test]
fn every_texture_form_is_recovered_from_its_own_declaration() {
    let texels = [
        "",
        ".v4f32",
        ".v4f16",
        ".s.v4i32",
        ".u.v4i16",
        ".u.v4i64",
        ".f32",
        ".f16",
        ".i16.v4f32",
        ".i16.s.v4i32",
    ];
    let mut forms = 0;
    let mut chains = 0;
    for &(stem, _, textures, depths) in TEX_OPS {
        for (kind, dims) in [("texture", textures), ("depth", depths)] {
            for dim in dims {
                for texel in texels {
                    let name = format!("air.{stem}_{kind}_{dim}{texel}");
                    let Some(first) = Texture::parse(name.strip_prefix("air.").unwrap()) else {
                        continue;
                    };
                    chains += 1;
                    let chain: Vec<Texture> =
                        std::iter::successors(Some(first), |t| t.alternative()).collect();
                    // The chain is exactly the axes this operation has: its current form, a
                    // read's or a 2D or 3D atomic's legacy operand positions, a residency status
                    // where the result carries one, and the trailing access qualifier where the
                    // layout states one.
                    let legacy = match first.op {
                        TexOp::Read => true,
                        TexOp::Atomic(_) => {
                            matches!(first.image.dim, ImageDim::D2 | ImageDim::D3)
                        }
                        _ => false,
                    };
                    let status = matches!(
                        first.op,
                        TexOp::Sample { .. }
                            | TexOp::SampleGrad { .. }
                            | TexOp::Gather { .. }
                            | TexOp::Read
                    );
                    let axes = usize::from(legacy)
                        + usize::from(status)
                        + usize::from(first.layout().index(TexRole::Access).is_some());
                    assert_eq!(
                        chain.len(),
                        1 + axes,
                        "{name}: {} forms for {axes} axes",
                        chain.len()
                    );
                    for (i, &form) in chain.iter().enumerate() {
                        let decl = texture_declaration(form);
                        let got = match declared(&name, &decl) {
                            DeclClass::Air(AirIntrinsic::Texture(t)) => t,
                            other => panic!("{name} form {i} ({decl}): {other:?}"),
                        };
                        assert_eq!(
                            got, form,
                            "{name} form {i} ({decl}) was answered by another form of the same \
                             name: the declaration does not choose between them"
                        );
                        forms += 1;
                    }
                }
            }
        }
    }
    // The grammar is wide and most names have more than one form, so a parse that stopped
    // admitting spellings could not pass quietly.
    assert!(chains > 400, "{chains} chains");
    assert!(forms > chains * 2, "{forms} forms over {chains} chains");
}

/// `t`'s signature spelled in the tokens [`declared`] reads, so a form can be declared by itself.
fn texture_declaration(t: Texture) -> String {
    let sig = t.layout().signature();
    let params = sig
        .params()
        .iter()
        .map(|&p| operand_token(p))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} ({params})", operand_token(sig.ret))
}

/// One operand as a [`ty`] token.
fn operand_token(op: Operand) -> String {
    match op {
        Operand::Void => "void".to_string(),
        Operand::Num(n) => num_token(n),
        Operand::WithStatus(n) => format!("{{{}}}", num_token(n)),
        Operand::Ptr(space) => format!("p{}", space as u8),
        Operand::Handle(h) => format!("p{}", h.space() as u8),
        Operand::Imm(ImmKind::Bool) => "i1".to_string(),
        Operand::Imm(ImmKind::I32) => "i32".to_string(),
        Operand::IntersectionResult(_) => panic!("not a texture operand"),
    }
}

fn num_token(n: NumTy) -> String {
    let elem = match n.elem {
        Elem::I1 => "i1",
        Elem::I8 => "i8",
        Elem::I16 => "i16",
        Elem::I32 => "i32",
        Elem::I64 => "i64",
        Elem::F16 => "f16",
        Elem::BF16 => "bf16",
        Elem::F32 => "f32",
    };
    if n.lanes == 1 {
        elem.to_string()
    } else {
        format!("v{}{elem}", n.lanes)
    }
}

/// Every texture table entry decodes against its own signature, and every layout fits.
#[test]
fn texture_tables_are_total() {
    let texels = [
        "v4f32", "v4f16", "s.v4i32", "u.v4i16", "u.v4i64", "f32", "f16",
    ];
    let mut decoded = 0;
    for &(stem, _, textures, depths) in TEX_OPS {
        for (kind, dims) in [("texture", textures), ("depth", depths)] {
            for dim in dims {
                let family = format!("air.{stem}_{kind}_{dim}");
                let mut spellings = vec![family.clone()];
                for coord in ["", ".i16"] {
                    for texel in texels {
                        spellings.push(format!("{family}{coord}.{texel}"));
                    }
                }
                let hits = spellings
                    .iter()
                    .filter(|n| matches!(classify_self(n), DeclClass::Air(_)))
                    .count();
                assert!(hits > 0, "{family}: no spelling decodes");
                decoded += hits;
            }
        }
    }
    assert!(decoded > 150, "{decoded}");
}

#[test]
fn group_tables_are_total() {
    let mut decoded = 0;
    for &(stem, op, ..) in GROUP_OPS {
        for scope in ["simd", "quad"] {
            for suffix in ["", ".f32", ".v4f16", ".s.i32", ".u.v2i16", ".i64"] {
                let name = format!("air.{scope}_{stem}{suffix}");
                if let DeclClass::Air(AirIntrinsic::Group(g)) = classify_self(&name) {
                    assert_eq!(g.op, op, "{name}");
                    decoded += 1;
                }
            }
        }
    }
    // Every operation decodes in at least one spelling.
    for &(stem, op, ..) in GROUP_OPS {
        let any = ["simd", "quad"].iter().any(|scope| {
            ["", ".f32", ".i64", ".s.i32"].iter().any(|s| {
                matches!(
                    classify_self(&format!("air.{scope}_{stem}{s}")),
                    DeclClass::Air(AirIntrinsic::Group(g)) if g.op == op
                )
            })
        });
        assert!(any, "{stem}");
    }
    assert!(decoded > 60, "{decoded}");
    for name in [
        "air.simd_sum.i32",                 // an integer value spells its sign
        "air.simd_sum.s.f32",               // a float value spells none
        "air.simd_and.f32",                 // bitwise reductions are integer-only
        "air.quad_ballot.i64",              // a quad ballot spells no type: its mask is an i16
        "air.simd_active_threads_mask",     // a simdgroup's active mask spells its i64
        "air.quad_active_threads_mask.i64", // a quad's spells none
        "air.simd_ballot.i32",              // the ballot mask is i64
        "air.simd_all.i1",                  // votes spell no type
        "air.simd_sum.bf16",                // no bfloat group operations
        "air.quad_and.f32",                 // a quad's bitwise reductions are integer-only too
    ] {
        assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
    }
}

fn texture_template(name: &str) -> DeclClass {
    DeclClass::Air(air(name))
}

#[test]
fn texture_immediates_bind() {
    let mut c = Consts::new();
    let v = |i| Value::Arg(crate::ir::ArgIdx::new(i));
    let sample = texture_template("air.sample_texture_2d.v4f32");
    // (tex, smp, coord, i1 offset, offset, i1 level, lod, min_lod, access)
    let (yes, no, zero) = (c.bool(true), c.bool(false), c.i32(0));
    let args = [v(0), v(1), v(2), yes, v(4), yes, v(6), v(7), zero];
    let Ok(Bound::Air(AirIntrinsic::Texture(t))) = bind(sample, &args, &c.table) else {
        panic!("bind");
    };
    assert_eq!(
        t.op,
        TexOp::Sample {
            compare: false,
            lod: LodMode::Level,
            offset: true
        }
    );
    assert_eq!(t.access, Some(Access::Sample));
    let args = [v(0), v(1), v(2), no, v(4), no, v(6), v(7), zero];
    let Ok(Bound::Air(AirIntrinsic::Texture(t))) = bind(sample, &args, &c.table) else {
        panic!("bind");
    };
    assert!(matches!(
        t.op,
        TexOp::Sample {
            lod: LodMode::Bias,
            offset: false,
            ..
        }
    ));
    // An access outside the qualifiers stays unknown; a non-constant flag is malformed.
    let args = [v(0), v(1), v(2), no, v(4), no, v(6), v(7), c.i32(4)];
    assert_eq!(bind(sample, &args, &c.table), Ok(Bound::Unknown));
    let args = [v(0), v(1), v(2), v(3), v(4), no, v(6), v(7), zero];
    assert_eq!(
        bind(sample, &args, &c.table),
        Err(IntrinsicError::ImmediateNotConstant)
    );
    // A gather reads its component; a depth gather has none.
    let gather = texture_template("air.gather_texture_2d.v4f32");
    let args = [v(0), v(1), v(2), yes, v(4), c.i32(2), zero];
    let Ok(Bound::Air(AirIntrinsic::Texture(t))) = bind(gather, &args, &c.table) else {
        panic!("bind");
    };
    assert_eq!(
        t.op,
        TexOp::Gather {
            component: Some(Component::Z),
            compare: false,
            offset: true
        }
    );
    let args = [v(0), v(1), v(2), yes, v(4), c.i32(4), zero];
    assert_eq!(bind(gather, &args, &c.table), Ok(Bound::Unknown));
    let depth = texture_template("air.gather_depth_2d.v4f32");
    let args = [v(0), v(1), v(2), v(3), yes, v(5), zero];
    let Ok(Bound::Air(AirIntrinsic::Texture(t))) = bind(depth, &args, &c.table) else {
        panic!("bind");
    };
    assert_eq!(
        t.op,
        TexOp::Gather {
            component: None,
            compare: false,
            offset: true
        }
    );
    // A write reads its access qualifier.
    let write = texture_template("air.write_texture_2d.v4f32");
    let args = [v(0), v(1), v(2), v(3), c.i32(3)];
    let Ok(Bound::Air(AirIntrinsic::Texture(t))) = bind(write, &args, &c.table) else {
        panic!("bind");
    };
    assert_eq!(t.access, Some(Access::ReadWrite));
}

#[test]
fn barrier_and_fence_immediates_bind() {
    let mut c = Consts::new();
    let wg = DeclClass::Air(air("air.wg.barrier"));
    let sg = DeclClass::Air(air("air.simdgroup.barrier"));
    let args = [c.i32(2 | 4), c.i32(1)];
    let Ok(Bound::Air(AirIntrinsic::Sync(s))) = bind(wg, &args, &c.table) else {
        panic!("bind");
    };
    let flags = MemFlags {
        threadgroup: true,
        texture: true,
        ..MemFlags::default()
    };
    assert_eq!(
        s,
        Sync::Barrier {
            name: BarrierName::Threadgroup,
            flags,
            order: None,
            scope: ThreadScope::Threadgroup,
            memory: ThreadScope::Threadgroup,
        }
    );
    assert_eq!(s.barrier_order(), Some(MemoryOrder::SeqCst));
    // The AIR 2.8 scope is the operand's, whichever intrinsic spells it, and is also the memory
    // scope.
    let args = [c.i32(0), c.i32(1)];
    let Ok(Bound::Air(AirIntrinsic::Sync(s))) = bind(sg, &args, &c.table) else {
        panic!("bind");
    };
    assert!(matches!(
        s,
        Sync::Barrier {
            name: BarrierName::Simdgroup,
            scope: ThreadScope::Threadgroup,
            memory: ThreadScope::Threadgroup,
            ..
        }
    ));
    // Unknown flag bits and non-barrier scopes stay unknown.
    for args in [
        [c.i32(32), c.i32(1)],
        [c.i32(1), c.i32(2)],
        [c.i32(1), c.i32(7)],
    ] {
        assert_eq!(bind(wg, &args, &c.table), Ok(Bound::Unknown));
    }
    let v = Value::Arg(crate::ir::ArgIdx::new(0));
    assert_eq!(
        bind(wg, &[v, c.i32(1)], &c.table),
        Err(IntrinsicError::ImmediateNotConstant)
    );
    // A fence reads flags, order, and scope, including the legacy all-devices scope.
    let fence = DeclClass::Air(air("air.atomic.fence"));
    let args = [c.i32(1), c.i32(5), c.i32(3)];
    let Ok(Bound::Air(AirIntrinsic::Sync(s))) = bind(fence, &args, &c.table) else {
        panic!("bind");
    };
    assert_eq!(
        s,
        Sync::Fence {
            flags: MemFlags {
                device: true,
                ..MemFlags::default()
            },
            order: MemoryOrder::SeqCst,
            scope: ThreadScope::AllDevices
        }
    );
    let args = [c.i32(1), c.i32(1), c.i32(2)];
    assert_eq!(bind(fence, &args, &c.table), Ok(Bound::Unknown));
    // `mem_object_data` is a flag (MSL 4.1's every-flag mask is 31).
    let args = [c.i32(31), c.i32(5), c.i32(2)];
    let Ok(Bound::Air(AirIntrinsic::Sync(Sync::Fence { flags, .. }))) =
        bind(fence, &args, &c.table)
    else {
        panic!("bind");
    };
    assert_eq!(
        flags,
        MemFlags {
            device: true,
            threadgroup: true,
            texture: true,
            threadgroup_imageblock: true,
            object_data: true,
        }
    );
}

/// AIR 2.9 (deployment target macOS 27) spells `frexp` as the intrinsic `air.frexp.<T>`, where
/// every earlier emitter outlines it as the runtime routine `___metal_frexp_<T>_pthread<I>`. Both
/// spellings read to the same `Math`, so one expansion serves them (`canon::routine`), and the
/// routine's name is still not an `air.*` one.
#[test]
fn air_2_9_frexp_reads_as_the_outlined_routine() {
    for (ty, routine_ty) in [
        ("f32", "float_pthreadint32"),
        ("f16", "half_pthreadint32"),
        ("v4f32", "v4float_pthreadv4int32"),
        ("v2f16", "v2half_pthreadv2int32"),
    ] {
        let decl = format!("{ty} ({ty}, p0)");
        let intrinsic = declared_air(&format!("air.frexp.{ty}"), &decl);
        let routine = declared_air(&format!("___metal_frexp_{routine_ty}"), &decl);
        assert_eq!(intrinsic, routine, "{ty}");
        let AirIntrinsic::Math(m) = intrinsic else {
            panic!("{ty}: not a math function")
        };
        assert_eq!(m.f, MathFn::Frexp, "{ty}");
    }
    // The operand form is still checked: the exponent pointer is a thread pointer, and no
    // other arity binds.
    for decl in [
        "v4f32 (v4f32)",
        "v4f32 (v4f32, p1)",
        "v4f32 (v4f32, p0, i32)",
    ] {
        assert_eq!(
            declared("air.frexp.v4f32", decl),
            DeclClass::Mismatch,
            "{decl}"
        );
    }
    // An integer frexp is no name of the family at all, so it stays an unknown intrinsic rather
    // than a mismatched one.
    assert_eq!(
        declared("air.frexp.v4i32", "v4i32 (v4i32, p0)"),
        DeclClass::Unknown
    );
    // `air.fast_frexp.<T>` reads as the same operation at `Precision::Fast`, as every float stem
    // does. MSL has no `fast::frexp` for an emitter to write, and a frexp has no precision to
    // trade: its mantissa and exponent are the operand's own bits.
    let AirIntrinsic::Math(fast) = declared_air("air.fast_frexp.v4f32", "v4f32 (v4f32, p0)") else {
        panic!("not a math function")
    };
    assert_eq!((fast.f, fast.precision), (MathFn::Frexp, Precision::Fast));
}

/// Before AIR 2.0 (deployment target macOS 10.12 and earlier, iOS 10 and earlier) a texture
/// operation states no trailing access qualifier and a memory atomic no trailing volatile flag.
/// Each such declaration decodes to the operation its later form decodes to: a texture's
/// `access` is then `None`, an atomic's `volatile` is `false`, and the calls' other immediates
/// keep their meaning. The forms are told apart by the declaration's own type, and a 64-bit,
/// `float`, or thread atomic -- all later than AIR 2.0 -- has no such form.
#[test]
fn pre_air_2_0_states_no_access_qualifier_or_volatile_flag() {
    let mut c = Consts::new();
    let ptr = Value::Arg(crate::ir::ArgIdx::new(0));
    let val = Value::Arg(crate::ir::ArgIdx::new(1));
    let coord = Value::Arg(crate::ir::ArgIdx::new(2));
    let smp = Value::Arg(crate::ir::ArgIdx::new(3));
    let (relaxed, device) = (c.i32(0), c.i32(2));
    let (yes, no) = (c.bool(true), c.bool(false));
    let (lod, read_access) = (c.i32(0), c.i32(1));
    let bias = Value::Arg(crate::ir::ArgIdx::new(4));

    // Textures: (name, the AIR 1.x declaration, the AIR 2.0 one, the 2.0 call, the 1.x call).
    let textures: [(&str, &str, &str, Vec<Value>, Vec<Value>); 3] = [
        (
            "air.read_texture_2d.v4f32",
            "v4f32 (p1, v2i32, i32)",
            "v4f32 (p1, v2i32, i32, i32)",
            vec![ptr, coord, lod, read_access],
            vec![ptr, coord, lod],
        ),
        (
            "air.write_texture_2d.v4f32",
            "void (p1, v2i32, v4f32, i32)",
            "void (p1, v2i32, v4f32, i32, i32)",
            vec![ptr, coord, val, lod, read_access],
            vec![ptr, coord, val, lod],
        ),
        (
            "air.sample_texture_2d.v4f32",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32)",
            "v4f32 (p1, p2, v2f32, i1, v2i32, i1, f32, i32)",
            vec![ptr, smp, coord, no, coord, yes, bias, read_access],
            vec![ptr, smp, coord, no, coord, yes, bias],
        ),
    ];
    for (name, old_decl, new_decl, new_args, old_args) in textures {
        let (AirIntrinsic::Texture(o), AirIntrinsic::Texture(n)) =
            (declared_air(name, old_decl), declared_air(name, new_decl))
        else {
            panic!("{name}");
        };
        assert_eq!((o.states_access, n.states_access), (false, true), "{name}");
        assert_eq!(
            Texture {
                states_access: true,
                ..o
            },
            n,
            "{name}"
        );
        let (old, new) = (declared(name, old_decl), declared(name, new_decl));
        let (Ok(Bound::Air(AirIntrinsic::Texture(a))), Ok(Bound::Air(AirIntrinsic::Texture(b)))) = (
            bind(old, &old_args, &c.table),
            bind(new, &new_args, &c.table),
        ) else {
            panic!("{name}");
        };
        // The same call in both forms: the qualifier the later one reads is the only difference.
        assert_eq!((a.access, b.access), (None, Some(Access::Read)), "{name}");
        assert_eq!(
            Texture {
                access: None,
                states_access: false,
                ..b
            },
            a,
            "{name}"
        );
    }

    // Atomics: (name, the AIR 1.x declaration, the AIR 2.0 one, the 2.0 call, the 1.x call).
    let atomics: [(&str, &str, &str, Vec<Value>, Vec<Value>); 3] = [
        (
            "air.atomic.global.add.u.i32",
            "i32 (p1, i32, i32, i32)",
            "i32 (p1, i32, i32, i32, i1)",
            vec![ptr, val, relaxed, device, yes],
            vec![ptr, val, relaxed, device],
        ),
        (
            "air.atomic.local.load.i32",
            "i32 (p3, i32, i32)",
            "i32 (p3, i32, i32, i1)",
            vec![ptr, relaxed, device, yes],
            vec![ptr, relaxed, device],
        ),
        (
            "air.atomic.local.store.i32",
            "void (p3, i32, i32, i32)",
            "void (p3, i32, i32, i32, i1)",
            vec![ptr, val, relaxed, device, yes],
            vec![ptr, val, relaxed, device],
        ),
    ];
    for (name, old_decl, new_decl, new_args, old_args) in atomics {
        let (AirIntrinsic::Atomic(o), AirIntrinsic::Atomic(n)) =
            (declared_air(name, old_decl), declared_air(name, new_decl))
        else {
            panic!("{name}");
        };
        assert_eq!(
            (o.stated_volatile, n.stated_volatile),
            (false, true),
            "{name}"
        );
        assert_eq!(
            Atomic {
                stated_volatile: true,
                ..o
            },
            n,
            "{name}"
        );
        let (old, new) = (declared(name, old_decl), declared(name, new_decl));
        let (Ok(Bound::Air(AirIntrinsic::Atomic(a))), Ok(Bound::Air(AirIntrinsic::Atomic(b)))) = (
            bind(old, &old_args, &c.table),
            bind(new, &new_args, &c.table),
        ) else {
            panic!("{name}");
        };
        // The volatile flag the later form reads is the only difference, and it orders nothing.
        assert_eq!((a.volatile, b.volatile), (false, true), "{name}");
        assert_eq!(
            Atomic {
                volatile: false,
                stated_volatile: false,
                ..b
            },
            a,
            "{name}"
        );
    }

    // No such form for a 64-bit, `float`, or thread atomic, nor for an operation whose layout
    // never stated a qualifier: their language versions are all later than AIR 2.0.
    for (name, decl) in [
        ("air.atomic.global.add.u.i64", "i64 (p1, i64, i32, i32)"),
        ("air.atomic.global.add.f32", "f32 (p1, f32, i32, i32)"),
        ("air.atomic.private.add.u.i32", "i32 (p0, i32, i32, i32)"),
        ("air.fence_texture_2d", "void ()"),
    ] {
        assert_eq!(declared(name, decl), DeclClass::Mismatch, "{name} {decl}");
    }
}

/// AIR 2.9 (deployment target macOS 27) inserts each memory atomic's `mem_flags` between its scope
/// and its volatile flag: the form is read from the declaration's type, the flags are read at their
/// position, the other immediates keep their meaning, and any other arity still mismatches.
#[test]
fn air_2_9_atomics_read_their_flags() {
    let mut c = Consts::new();
    let ptr = Value::Arg(crate::ir::ArgIdx::new(0));
    let val = Value::Arg(crate::ir::ArgIdx::new(1));
    let (relaxed, acquire, seq_cst) = (c.i32(0), c.i32(2), c.i32(5));
    let (device, tg) = (c.i32(2), c.i32(1));
    let (yes, no) = (c.bool(true), c.bool(false));
    let all = MemFlags {
        device: true,
        threadgroup: true,
        texture: true,
        threadgroup_imageblock: true,
        object_data: true,
    };
    let dev = MemFlags {
        device: true,
        ..MemFlags::default()
    };
    // (name, AIR 2.8 declaration, AIR 2.9 declaration, the 2.9 call's operands, and the 2.8
    // call's: the same without the flags).
    let flag_31 = c.i32(31);
    let flag_1 = c.i32(1);
    let cases: [(&str, &str, &str, Vec<Value>, Vec<Value>, MemFlags); 5] = [
        (
            "air.atomic.global.add.u.i32",
            "i32 (p1, i32, i32, i32, i1)",
            "i32 (p1, i32, i32, i32, i32, i1)",
            vec![ptr, val, seq_cst, device, flag_31, no],
            vec![ptr, val, seq_cst, device, no],
            all,
        ),
        (
            "air.atomic.global.load.i32",
            "i32 (p1, i32, i32, i1)",
            "i32 (p1, i32, i32, i32, i1)",
            vec![ptr, acquire, device, flag_1, yes],
            vec![ptr, acquire, device, yes],
            dev,
        ),
        (
            "air.atomic.local.store.i32",
            "void (p3, i32, i32, i32, i1)",
            "void (p3, i32, i32, i32, i32, i1)",
            vec![ptr, val, relaxed, tg, flag_1, yes],
            vec![ptr, val, relaxed, tg, yes],
            dev,
        ),
        (
            "air.atomic.global.cmpxchg.weak.i32",
            "i32 (p1, p0, i32, i32, i32, i32, i1)",
            "i32 (p1, p0, i32, i32, i32, i32, i32, i1)",
            vec![ptr, ptr, val, seq_cst, acquire, device, flag_31, no],
            vec![ptr, ptr, val, seq_cst, acquire, device, no],
            all,
        ),
        (
            "air.atomic.global.max.u.i64",
            "void (p1, i64, i32, i32, i1)",
            "void (p1, i64, i32, i32, i32, i1)",
            vec![ptr, val, relaxed, device, flag_1, yes],
            vec![ptr, val, relaxed, device, yes],
            dev,
        ),
    ];
    for (name, old_decl, new_decl, new_args, old_args, flags) in cases {
        let (old, new) = (declared(name, old_decl), declared(name, new_decl));
        let (DeclClass::Air(AirIntrinsic::Atomic(o)), DeclClass::Air(AirIntrinsic::Atomic(n))) =
            (old, new)
        else {
            panic!("{name}: {old:?} {new:?}");
        };
        assert_eq!(
            (o.flags, n.flags),
            (None, Some(MemFlags::default())),
            "{name}"
        );
        assert_eq!(Atomic { flags: None, ..n }, o, "{name}");
        let (Ok(Bound::Air(AirIntrinsic::Atomic(a))), Ok(Bound::Air(AirIntrinsic::Atomic(b)))) = (
            bind(old, &old_args, &c.table),
            bind(new, &new_args, &c.table),
        ) else {
            panic!("{name}");
        };
        // The same call in both forms: only the flags differ.
        assert_eq!(b.flags, Some(flags), "{name}");
        assert_eq!(Atomic { flags: None, ..b }, a, "{name}");
        // A flag bit outside Metal's five stays unknown; a non-constant one is malformed.
        let mut args = new_args.clone();
        let at = args.len() - 2;
        args[at] = c.i32(32);
        assert_eq!(bind(new, &args, &c.table), Ok(Bound::Unknown), "{name}");
        args[at] = val;
        assert_eq!(
            bind(new, &args, &c.table),
            Err(IntrinsicError::ImmediateNotConstant),
            "{name}"
        );
        // One operand more or fewer than either form is malformed.
        let (head, _) = new_decl.rsplit_once(", i1)").unwrap();
        let extra = format!("{head}, i32, i1)");
        assert_eq!(declared(name, &extra), DeclClass::Mismatch, "{name}");
    }
    // The result-less 64-bit min has both forms too, and only it: the 2.9 form of another void
    // atomic still mismatches.
    let AirIntrinsic::Atomic(a) = declared_air(
        "air.atomic.global.min.u.i64",
        "void (p1, i64, i32, i32, i32, i1)",
    ) else {
        panic!("min.u.i64");
    };
    assert_eq!((a.fetch, a.flags), (false, Some(MemFlags::default())));
    assert_eq!(
        declared(
            "air.atomic.global.add.u.i64",
            "void (p1, i64, i32, i32, i32, i1)"
        ),
        DeclClass::Mismatch
    );
}

/// MSL 4.1's atomics on thread memory (`air.atomic.private.*`) exist in the AIR 2.9 layout alone,
/// on a thread pointer, for the 32-bit integer and `float` operations Metal's headers declare;
/// their immediates bind as the other memories' do.
#[test]
fn thread_atomics_have_the_air_2_9_form_alone() {
    let AirIntrinsic::Atomic(a) = declared_air(
        "air.atomic.private.add.u.i32",
        "i32 (p0, i32, i32, i32, i32, i1)",
    ) else {
        panic!("add.u.i32");
    };
    assert_eq!(
        (a.space, a.flags, a.space.space()),
        (
            AtomicSpace::Private,
            Some(MemFlags::default()),
            AddrSpace::Thread
        )
    );
    // No AIR 2.8 form, nor another memory's pointer.
    for (name, decl) in [
        (
            "air.atomic.private.add.u.i32",
            "i32 (p0, i32, i32, i32, i1)",
        ),
        ("air.atomic.private.load.i32", "i32 (p0, i32, i32, i1)"),
        (
            "air.atomic.private.add.u.i32",
            "i32 (p1, i32, i32, i32, i32, i1)",
        ),
    ] {
        assert_eq!(declared(name, decl), DeclClass::Mismatch, "{name} {decl}");
    }
    // Nor the operations Metal's headers have no thread atomic of: 64-bit integers, a `float`
    // minimum, maximum, or compare-exchange.
    for (name, decl) in [
        (
            "air.atomic.private.max.u.i64",
            "i64 (p0, i64, i32, i32, i32, i1)",
        ),
        (
            "air.atomic.private.min.f32",
            "f32 (p0, f32, i32, i32, i32, i1)",
        ),
        (
            "air.atomic.private.max.f32",
            "f32 (p0, f32, i32, i32, i32, i1)",
        ),
        (
            "air.atomic.private.cmpxchg.weak.f32",
            "f32 (p0, p0, f32, i32, i32, i32, i32, i1)",
        ),
    ] {
        assert_eq!(declared(name, decl), DeclClass::Unknown, "{name}");
    }
    // A call binds its order, scope, flags, and volatile flag.
    let mut c = Consts::new();
    let ptr = Value::Arg(crate::ir::ArgIdx::new(0));
    let val = Value::Arg(crate::ir::ArgIdx::new(1));
    let new = declared(
        "air.atomic.private.cmpxchg.weak.i32",
        "i32 (p0, p0, i32, i32, i32, i32, i32, i1)",
    );
    let args = [
        ptr,
        ptr,
        val,
        c.i32(4),
        c.i32(2),
        c.i32(0),
        c.i32(1),
        c.bool(false),
    ];
    let Ok(Bound::Air(AirIntrinsic::Atomic(b))) = bind(new, &args, &c.table) else {
        panic!("bind");
    };
    assert_eq!(
        (b.order, b.failure, b.scope, b.flags, b.volatile),
        (
            MemoryOrder::AcqRel,
            Some(MemoryOrder::Acquire),
            ThreadScope::Thread,
            Some(MemFlags {
                device: true,
                ..MemFlags::default()
            }),
            false
        )
    );
}

/// AIR 2.9 (deployment target macOS 27) declares both barriers `void (i32 flags, i32 order, i32
/// scope)`: the form is read from the declaration's type, the order is the operand (MSL 4.1's
/// `relaxed`, `acquire`, `release`, `acq_rel`, or the `seq_cst` every older barrier has), the scope
/// operand is the memory scope alone and the name the execution scope, and any other arity still
/// mismatches.
#[test]
fn air_2_9_barriers_read_their_order() {
    for (name, barrier, own) in [
        (
            "air.wg.barrier",
            BarrierName::Threadgroup,
            ThreadScope::Threadgroup,
        ),
        (
            "air.simdgroup.barrier",
            BarrierName::Simdgroup,
            ThreadScope::Simdgroup,
        ),
    ] {
        let old = declared(name, "void (i32, i32)");
        let new = declared(name, "void (i32, i32, i32)");
        let DeclClass::Air(AirIntrinsic::Sync(s)) = new else {
            panic!("{name}: {new:?}");
        };
        assert!(matches!(s, Sync::Barrier { order: Some(_), .. }), "{name}");
        assert!(matches!(
            old,
            DeclClass::Air(AirIntrinsic::Sync(Sync::Barrier { order: None, .. }))
        ));
        for decl in [
            "void (i32)",
            "void (i32, i32, i32, i32)",
            "i32 (i32, i32, i32)",
        ] {
            assert_eq!(declared(name, decl), DeclClass::Mismatch, "{name} {decl}");
        }
        let mut c = Consts::new();
        for (raw, order) in [
            (0, MemoryOrder::Relaxed),
            (2, MemoryOrder::Acquire),
            (3, MemoryOrder::Release),
            (4, MemoryOrder::AcqRel),
            (5, MemoryOrder::SeqCst),
        ] {
            let args = [c.i32(2), c.i32(raw), c.i32(1)];
            let Ok(Bound::Air(AirIntrinsic::Sync(s))) = bind(new, &args, &c.table) else {
                panic!("{name} {raw}");
            };
            assert_eq!(
                s,
                Sync::Barrier {
                    name: barrier,
                    flags: MemFlags {
                        threadgroup: true,
                        ..MemFlags::default()
                    },
                    order: Some(order),
                    scope: own,
                    memory: ThreadScope::Threadgroup,
                },
                "{name} {raw}"
            );
            assert_eq!(s.barrier_order(), Some(order));
        }
        // Every `thread_scope` is a memory scope (MSL 4.1 `threadgroup_barrier(flags, order,
        // thread_scope_device)` is `air.wg.barrier(flags, order, 2)`); the execution scope stays
        // the name's.
        for (raw, memory) in [
            (0, ThreadScope::Thread),
            (1, ThreadScope::Threadgroup),
            (2, ThreadScope::Device),
            (3, ThreadScope::AllDevices),
            (4, ThreadScope::Simdgroup),
        ] {
            let args = [c.i32(1), c.i32(5), c.i32(raw)];
            let Ok(Bound::Air(AirIntrinsic::Sync(s))) = bind(new, &args, &c.table) else {
                panic!("{name} scope {raw}");
            };
            assert!(
                matches!(s, Sync::Barrier { scope, memory: m, .. } if scope == own && m == memory),
                "{name} scope {raw}: {s:?}"
            );
        }
        // The same barrier in both forms: the 2.8 form's order is `seq_cst`.
        let (Ok(Bound::Air(AirIntrinsic::Sync(a))), Ok(Bound::Air(AirIntrinsic::Sync(b)))) = (
            bind(old, &[c.i32(2), c.i32(4)], &c.table),
            bind(new, &[c.i32(2), c.i32(5), c.i32(4)], &c.table),
        ) else {
            panic!("{name}");
        };
        assert_eq!(a.barrier_order(), b.barrier_order());
        // `consume` (1) and values past `seq_cst` have no Metal spelling, nor a scope past the
        // simdgroup's or a flag past `mem_object_data`.
        for args in [
            [c.i32(2), c.i32(1), c.i32(1)],
            [c.i32(2), c.i32(6), c.i32(1)],
            [c.i32(2), c.i32(5), c.i32(5)],
            [c.i32(64), c.i32(5), c.i32(1)],
        ] {
            assert_eq!(bind(new, &args, &c.table), Ok(Bound::Unknown), "{name}");
        }
        let v = Value::Arg(crate::ir::ArgIdx::new(0));
        assert_eq!(
            bind(new, &[c.i32(2), v, c.i32(1)], &c.table),
            Err(IntrinsicError::ImmediateNotConstant)
        );
    }
}

#[test]
fn handles_are_typed_seeds() {
    let shape = ImageShape {
        dim: ImageDim::D2,
        array: true,
        ms: false,
        depth: true,
    };
    let sig = air("air.sample_depth_2d_array.f32").signature();
    assert_eq!(
        sig.handles().collect::<Vec<_>>(),
        vec![(0, HandleKind::Texture(shape)), (1, HandleKind::Sampler)]
    );
    let null = air("air.get_null_texture_3d").signature();
    assert!(matches!(
        null.ret_handle(),
        Some(HandleKind::Texture(ImageShape {
            dim: ImageDim::D3,
            ..
        }))
    ));
    for (name, want) in [
        ("air.get_read_sampler", HandleKind::Sampler),
        ("air.get_null_simdgroup_event", HandleKind::Event),
        (
            "air.get_null_intersection_function_table",
            HandleKind::IntersectionFunctionTable,
        ),
    ] {
        assert_eq!(air(name).signature().ret_handle(), Some(want), "{name}");
    }
    let vft = air("air.get_function_pointer_visible_function_table").signature();
    assert_eq!(
        vft.handles().collect::<Vec<_>>(),
        vec![(0, HandleKind::VisibleFunctionTable)]
    );
    let cmd = air("air.set_kernel_buffer_compute_command.p1i8").signature();
    assert_eq!(
        cmd.handles().collect::<Vec<_>>(),
        vec![(0, HandleKind::Command)]
    );
    // The pipeline a compute command is set to is a pipeline-state handle, not bytes.
    let pso = air("air.set_pipeline_state_compute_command").signature();
    assert_eq!(
        pso.handles().collect::<Vec<_>>(),
        vec![
            (0, HandleKind::Command),
            (2, HandleKind::ComputePipelineState)
        ]
    );
    let render = air("air.set_pipeline_state_render_command").signature();
    assert_eq!(
        render.handles().collect::<Vec<_>>(),
        vec![
            (0, HandleKind::Command),
            (2, HandleKind::RenderPipelineState)
        ]
    );
    // An indexed draw's overload names its index buffer's space and index type.
    for (name, space, wide) in [
        ("p1i16", AddrSpace::Device, false),
        ("p1i32", AddrSpace::Device, true),
        ("p2i32", AddrSpace::Constant, true),
    ] {
        let name = format!("air.draw_indexed_primitives_render_command.{name}");
        assert_eq!(
            air(&name),
            AirIntrinsic::IndirectCommand(IndirectCommand::DrawIndexedPrimitives { space, wide }),
            "{name}"
        );
    }
    // A patch draw's overload names its patch index buffer's space and its factor buffer's.
    for (name, patch_indices, factors) in [
        ("p1i32.p1i8", AddrSpace::Device, AddrSpace::Device),
        ("p2i32.p2i8", AddrSpace::Constant, AddrSpace::Constant),
        ("p1i32.p2i8", AddrSpace::Device, AddrSpace::Constant),
    ] {
        let name = format!("air.draw_patches_render_command.{name}");
        assert_eq!(
            air(&name),
            AirIntrinsic::IndirectCommand(IndirectCommand::DrawPatches {
                patch_indices,
                factors
            }),
            "{name}"
        );
    }
    // An indexed patch draw's overload names its patch index buffer's space, its control-point
    // index buffer's and its factor buffer's; its operands are a patch draw's with the
    // control-point index buffer after the patch indices.
    for (name, patch_indices, control_point_indices, factors) in [
        (
            "p1i32.p1i8.p1i8",
            AddrSpace::Device,
            AddrSpace::Device,
            AddrSpace::Device,
        ),
        (
            "p1i32.p2i8.p1i8",
            AddrSpace::Device,
            AddrSpace::Constant,
            AddrSpace::Device,
        ),
        (
            "p2i32.p1i8.p2i8",
            AddrSpace::Constant,
            AddrSpace::Device,
            AddrSpace::Constant,
        ),
    ] {
        let name = format!("air.draw_indexed_patches_render_command.{name}");
        let op = air(&name);
        assert_eq!(
            op,
            AirIntrinsic::IndirectCommand(IndirectCommand::DrawIndexedPatches {
                patch_indices,
                control_point_indices,
                factors
            }),
            "{name}"
        );
        let i32s = Operand::Num(NumTy::scalar(Elem::I32));
        assert_eq!(
            op.signature(),
            Signature::new(
                Operand::Void,
                &[
                    Operand::Handle(HandleKind::Command),
                    i32s,
                    i32s,
                    i32s,
                    i32s,
                    Operand::Ptr(patch_indices),
                    Operand::Ptr(control_point_indices),
                    i32s,
                    i32s,
                    Operand::Ptr(factors),
                    i32s,
                    Operand::Num(NumTy::scalar(Elem::F32)),
                ]
            ),
            "{name}"
        );
    }
    // A render command's state setters, as Metal's compiler declares them: the command, its index,
    // then the state's values.
    let i32s = Operand::Num(NumTy::scalar(Elem::I32));
    let f32s = Operand::Num(NumTy::scalar(Elem::F32));
    let cmd = Operand::Handle(HandleKind::Command);
    for (name, state, values) in [
        ("cull_mode", RenderState::CullMode, vec![i32s]),
        ("depth_bias", RenderState::DepthBias, vec![f32s, f32s, f32s]),
        ("depth_clip_mode", RenderState::DepthClipMode, vec![i32s]),
        (
            "front_facing_winding",
            RenderState::FrontFacingWinding,
            vec![i32s],
        ),
        (
            "triangle_fill_mode",
            RenderState::TriangleFillMode,
            vec![i32s],
        ),
        (
            "depth_stencil_state",
            RenderState::DepthStencilState,
            vec![Operand::Handle(HandleKind::DepthStencilState)],
        ),
    ] {
        let name = format!("air.set_{name}_render_command");
        let op = air(&name);
        assert_eq!(
            op,
            AirIntrinsic::IndirectCommand(IndirectCommand::SetRenderState(state)),
            "{name}"
        );
        let mut operands = vec![cmd, i32s];
        operands.extend(values);
        assert_eq!(
            op.signature(),
            Signature::new(Operand::Void, &operands),
            "{name}"
        );
    }
    // A render command's barrier, mesh draws, object threadgroup length, and object and mesh
    // buffer setters (the fixture `mesh_commands`, a Metal probe); the buffer setters' overloads
    // name the buffer's space, device or constant.
    for (name, op) in [
        (
            "air.set_barrier_render_command",
            IndirectCommand::RenderBarrier { enabled: true },
        ),
        (
            "air.clear_barrier_render_command",
            IndirectCommand::RenderBarrier { enabled: false },
        ),
        (
            "air.draw_mesh_threadgroups_render_command",
            IndirectCommand::DrawMesh { threads: false },
        ),
        (
            "air.draw_mesh_threads_render_command",
            IndirectCommand::DrawMesh { threads: true },
        ),
        (
            "air.set_object_threadgroup_memory_length_render_command",
            IndirectCommand::SetObjectThreadgroupMemoryLength,
        ),
        (
            "air.set_object_buffer_render_command.p2i8",
            IndirectCommand::SetBuffer {
                stage: CommandStage::Object,
                space: AddrSpace::Constant,
                strided: false,
            },
        ),
        (
            "air.set_mesh_buffer_render_command.p1i8",
            IndirectCommand::SetBuffer {
                stage: CommandStage::Mesh,
                space: AddrSpace::Device,
                strided: false,
            },
        ),
    ] {
        assert_eq!(air(name), AirIntrinsic::IndirectCommand(op), "{name}");
    }
    for name in [
        "air.set_mesh_buffer_render_command.p3i8",
        "air.set_object_buffer_render_command",
        "air.draw_mesh_threads_render_command.i32",
    ] {
        assert!(AirIntrinsic::parse(name).is_none(), "{name}");
    }
    for name in [
        "air.draw_indexed_primitives_render_command.p1i8",
        "air.draw_indexed_primitives_render_command.p3i32",
        "air.draw_patches_render_command.p1i16.p1i8",
        "air.draw_patches_render_command.p1i32",
        "air.draw_patches_render_command.p1i32.p3i8",
        "air.draw_indexed_patches_render_command.p1i32.p1i8",
        "air.draw_indexed_patches_render_command.p1i16.p1i8.p1i8",
        "air.draw_indexed_patches_render_command.p1i32.p3i8.p1i8",
        "air.draw_indexed_patches_render_command.p1i32.p1i8.p3i8",
        "air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8.p1i8",
    ] {
        assert!(AirIntrinsic::parse(name).is_none(), "{name}");
    }
    // Values that are not handles are not seeds.
    for name in ["air.imageblock_data", "air.simd_sum.f32", "air.dfdx.f32"] {
        let sig = air(name).signature();
        assert_eq!(
            (sig.handles().count(), sig.ret_handle()),
            (0, None),
            "{name}"
        );
    }
}

#[test]
fn layouts_poison_on_overflow() {
    let mut l: Layout<u8> = Layout::new(Operand::Void);
    let i32s = Operand::Num(NumTy::scalar(Elem::I32));
    let max = super::sig::MAX_OPERANDS;
    for i in 0..=max {
        l.push(u8::try_from(i).unwrap(), i32s);
    }
    assert_eq!(l.positions().count(), max);
    let sig = l.signature();
    let mut t = TypeTable::new();
    let full = Signature::new(Operand::Void, &vec![i32s; max]);
    let ty = fn_ty(&mut t, &full);
    assert!(!sig.matches(&t, ty));
}

/// Every name below, one or more of each known AIR family, decodes here or its family is listed
/// in [`UNDECODED`], never both; and every listed family has a name below.
#[test]
fn known_families_decode_or_are_listed() {
    let names = [
        "air.simdgroup.barrier",
        "air.wg.barrier",
        "air.atomic.fence",
        "air.simd_any",
        "air.simd_all",
        "air.simd_ballot.i64",
        "air.quad_all",
        "air.quad_any",
        "air.quad_active_threads_mask",
        "air.quad_is_first",
        "air.simd_is_first",
        "air.is_function_constant_defined",
        "air.get_num_samples.i32",
        "air.get_imageblock_width",
        "air.get_imageblock_height",
        "air.get_read_sampler",
        "air.get_instance_count_instance_acceleration_structure",
        "air.get_primitive_acceleration_structure_instance_acceleration_structure",
        "air.get_data_pointer_instance_acceleration_structure",
        "air.imageblock_data",
        "air.get_descriptor_size_tensor",
        "air.get_null_intersection_function_table",
        "air.get_size_visible_function_table",
        "air.get_function_pointer_visible_function_table",
        "air.is_null_visible_function_table",
        "air.get_extent_private_tensor.i32",
        "air.init_strided_private_tensor.i32.global",
        "air.init_strided_private_tensor.i32.local",
        "air.slice_private_tensor_private_tensor.s.i32",
        "air.set_buffer_intersection_function_table.p1i8",
        "air.set_intersection_function_table.p1i8",
        "air.intersect.instancing.triangle_data",
        "air.calculate_unclamped_lod_texture_2d",
        "air.calculate_clamped_lod_texture_2d",
        "air.fence_texture_2d",
        "air.sample_texture_2d.v4f32",
        "air.sample_depth_2d.f32",
        "air.sample_compare_depth_2d.f32",
        "air.gather_texture_2d.v4f32",
        "air.gather_depth_2d.v4f32",
        "air.read_texture_2d.v4f32",
        "air.read_depth_2d.f32",
        "air.write_texture_2d.v4f32",
        "air.atomic_fetch_max_explicit_texture_2d.i16.u.v4i32",
        "air.write_imageblock_slice_to_texture_2d.v4f32",
        "air.get_width_texture_2d",
        "air.get_height_texture_2d",
        "air.get_depth_texture_3d",
        "air.get_array_size_texture_2d_array",
        "air.get_width_depth_2d",
        "air.get_height_depth_2d",
        "air.get_num_mip_levels_texture_2d",
        "air.get_num_mip_levels_depth_2d",
        "air.get_num_samples_texture_2d_ms",
        "air.is_null_texture_2d",
        "air.get_null_texture_2d",
        "air.load.implicit_imageblock.v4f16",
        "air.store.implicit_imageblock.v4f16",
        "air.discard_fragment",
        "air.map_screen_to_physical_coordinates.v2f32.p2i8.i32",
        "air.map_physical_to_screen_coordinates.v2f32.p2i8.i32",
        "air.map_screen_to_physical_coordinates.v2i32.p2i8.i32",
        "air.map_physical_to_screen_coordinates.v2i32.p2i8.i32",
        "air.simdgroup_matrix_8x8_load.v64f32.p1f32",
        "air.simdgroup_matrix_8x8_store.v64f32.p3f32",
        "air.simdgroup_matrix_8x8_init_diag.v64f32.f32",
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64bf16.v64bf16.v64f32",
        "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32",
        "air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.s.v8i32.v8i8.v8i8.v8i32",
        "air.simdgroup_async_copy_2d.p3i8.p1i8",
        "air.get_null_simdgroup_event",
        "air.is_null_simdgroup_event",
        "air.wait_simdgroup_events",
        "air.function_constant_predicate",
        "air.normalize_function_constant_predicate.i8",
        "air.concurrent_dispatch_threadgroups_compute_command",
        "air.concurrent_dispatch_threads_compute_command",
        "air.set_pipeline_state_compute_command",
        "air.set_pipeline_state_render_command",
        "air.set_barrier_compute_command",
        "air.clear_barrier_compute_command",
        "air.set_barrier_render_command",
        "air.clear_barrier_render_command",
        "air.set_threadgroup_memory_length_compute_command",
        "air.draw_primitives_render_command",
        "air.set_kernel_buffer_compute_command.p1i8",
        "air.set_vertex_buffer_render_command.p2i8",
        "air.set_fragment_buffer_render_command.p1i8",
        "air.set_object_buffer_render_command.p1i8",
        "air.dfdx.f32",
        "air.fast_dfdx.v2f32",
        "air.dfdy.f16",
        "air.fast_dfdy.f32",
        "air.fwidth.v3f32",
        "air.is_uniform.i32",
        "air.get_simdgroup_size.i32",
        "air.simd_broadcast_first.f32",
        "air.simd_broadcast.s.i16",
        "air.simd_shuffle.u.i32",
        "air.simd_shuffle_down.f32",
        "air.simd_shuffle_rotate_down.f32",
        "air.simd_shuffle_and_fill_down.v4f16",
        "air.simd_shuffle_and_fill_up.u.i32",
        "air.simd_shuffle_up.v3f32",
        "air.simd_shuffle_xor.s.v2i16",
        "air.quad_shuffle.v4f32",
        "air.quad_broadcast.u.i32",
        "air.quad_sum.v2f32",
        "air.quad_min.s.i32",
        "air.quad_max.u.i32",
        "air.quad_shuffle_xor.f16",
        "air.quad_shuffle_rotate_down.f32",
        "air.quad_shuffle_up.f32",
        "air.quad_shuffle_down.v3f32",
        "air.simd_prefix_exclusive_sum.u.i16",
        "air.simd_prefix_inclusive_sum.f32",
        "air.simd_sum.u.v2i32",
        "air.simd_or.u.i8",
        "air.simd_xor.u.v2i16",
        "air.simd_and.s.i32",
        "air.simd_min.f16",
        "air.simd_max.s.v2i16",
        "air.all.v3i1",
        "air.any.v2i1",
        // MPS's 8-bit float tensor writes: listed, not decoded.
        "air.convert.f.f8e4m3.f.f32",
        "air.convert.f.f8e4m3fn.s.i16",
        "air.convert.f.f8e5m2.u.i64",
        // Its reads decode (an 8-bit float to a float or an integer).
        "air.convert.f.f32.f.f8e4m3",
        "air.convert.s.i32.f.f8e4m3fn",
        "air.convert.u.i8.f.f8e5m2",
        // A multisample imageblock's operations: listed, not decoded (the single-sample ones
        // above, its size and the implicit loads and stores, decode).
        "air.get_imageblock_samples",
        "air.get_imageblock_num_colors",
        "air.get_color_coverage_mask",
        "air.store.implicit_imageblock.mask.v4f16",
    ];
    for name in names {
        let listed = undecoded_family(name).is_some();
        let decodes = matches!(classify_self(name), DeclClass::Air(_));
        assert!(
            decodes != listed,
            "{name}: decodes={decodes} listed={listed}"
        );
    }
    // Every listed family has a name above.
    for &(prefix, reason) in UNDECODED {
        assert!(!reason.is_empty());
        assert!(
            names.iter().any(|n| n.starts_with(prefix)),
            "{prefix} has no name in the list"
        );
    }
}

/// The typed undecoded families are undecoded `air.*` families, classified by their ABI prefix.
/// Binding an intersection function table is a function table (dynamic call), and a multisample
/// imageblock's operations are imageblocks. Decoded siblings of these families (and ray
/// intersection, decoded, the single-sample imageblock operations, and the render commands'
/// barriers and object and mesh setters, decoded) and other undecoded names are not classified
/// here.
#[test]
fn undecoded_features_are_undecoded_prefixes() {
    for (prefix, _) in UNDECODED_FEATURES {
        assert!(
            UNDECODED.iter().any(|(listed, _)| listed == prefix),
            "{prefix} is not an undecoded family"
        );
    }
    // Binding an intersection function table is that feature.
    let name = "air.set_intersection_function_table.p1i8";
    assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
    assert_eq!(
        undecoded_feature(name.as_bytes()),
        Some(Feature::IntersectionFunctionTable),
        "{name}"
    );
    // A multisample imageblock's operations, as Apple's compiler declares them, are imageblocks.
    for name in [
        "air.get_imageblock_samples",
        "air.get_imageblock_num_colors",
        "air.get_color_coverage_mask",
        "air.store.implicit_imageblock.mask.f32",
        "air.store.implicit_imageblock.mask.v4f16",
    ] {
        assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
        assert_eq!(
            undecoded_feature(name.as_bytes()),
            Some(Feature::Imageblocks),
            "{name}"
        );
    }
    // Decoded members of these families are refused by lowering, not classified here.
    for name in [
        "air.get_imageblock_width",
        "air.get_imageblock_height",
        "air.load.implicit_imageblock.f32",
        "air.store.implicit_imageblock.v4f16",
        "air.intersect",
        "air.intersect.instancing.triangle_data",
        "air.get_null_intersection_function_table",
        "air.get_data_pointer_instance_acceleration_structure",
        "air.set_buffer_intersection_function_table.p1i8",
        "air.set_barrier_compute_command",
        "air.clear_barrier_compute_command",
        "air.reset_compute_command",
        "air.copy_compute_command",
        "air.set_stage_in_region_compute_command",
        "air.set_imageblock_size_compute_command",
        "air.draw_primitives_render_command",
        // The render commands that decode.
        "air.set_pipeline_state_render_command",
        "air.reset_render_command",
        "air.draw_indexed_primitives_render_command.p1i16",
        "air.draw_indexed_primitives_render_command.p2i32",
        "air.draw_patches_render_command.p1i32.p1i8",
        "air.draw_patches_render_command.p2i32.p2i8",
        "air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8",
        "air.draw_indexed_patches_render_command.p1i32.p2i8.p1i8",
        "air.set_barrier_render_command",
        "air.clear_barrier_render_command",
        "air.set_object_buffer_render_command.p1i8",
        "air.set_mesh_buffer_render_command.p2i8",
        "air.draw_mesh_threadgroups_render_command",
        "air.draw_mesh_threads_render_command",
        "air.set_object_threadgroup_memory_length_render_command",
    ] {
        assert!(
            matches!(classify_self(name), DeclClass::Air(_)),
            "{name} decodes"
        );
        assert_eq!(undecoded_feature(name.as_bytes()), None, "{name}");
    }
    // An intersector name the grammar rejects, and every `intersection_query` operation, are ray
    // tracing by their family's first name component.
    for name in [
        "air.intersect.",
        "air.intersect.probe",
        "air.intersect.triangle_data.instancing",
        "air.intersect.instancing.some_future_tag",
        "air.next_intersection_query.instancing.triangle_data",
        "air.reset_intersection_query.instancing.triangle_data",
        "air.get_committed_distance_intersection_query",
    ] {
        assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
        assert_eq!(
            undecoded_feature(name.as_bytes()),
            Some(Feature::RayTracing),
            "{name}"
        );
    }
    // Other undecoded names, and near misses of the prefixes, stay unknown intrinsics.
    for name in [
        "air.function_constant_predicate",
        "air.intersection_query.foo",
        "air.intersector",
        "air.some_future_op",
    ] {
        assert_eq!(undecoded_feature(name.as_bytes()), None, "{name}");
    }
}

/// Object and mesh functions' operations as Apple's compiler declares them (the fixtures
/// `mesh_order` and `mesh_outputs`): the grid handle in `threadgroup` (3), the mesh output handle
/// in address space 7.
const MESH_OBSERVED: &str = "\
air.set_threadgroups_per_grid_mesh_properties void (p3, v3i32)
air.set_position_mesh void (p7, i32, v4f32)
air.set_point_size_mesh void (p7, i32, f32)
air.set_clip_distance_mesh void (p7, i32, i32, f32)
air.set_vertex_data_mesh.f32 void (p7, i32, i32, f32)
air.set_vertex_data_mesh.i32 void (p7, i32, i32, i32)
air.set_vertex_data_mesh.v2f16 void (p7, i32, i32, v2f16)
air.set_vertex_data_mesh.v3f32 void (p7, i32, i32, v3f32)
air.set_index_mesh void (p7, i32, i8)
air.set_indices_mesh.v2i8 void (p7, i32, v2i8)
air.set_indices_mesh.v4i8 void (p7, i32, v4i8)
air.set_primitive_data_mesh.f32 void (p7, i32, i32, f32)
air.set_primitive_data_mesh.i32 void (p7, i32, i32, i32)
air.set_primitive_data_mesh.v2f32 void (p7, i32, i32, v2f32)
air.set_primitive_data_mesh.v4f32 void (p7, i32, i32, v4f32)
air.set_primitive_culled_mesh void (p7, i32, i1)
air.set_primitive_id_mesh void (p7, i32, i32)
air.set_render_target_array_index_mesh.i32 void (p7, i32, i32)
air.set_viewport_array_index_mesh.i32 void (p7, i32, i32)
air.set_primitive_count_mesh void (p7, i32)
";

#[test]
fn mesh_declarations_decode() {
    for line in MESH_OBSERVED.lines() {
        let (name, decl) = line.split_once(' ').unwrap();
        let AirIntrinsic::Mesh(_) = declared_air(name, decl) else {
            panic!("{name}");
        };
    }
    assert_eq!(
        declared_air("air.set_indices_mesh.v4i8", "void (p7, i32, v4i8)"),
        AirIntrinsic::Mesh(Mesh::Indices(4))
    );
    // A known name declared any other way is malformed at its calls: the handle in another
    // address space, a missing slot operand, a value of another type.
    for (name, decl) in [
        ("air.set_position_mesh", "void (p1, i32, v4f32)"),
        (
            "air.set_threadgroups_per_grid_mesh_properties",
            "void (p7, v3i32)",
        ),
        ("air.set_vertex_data_mesh.f32", "void (p7, i32, f32)"),
        ("air.set_index_mesh", "void (p7, i32, i32)"),
        ("air.set_primitive_count_mesh", "void (p7, i64)"),
    ] {
        assert_eq!(declared(name, decl), DeclClass::Mismatch, "{name} {decl}");
    }
    // Shapes outside the grammar are no mesh operation at all.
    for name in [
        "air.set_indices_mesh.v3i8",
        "air.set_indices_mesh.v4i16",
        "air.set_vertex_data_mesh.f64",
        "air.set_vertex_data_mesh.v8f32",
        "air.set_viewport_array_index_mesh.f32",
        "air.set_primitive_count_mesh.i32",
    ] {
        assert!(AirIntrinsic::parse(name).is_none(), "{name}");
    }
}

/// Every global built-in variable's read decodes at `i16` and `i32`, a vector built-in's with its
/// lane operand and a scalar one's with none, as the built-in its name spells; another width, a
/// vector type, and `threads_per_simdgroup` (`get_simdgroup_size`) are not global built-in reads.
#[test]
fn global_builtin_reads_decode_as_their_builtins() {
    for &(var, name, vector) in &GlobalVar::ALL {
        for (ty, elem) in [("i16", Elem::I16), ("i32", Elem::I32)] {
            let full = format!("air.{name}.{ty}");
            let Some(AirIntrinsic::GlobalBuiltin(g)) = AirIntrinsic::parse(&full) else {
                panic!("{full} does not decode as a global built-in");
            };
            assert_eq!(g, GlobalBuiltin { var, ty: elem }, "{full}");
            assert_eq!(var.is_vector(), vector, "{full}");
            let operands = if vector { 1 } else { 0 };
            assert_eq!(g.signature().params().len(), operands, "{full}");
        }
        for bad in ["i64", "v3i32", "f32", ""] {
            let full = format!("air.{name}.{bad}");
            assert!(
                !matches!(
                    AirIntrinsic::parse(&full),
                    Some(AirIntrinsic::GlobalBuiltin(_))
                ),
                "{full}"
            );
        }
    }
    assert!(matches!(
        AirIntrinsic::parse("air.get_simdgroup_size.i32"),
        Some(AirIntrinsic::SimdgroupSize(_))
    ));
}

/// A block load's or store's name declared with AIR 2.7's operands decodes as the AIR 2.7 form,
/// whose AIR 2.8 form is the operation the same name declares with AIR 2.8's.
#[test]
fn legacy_block_forms_are_chosen_by_the_declaration() {
    use AddrSpace::{Device, Threadgroup};
    let cases = [
        (
            "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
            "v64f32 (p3, i64, v2i64, i1)",
            "v64f32 (p3, v2i64, v2i64, v2i64)",
            SimdMatrix::Load8x8Legacy {
                elem: MatElem::F32,
                space: Threadgroup,
            },
        ),
        (
            "air.simdgroup_matrix_8x8_store.v64f16.p1f16",
            "void (v64f16, p1, i64, v2i64, i1)",
            "void (v64f16, p1, v2i64, v2i64, v2i64)",
            SimdMatrix::Store8x8Legacy {
                elem: MatElem::F16,
                space: Device,
            },
        ),
    ];
    for (name, old, new, legacy) in cases {
        assert_eq!(
            declared_air(name, old),
            AirIntrinsic::SimdMatrix(legacy),
            "{name}"
        );
        let AirIntrinsic::SimdMatrix(modern) = declared_air(name, new) else {
            panic!("{name} {new}");
        };
        assert_eq!(legacy.modern(), Some(modern), "{name}");
        assert_eq!(modern.modern(), None, "{name}");
    }
    // A flag that is no `i1` or an extent that is no `i64` decodes as neither form.
    for decl in [
        "v64f32 (p3, i64, v2i64, i32)",
        "v64f32 (p3, i32, v2i64, i1)",
    ] {
        assert_eq!(
            declared("air.simdgroup_matrix_8x8_load.v64f32.p3f32", decl),
            DeclClass::Mismatch,
            "{decl}"
        );
    }
}

#[test]
fn interpolation_functions_read_their_location_and_interpolant() {
    let i = |name: &str, decl: &str| match declared_air(name, decl) {
        AirIntrinsic::Interpolate(i) => i,
        other => panic!("{name}: {other:?}"),
    };
    let f32x = |lanes| NumTy {
        elem: Elem::F32,
        lanes,
    };
    let offset = i(
        "air.interpolate_offset_no_perspective.v2f32",
        "v2f32 (p1, v2f32)",
    );
    assert_eq!(
        offset,
        Interpolate {
            at: InterpolateAt::Offset,
            perspective: false,
            ty: f32x(2),
        }
    );
    let sample = i(
        "air.interpolate_sample_perspective.v4f16",
        "v4f16 (p1, i32)",
    );
    assert_eq!(
        (sample.at, sample.perspective),
        (InterpolateAt::Sample, true)
    );
    let center = i("air.interpolate_center_perspective.f32", "f32 (p1)");
    assert_eq!(center.ty, f32x(1));
    // The interpolant is the one handle, a device-space pointer.
    for x in [offset, sample, center] {
        let handles: Vec<_> = x.signature().handles().collect();
        assert_eq!(handles, [(0, HandleKind::Interpolant)]);
        assert_eq!(HandleKind::Interpolant.space(), AddrSpace::Device);
    }
    // No other location, qualifier or value type is an interpolation function.
    for name in [
        "air.interpolate_center_flat.f32",
        "air.interpolate_pixel_perspective.f32",
        "air.interpolate_center_perspective.i32",
        "air.interpolate_center_perspective.v8f32",
        "air.interpolate_centerperspective.f32",
    ] {
        assert_eq!(AirIntrinsic::parse(name), None, "{name}");
    }
}
