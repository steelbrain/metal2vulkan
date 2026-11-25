//! The access and descriptor rules of the declared interface, held to fixed expectations.

use super::*;

/// A kernel whose device buffer `%buffer`, declared read-write, is used only by `use_line`.
fn kernel_using_buffer_only_in(attributes: &str, use_line: &str, declarations: &str) -> String {
    format!(
        r#"define void @k(ptr addrspace(1) {attributes} %buffer, ptr addrspace(3) %tile, ptr addrspace(2) %length) {{
entry:
{use_line}
  ret void
}}
{declarations}
!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buffer"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"tile"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"length"}}
"#
    )
}

/// The access the declared interface gives `%buffer`.
fn buffer_access(ll: &str) -> (Option<ResourceAccess>, String) {
    let ours = inspect(ll, Stage::Kernel, "k", &[]).unwrap();
    let access = ours.binding_at(ResourceKind::Buffer, 0).unwrap().access;
    (access, format!("{access:?}"))
}

/// A buffer whose only use is an asynchronous copy keeps the access its metadata declares, as it
/// did after the old engine lowered the copy: the copy still names the buffer, and its attributes
/// say nothing narrower.
#[test]
fn an_async_copy_as_a_buffers_only_use_keeps_its_declared_access() {
    let ll = kernel_using_buffer_only_in(
        "",
        "  %event = call ptr @air.simdgroup_async_copy_2d.p3i8.p1i8(i64 4, i64 4, ptr addrspace(3) %tile, i64 1, i64 1, <2 x i64> <i64 1, i64 1>, ptr addrspace(1) %buffer, i64 1, i64 1, <2 x i64> <i64 1, i64 1>, <2 x i64> zeroinitializer, i32 0)\n  call void @air.wait_simdgroup_events(i32 1, ptr %event)",
        "declare ptr @air.simdgroup_async_copy_2d.p3i8.p1i8(i64, i64, ptr addrspace(3), i64, i64, <2 x i64>, ptr addrspace(1), i64, i64, <2 x i64>, <2 x i64>, i32)\ndeclare void @air.wait_simdgroup_events(i32, ptr)",
    );
    assert_eq!(buffer_access(&ll).0, Some(ResourceAccess::ReadWrite));
    // The compiler's own attribute narrows it, unlowered as lowered.
    let readonly = ll.replacen(
        "ptr addrspace(1)  %buffer",
        "ptr addrspace(1) readonly %buffer",
        1,
    );
    assert_eq!(buffer_access(&readonly).0, Some(ResourceAccess::ReadOnly));
}

/// A buffer whose only use is a dynamic-length copy keeps its declared access, as it did after
/// the old engine lowered the copy into a loop over the same operands.
#[test]
fn a_dynamic_copy_as_a_buffers_only_use_keeps_its_declared_access() {
    let ll = kernel_using_buffer_only_in(
        "",
        "  %n = load i64, ptr addrspace(2) %length\n  tail call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) align 4 %tile, ptr addrspace(1) align 4 %buffer, i64 %n, i1 false)",
        "declare void @llvm.memcpy.p3.p1.i64(ptr addrspace(3), ptr addrspace(1), i64, i1)",
    );
    assert_eq!(buffer_access(&ll).0, Some(ResourceAccess::ReadWrite));
}

/// A buffer the body never names is unused, and an unknown call through the buffer leaves its
/// declared access alone: refinement narrows only by the parameter's attributes and by whether
/// the body names it at all.
#[test]
fn only_attributes_and_absence_narrow_a_buffers_access() {
    let unused = kernel_using_buffer_only_in("", "", "");
    assert_eq!(buffer_access(&unused).0, Some(ResourceAccess::Unused));
    let unknown = kernel_using_buffer_only_in(
        "",
        "  call void @opaque(ptr addrspace(1) %buffer)",
        "declare void @opaque(ptr addrspace(1))",
    );
    assert_eq!(buffer_access(&unknown).0, Some(ResourceAccess::ReadWrite));
    let writeonly = kernel_using_buffer_only_in(
        "writeonly",
        "  call void @opaque(ptr addrspace(1) %buffer)",
        "declare void @opaque(ptr addrspace(1))",
    );
    assert_eq!(buffer_access(&writeonly).0, Some(ResourceAccess::WriteOnly));
}

/// The descriptor bands refuse a buffer index past the buffer band, and a texture index past the
/// texture bands, naming the resource (the old engine's refusal ended with the same words).
#[test]
fn an_index_past_its_band_is_refused() {
    let module = |role: &str, index: u32, name: &str| {
        format!(
            r#"define void @k(ptr addrspace(1) %0) {{
entry:
  ret void
}}
!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.{role}", !"air.location_index", i32 {index}, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"{name}", !"air.arg_name", !"r"}}
"#
        )
    };
    for (ll, owner) in [
        (module("buffer", 40, "float"), "Buffer(40)"),
        (
            module("texture", 200, "texture2d<float, sample>"),
            "Texture(200)",
        ),
    ] {
        assert_eq!(
            inspect(&ll, Stage::Kernel, "k", &[]).unwrap_err(),
            format!("{owner} exceeds its descriptor ABI band")
        );
    }
}

/// A module of several tile functions: each one's interface is its own, read from what it reaches,
/// so another entry's implicit imageblock calls add no attachment to it, and one of a suffix
/// reflection does not decode (here `v3f16`) does not empty its bindings. Each entry's interface is
/// its projection's.
#[test]
fn another_entrys_implicit_imageblock_calls_are_not_this_ones() {
    let ll = r#"define void @a(ptr addrspace(1) %out, <2 x i16> %lid) {
  %value = load float, ptr addrspace(1) %out, align 4
  tail call void @air.store.implicit_imageblock.f32(float %value, i32 0, <2 x i16> %lid, i32 0, i16 0)
  ret void
}
define void @b(<2 x i16> %lid) {
  %value = tail call <3 x half> @air.load.implicit_imageblock.v3f16(i32 0, <2 x i16> %lid, i32 0, i16 0)
  ret void
}
define void @c(<2 x i16> %lid) {
  %value = tail call <4 x float> @air.load.implicit_imageblock.v4f32(i32 1, <2 x i16> %lid, i32 0, i16 0)
  ret void
}
declare void @air.store.implicit_imageblock.f32(float, i32, <2 x i16>, i32, i16)
declare <3 x half> @air.load.implicit_imageblock.v3f16(i32, <2 x i16>, i32, i16)
declare <4 x float> @air.load.implicit_imageblock.v4f32(i32, <2 x i16>, i32, i16)
!air.kernel = !{!0, !5, !7}
!0 = !{ptr @a, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}
!5 = !{ptr @b, !1, !6}
!6 = !{!8}
!7 = !{ptr @c, !1, !6}
!8 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}
"#;
    let a = inspect(ll, Stage::Kernel, "a", &[]).unwrap();
    assert!(a.binding_at(ResourceKind::Buffer, 0).is_some(), "{a:?}");
    let planes: Vec<_> = a
        .implicit_imageblock_attachments
        .iter()
        .map(|plane| (plane.attachment, plane.format, plane.access))
        .collect();
    assert_eq!(
        planes,
        [(0, TextureFormat::R32f, ResourceAccess::WriteOnly)]
    );
    let c = inspect(ll, Stage::Kernel, "c", &[]).unwrap();
    let planes: Vec<_> = c
        .implicit_imageblock_attachments
        .iter()
        .map(|plane| (plane.attachment, plane.format, plane.access))
        .collect();
    assert_eq!(
        planes,
        [(1, TextureFormat::Rgba32f, ResourceAccess::ReadOnly)]
    );
    for entry in ["a", "b", "c"] {
        let projected = crate::air::entry_projection(ll, entry).unwrap();
        assert_eq!(
            inspect(ll, Stage::Kernel, entry, &[]),
            inspect(&projected, Stage::Kernel, entry, &[]),
            "{entry}"
        );
    }
}

/// A read of a sample's texel, `read(coord, 5, imageblock_data_rate::sample)` (`i32 2, i16 5` on
/// the M3 Max), and one of the pixel's (`i32 0, i16 0`) are two planes of their attachment, each
/// at the binding of its data rate: the imageblock band's first binding, 200, plus 3 for each
/// attachment before it, plus the rate. The sample's index names no plane.
#[test]
fn a_sample_rate_read_takes_its_data_rate_s_plane() {
    let ll = r#"define void @k(<2 x i16> %lid) {
  %sample = tail call <4 x half> @air.load.implicit_imageblock.v4f16(i32 1, <2 x i16> %lid, i32 2, i16 5)
  %pixel = tail call <4 x half> @air.load.implicit_imageblock.v4f16(i32 1, <2 x i16> %lid, i32 0, i16 0)
  ret void
}
declare <4 x half> @air.load.implicit_imageblock.v4f16(i32, <2 x i16>, i32, i16)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}
"#;
    let k = inspect(ll, Stage::Kernel, "k", &[]).unwrap();
    let planes: Vec<_> = k
        .implicit_imageblock_attachments
        .iter()
        .map(|plane| (plane.attachment, plane.data_rate, plane.binding))
        .collect();
    assert_eq!(planes, [(1, 0, 203), (1, 2, 205)]);
}

/// A module of several entries sharing one argument buffer: the buffer's layout declares its
/// embedded texture member for every entry that takes it, so the member is in each entry's
/// `argument_buffer_fields`, but only the entry whose own reachable code loads the handle and
/// samples it binds a descriptor for it. The entry that reads another member alone binds none, and
/// each entry's interface is its projection's.
#[test]
fn another_entrys_sample_binds_no_embedded_texture_for_this_one() {
    let ll = r#"%struct.bundle = type <{ %"struct.metal::texture2d", ptr addrspace(1) }>
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define void @samples(ptr addrspace(2) %bundle) {
  %handle = getelementptr inbounds %struct.bundle, ptr addrspace(2) %bundle, i64 0, i32 0, i32 0
  %texture = load ptr addrspace(1), ptr addrspace(2) %handle, align 8
  %texel = tail call <4 x float> @air.sample_texture_2d.v4f32(ptr addrspace(1) %texture, <2 x float> zeroinitializer)
  ret void
}
define void @writes(ptr addrspace(2) %bundle) {
  %slot = getelementptr inbounds %struct.bundle, ptr addrspace(2) %bundle, i64 0, i32 1
  %out = load ptr addrspace(1), ptr addrspace(2) %slot, align 8
  store <4 x float> zeroinitializer, ptr addrspace(1) %out, align 16
  ret void
}
declare <4 x float> @air.sample_texture_2d.v4f32(ptr addrspace(1), <2 x float>)
!air.kernel = !{!0, !7}
!0 = !{ptr @samples, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"bundle", !"air.arg_name", !"bundle"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"tex", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"float4", !"out", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!7 = !{ptr @writes, !1, !2}
"#;
    let samples = inspect(ll, Stage::Kernel, "samples", &[]).unwrap();
    let writes = inspect(ll, Stage::Kernel, "writes", &[]).unwrap();
    // The ABI declares the texture member to both entries.
    for interface in [&samples, &writes] {
        assert_eq!(
            interface
                .argument_buffer_fields
                .iter()
                .map(|field| (field.argument_index, field.resource_kind))
                .collect::<Vec<_>>(),
            [
                (0, Some(ArgumentBufferResourceKind::Texture)),
                (1, Some(ArgumentBufferResourceKind::Buffer)),
            ],
            "{interface:?}"
        );
    }
    // Only the entry that reaches a texture intrinsic binds a descriptor for it.
    let embedded_textures = |interface: &ShaderInterface| {
        interface
            .bindings
            .iter()
            .filter(|binding| binding.kind == ResourceKind::EmbeddedArgBufferTexture)
            .map(|binding| (binding.metal_index, binding.descriptor, binding.access))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        embedded_textures(&samples),
        [(
            0,
            Some(DescriptorLocation {
                set: 0,
                binding: 32,
                count: 1
            }),
            Some(ResourceAccess::Sampled)
        )],
        "{samples:?}"
    );
    assert_eq!(embedded_textures(&writes), [], "{writes:?}");
    // Each entry's own buffer member binds either way: a buffer member comes from the layout
    // alone, with no use gate of its own.
    for interface in [&samples, &writes] {
        assert!(
            interface
                .bindings
                .iter()
                .any(|binding| binding.kind == ResourceKind::EmbeddedArgBufferBuffer),
            "{interface:?}"
        );
    }
    for entry in ["samples", "writes"] {
        let projected = crate::air::entry_projection(ll, entry).unwrap();
        assert_eq!(
            inspect(ll, Stage::Kernel, entry, &[]),
            inspect(&projected, Stage::Kernel, entry, &[]),
            "{entry}"
        );
    }
}
