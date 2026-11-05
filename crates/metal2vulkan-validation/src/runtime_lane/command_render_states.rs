//! A render command's own render states and object threadgroup lengths on the device the cfg
//! selects, under the validation layer, with the functions of a Metal probe on an M4
//! (`render_state.rs`) in its 4x4 pass of a `Depth32Float` attachment cleared to 1.0.
//!
//! Every answer is plain Metal's on an M4 with macOS 27.0.1 (each case plain and under Metal's
//! validation):
//!
//! - A buffer whose descriptor leaves the depth/stencil state, the depth bias, the depth clip
//!   mode, the cull mode, the front-facing winding and the triangle fill mode to its commands
//!   (each `inherit_*` false) draws each command with the encoder's states at the call, as an
//!   inheriting buffer's commands and the encoder's own draw do: Metal's validation stops the
//!   buffer's creation, and plain Metal draws so. A buffer that supports color attachment mapping
//!   executes in an encoder that maps none as any other. So do Apple's private properties (the same
//!   M4): stencil reference values, depth test bounds, scissor rectangles,
//!   viewports and the blend color left to the commands, four scissor rectangles and viewports, and
//!   render state overrides allowed: the commands draw with the encoder's viewport and scissor
//!   rectangle too.
//! - Each render state setter of a command ends Metal's process (`APINotSupported`), whatever the
//!   buffer inherits, and is refused `Invalid` (`RenderState`) on a device that does not claim
//!   Apple10; on one that claims it, which would draw the command with its own state, the setter
//!   and the creation of a buffer that does not inherit are `NotImplemented`.
//! - An object threadgroup length is taken at any index of a buffer of mesh draws, past the
//!   descriptor's count too (where Metal's validation stops it), and its command draws as before;
//!   on a buffer of no mesh draw it ends Metal's process and is refused `Invalid`
//!   (`ObjectThreadgroupMemory`).
//! - A kernel that sets a render state on a render command (`render_command::set_cull_mode` and
//!   the rest, `set_depth_stencil_state` among them) fails its compute pipeline state's creation,
//!   and the runtime refuses the creation `Unsupported` (`IndirectRenderState`,
//!   `Needs::MetalGpuFamily` of Apple10) on a device that does not claim
//!   Apple10, `NotImplemented` on one that does. A kernel whose argument buffer holds a
//!   `depth_stencil_state` it does not read is made.

use super::render_state::{quiet, rows, validated, Pipe, Probe, EMPTY, FULL, PASS, WHITE};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkProperty;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::{
    CompareFunction, CullMode, DepthClipMode, IndirectCommandType, PixelFormat, PrimitiveType,
    TriangleFillMode, Winding,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, Device, Error, GpuFamily, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectCommandBufferProblem, Invalid, PipelineOptions, Problem, RenderCommandEncoder,
    ResourceOptions, RuntimeGap, ScissorRect, Viewport,
};

/// The command's and the direct draw's pipe: white, at the pass's depth format.
fn white() -> Pipe {
    WHITE.depth(PixelFormat::Depth32Float)
}

/// A buffer of one draw of primitives (and mesh draws where `mesh`), one vertex buffer, its
/// render states inherited unless `own`, which also makes it support color attachment mapping
/// and sets Apple's private counts and overrides off their defaults.
fn descriptor(own: bool, mesh: bool) -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = if mesh {
        IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_MESH_THREADGROUPS)
    } else {
        IndirectCommandType::DRAW
    };
    descriptor.max_vertex_buffer_bind_count = 1;
    descriptor.inherit_depth_stencil_state = !own;
    descriptor.inherit_depth_bias = !own;
    descriptor.inherit_depth_clip_mode = !own;
    descriptor.inherit_cull_mode = !own;
    descriptor.inherit_front_facing_winding = !own;
    descriptor.inherit_triangle_fill_mode = !own;
    descriptor.support_color_attachment_mapping = own;
    descriptor.inherit_stencil_reference_values = !own;
    descriptor.inherit_depth_test_bounds = !own;
    descriptor.inherit_scissor_rects = !own;
    descriptor.inherit_viewports = !own;
    descriptor.inherit_blend_color = !own;
    if own {
        descriptor.max_scissor_rect_count = 4;
        descriptor.max_viewport_count = 4;
        descriptor.allow_override_render_states = 1;
    }
    descriptor
}

/// A buffer of `descriptor` whose command 0 draws the white fullscreen triangle at depth `z`,
/// with the depth buffer it reads.
fn command(
    probe: &Probe,
    descriptor: &IndirectCommandBufferDescriptor,
    z: f32,
) -> (IndirectCommandBuffer, Buffer) {
    let icb = probe
        .device
        .new_indirect_command_buffer(descriptor, 1, ResourceOptions::shared())
        .unwrap();
    let at = icb.indirect_render_command_at(0).unwrap();
    at.set_render_pipeline_state(&probe.pipeline(white().indirect()))
        .unwrap();
    let depth = probe.depth(z, 0.0);
    at.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
    at.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    (icb, depth)
}

/// The encoder's states before the command, set by a case: `Err` where the device refuses one.
type Setup = fn(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>) -> Result<(), Error>;

/// Whether `seen` holds Metal's texels `metal`, where a channel Metal stores as `0x80`, its code
/// for `rt6_half`'s 0.5, a half step between two codes (127.5), may hold either nearest code on a
/// device whose color writes are not Metal's (no `VkProperty::MetalColorWrites`: the pipeline
/// records `PrecisionNote::DeviceColorConversion`). Every other channel compares exactly.
fn metals_texels(device: &Device, seen: &[u8], metal: &[u8]) -> bool {
    let metals = device
        .compile_device()
        .profile()
        .enabled()
        .properties
        .contains(&VkProperty::MetalColorWrites);
    seen.len() == metal.len()
        && seen
            .iter()
            .zip(metal)
            .all(|(&code, &want)| code == want || (!metals && want == 0x80 && code == 0x7f))
}

/// The white draw at depth 0.5 of `rt6_half`'s 0x80 texels, under `Less` with writes.
fn half_first(probe: &Probe, e: &mut RenderCommandEncoder, held: &mut Vec<Buffer>) {
    e.set_depth_stencil_state(&probe.state(CompareFunction::Less, true, None))
        .unwrap();
    let half = WHITE.fragment("rt6_half").depth(PixelFormat::Depth32Float);
    probe.draw(e, held, half, 0.5, 0.0);
}

/// Commands of a buffer that does not inherit its render states draw with the encoder's at the
/// call, as an inheriting buffer's commands and the encoder's own draw do (the
/// M4's plain answer; the module docs): under `Never` nothing; under `Less` behind an earlier
/// draw at the same depth nothing, and through a bias of -1000 everything; culled, wound and
/// filled as the encoder's draw; at depth 1.5 in clamp mode everything; and through the encoder's
/// left-half viewport and top-half scissor rectangle as its draw. A device that refuses fill mode
/// `Lines` or clip mode `Clamp` skips that case. The earlier draw's 0.5 is a half step between two
/// codes, held to Metal's code where the device's color writes are Metal's and to either nearest
/// one elsewhere ([`metals_texels`]).
#[test]
fn commands_that_do_not_inherit_render_states_draw_with_the_encoder_s() {
    let (messages, _cache, probe) = validated();
    let name = probe.device.name().to_owned();
    if probe.device.supports_family(GpuFamily::Apple10.raw()) {
        assert_eq!(
            probe
                .device
                .new_indirect_command_buffer(&descriptor(true, false), 1, ResourceOptions::shared())
                .err(),
            Some(Error::NotImplemented(RuntimeGap::IndirectRenderCommands)),
            "{name}: a buffer that leaves its render states to its commands on Apple10"
        );
        quiet(probe, &messages);
        return;
    }
    let halves = [0x80; 4].repeat(16);
    let cases: [(&str, f32, Setup, Option<Vec<u8>>); 10] = [
        (
            "never",
            0.5,
            |p, e, _| e.set_depth_stencil_state(&p.state(CompareFunction::Never, false, None)),
            Some(rows(&EMPTY)),
        ),
        (
            "less, behind a draw",
            0.5,
            |p, e, h| {
                half_first(p, e, h);
                Ok(())
            },
            Some(halves.clone()),
        ),
        (
            "less, biased past a draw",
            0.5,
            |p, e, h| {
                half_first(p, e, h);
                e.set_depth_bias(-1000.0, 0.0, 0.0)
            },
            Some(rows(&FULL)),
        ),
        (
            "cull back",
            0.5,
            |_, e, _| {
                e.set_cull_mode(CullMode::Back);
                Ok(())
            },
            None,
        ),
        (
            "cull front",
            0.5,
            |_, e, _| {
                e.set_cull_mode(CullMode::Front);
                Ok(())
            },
            None,
        ),
        (
            "counter-clockwise, cull back",
            0.5,
            |_, e, _| {
                e.set_front_facing_winding(Winding::CounterClockwise);
                e.set_cull_mode(CullMode::Back);
                Ok(())
            },
            None,
        ),
        (
            "lines",
            0.5,
            |_, e, _| e.set_triangle_fill_mode(TriangleFillMode::Lines),
            None,
        ),
        (
            "clamp at 1.5",
            1.5,
            |_, e, _| e.set_depth_clip_mode(DepthClipMode::Clamp),
            Some(rows(&FULL)),
        ),
        (
            "viewport, left half",
            0.5,
            |_, e, _| {
                e.set_viewport(Viewport {
                    origin_x: 0.0,
                    origin_y: 0.0,
                    width: 2.0,
                    height: 4.0,
                    znear: 0.0,
                    zfar: 1.0,
                })
            },
            None,
        ),
        (
            "scissor, top half",
            0.5,
            |_, e, _| {
                e.set_scissor_rect(ScissorRect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 2,
                })
            },
            None,
        ),
    ];
    let pass = PASS.depth(PixelFormat::Depth32Float, 1.0);
    for (case, z, setup, metal) in cases {
        // The device's own refusal of the encoder's state skips the case.
        let mut refused = None;
        let direct = probe.run(pass, |p, e, h| match setup(p, e, h) {
            Ok(()) => p.draw(e, h, white(), z, 0.0),
            Err(error) => refused = Some(error),
        });
        if let Some(error) = refused {
            println!("{name}: {case}: the encoder refuses the state: {error}");
            continue;
        }
        for own in [false, true] {
            let (icb, depth) = command(&probe, &descriptor(own, false), z);
            let executed = probe.run(pass, |p, e, h| {
                setup(p, e, h).unwrap();
                e.execute_commands_in_buffer(&icb, 0..1).unwrap();
            });
            assert_eq!(
                executed.color, direct.color,
                "{name}: {case}, own states {own}: the command against the encoder's draw"
            );
            drop((icb, depth));
        }
        if let Some(metal) = metal {
            assert!(
                metals_texels(&probe.device, &direct.color, &metal),
                "{name}: {case}: Metal's texels {metal:?}, the encoder's draw {:?}",
                direct.color
            );
        }
    }
    quiet(probe, &messages);
}

/// Each render state setter of a command is refused, on a buffer that inherits the states and
/// on one that does not, where Metal's process ends: `Invalid` (`RenderState`) off Apple10,
/// `NotImplemented` on it. An object threadgroup length is taken at any index of a buffer of mesh
/// draws, its command drawing as before, and refused `Invalid` (`ObjectThreadgroupMemory`) on a
/// buffer of no mesh draw.
#[test]
fn a_command_s_render_states_are_refused_and_object_lengths_taken() {
    let (messages, _cache, probe) = validated();
    let name = probe.device.name().to_owned();
    let apple10 = probe.device.supports_family(GpuFamily::Apple10.raw());
    let refused = |call| match apple10 {
        true => Error::NotImplemented(RuntimeGap::IndirectRenderCommands),
        false => Error::Invalid(Invalid::new(
            call,
            Problem::IndirectCommandBuffer(IndirectCommandBufferProblem::RenderState),
        )),
    };
    let state = probe.state(CompareFunction::Always, false, None);
    for own in [false, true] {
        if own && apple10 {
            continue;
        }
        let (icb, depth) = command(&probe, &descriptor(own, false), 0.5);
        let at = icb.indirect_render_command_at(0).unwrap();
        let calls: [(&'static str, Result<(), Error>); 7] = [
            (
                "set_depth_stencil_state",
                at.set_depth_stencil_state(Some(&state)),
            ),
            ("set_depth_stencil_state", at.set_depth_stencil_state(None)),
            ("set_depth_bias", at.set_depth_bias(-1000.0, 0.0, 0.0)),
            (
                "set_depth_clip_mode",
                at.set_depth_clip_mode(DepthClipMode::Clamp),
            ),
            ("set_cull_mode", at.set_cull_mode(CullMode::Back)),
            (
                "set_front_facing_winding",
                at.set_front_facing_winding(Winding::CounterClockwise),
            ),
            (
                "set_triangle_fill_mode",
                at.set_triangle_fill_mode(TriangleFillMode::Lines),
            ),
        ];
        for (call, answer) in calls {
            assert_eq!(
                answer,
                Err(refused(call)),
                "{name}: {call}, own states {own}"
            );
        }
        drop((icb, depth));
    }
    // Object threadgroup lengths: taken at and past the count of a buffer of mesh draws, whose
    // command draws as before.
    let mut mesh = descriptor(false, true);
    mesh.max_object_threadgroup_memory_bind_count = 2;
    let (icb, depth) = command(&probe, &mesh, 0.5);
    let at = icb.indirect_render_command_at(0).unwrap();
    for index in [1, 2] {
        assert_eq!(
            at.set_object_threadgroup_memory_length(16, index),
            Ok(()),
            "{name}: object threadgroup length {index}"
        );
    }
    let executed = probe.run(PASS.depth(PixelFormat::Depth32Float, 1.0), |_, e, _| {
        e.execute_commands_in_buffer(&icb, 0..1).unwrap();
    });
    assert_eq!(
        executed.color,
        rows(&FULL),
        "{name}: the command draws as before"
    );
    drop((icb, depth));
    let (icb, depth) = command(&probe, &descriptor(false, false), 0.5);
    let at = icb.indirect_render_command_at(0).unwrap();
    assert_eq!(
        at.set_object_threadgroup_memory_length(16, 0),
        Err(Error::Invalid(Invalid::new(
            "set_object_threadgroup_memory_length",
            Problem::IndirectCommandBuffer(IndirectCommandBufferProblem::ObjectThreadgroupMemory),
        ))),
        "{name}: an object threadgroup length on a buffer of no mesh draw"
    );
    drop((icb, depth));
    quiet(probe, &messages);
}

/// Each kernel of [`setter_kernels`]: its name, and the setter it calls with its values (`None`
/// for the control), as our probe's kernels set them.
const SETTER_KERNELS: [(&str, Option<(&str, &str)>); 7] = [
    ("k_ctl", None),
    ("k_cull", Some(("set_cull_mode", "i32 2"))),
    ("k_winding", Some(("set_front_facing_winding", "i32 1"))),
    ("k_fill", Some(("set_triangle_fill_mode", "i32 1"))),
    (
        "k_bias",
        Some((
            "set_depth_bias",
            "float -1.000000e+03, float 0.0, float 0.0",
        )),
    ),
    ("k_clip", Some(("set_depth_clip_mode", "i32 1"))),
    (
        "k_ds",
        Some(("set_depth_stencil_state", "ptr addrspace(1) %ds")),
    ),
];

/// One module of [`SETTER_KERNELS`], each `kernel void k(device const Enc &e [[buffer(0)]])` of
/// `struct Enc { command_buffer cmds; depth_stencil_state ds; }` writing render command 0: its
/// setter, then a draw of 6 vertices of triangles, as Metal's compiler declares them.
fn setter_kernels() -> String {
    let mut text = String::from("%enc = type { ptr addrspace(1), ptr addrspace(1) }\n\n");
    let mut declared = Vec::new();
    for (kernel, setter) in SETTER_KERNELS {
        text += &format!(
            "define void @{kernel}(ptr addrspace(1) noundef %e) {{\nentry:\n  \
             %pc = getelementptr inbounds %enc, ptr addrspace(1) %e, i64 0, i32 0\n  \
             %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8\n"
        );
        if let Some((setter, values)) = setter {
            // The state is read only by the setter that takes it, as Metal's compiler reads it.
            if values.contains("%ds") {
                text +=
                    "  %ps = getelementptr inbounds %enc, ptr addrspace(1) %e, i64 0, i32 1\n  \
                         %ds = load ptr addrspace(1), ptr addrspace(1) %ps, align 8\n";
            }
            text += &format!(
                "  call void @air.{setter}_render_command(ptr addrspace(1) %cmd, i32 0, {values})\n"
            );
            let types: Vec<&str> = values
                .split(", ")
                .map(|value| value.rsplit_once(' ').unwrap().0)
                .collect();
            declared.push(format!(
                "declare void @air.{setter}_render_command(ptr addrspace(1), i32, {})",
                types.join(", ")
            ));
        }
        text += "  call void @air.draw_primitives_render_command(ptr addrspace(1) %cmd, i32 0, \
                 i32 3, i32 0, i32 6, i32 1, i32 0)\n  ret void\n}\n\n";
    }
    for declaration in declared {
        text += &declaration;
        text.push('\n');
    }
    text += "declare void @air.draw_primitives_render_command(ptr addrspace(1), i32, i32, i32, \
             i32, i32, i32)\n\n";
    let kernels: Vec<String> = (0..SETTER_KERNELS.len()).map(|i| format!("!{i}")).collect();
    text += &format!(
        "!air.kernel = !{{{}}}\n!air.version = !{{!20}}\n!air.language_version = !{{!21}}\n",
        kernels.join(", ")
    );
    for (i, (kernel, _)) in SETTER_KERNELS.iter().enumerate() {
        text += &format!("!{i} = !{{ptr @{kernel}, !10, !11}}\n");
    }
    text += r#"!10 = !{}
!11 = !{!12}
!12 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !13, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Enc", !"air.arg_name", !"e"}
!13 = !{i32 0, i32 8, i32 0, !"command_buffer", !"cmds", !"air.indirect_argument", !14, i32 8, i32 8, i32 0, !"depth_stencil_state", !"ds", !"air.indirect_argument", !15}
!14 = !{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"cmds"}
!15 = !{i32 1, !"air.depth_stencil_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"depth_stencil_state", !"air.arg_name", !"ds"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;
    text
}

/// A kernel that sets a render state on a render command fails its compute pipeline state's
/// creation, as on an M4 with macOS 27.0.1, plain and under Metal's validation, for each of the
/// six setters ("Encountered unlowered function call to air.set_<state>_render_command. Verify
/// device or GPU architecture supports this feature"), and on an M3 Max with macOS 26.7: refused
/// `Unsupported` (`IndirectRenderState`, `Needs::MetalGpuFamily` of Apple10) on a device that does
/// not claim Apple10, `NotImplemented(IndirectRenderCommands)` on one that does, whose kernel would
/// set it. The control, whose kernel draws beside a `depth_stencil_state` it does not read, is
/// made, as Metal makes it.
#[test]
fn a_kernel_s_render_state_setters_fail_its_pipeline_s_creation() {
    let (messages, _cache, probe) = validated();
    let name = probe.device.name().to_owned();
    let apple10 = GpuFamily::Apple10.raw();
    let library = probe
        .device
        .new_library_with_air(&llvm_assemble_text(&setter_kernels()).unwrap())
        .unwrap();
    for (kernel, setter) in SETTER_KERNELS {
        let function = library.new_function_with_name(kernel).unwrap();
        let made = probe.device.new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        );
        match (setter, made) {
            (None, made) => assert!(made.is_ok(), "{name}: {kernel}: {:?}", made.err()),
            (Some(_), Err(Error::NotImplemented(gap))) if probe.device.supports_family(apple10) => {
                assert_eq!(gap, RuntimeGap::IndirectRenderCommands, "{name}: {kernel}")
            }
            (Some(_), Err(Error::Unsupported(unsupported))) => assert_eq!(
                (unsupported.feature, unsupported.needs),
                (Feature::IndirectRenderState, Needs::MetalGpuFamily(apple10)),
                "{name}: {kernel}"
            ),
            (Some(_), made) => panic!("{name}: {kernel}: {:?}", made.err()),
        }
    }
    quiet(probe, &messages);
}
