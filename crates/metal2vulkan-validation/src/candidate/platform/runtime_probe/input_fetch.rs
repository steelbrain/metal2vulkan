//! A framebuffer fetch through an input attachment that aliases the color attachment, at `MSAA`
//! samples beside a depth/stencil attachment: what the device stores when a fragment shader reads
//! its own sample of the color attachment and writes the color back, with and without early
//! fragment tests.
//!
//! MoltenVK on an Apple GPU was measured to lose every storage-image write of an early-tests
//! fragment shader at more than one sample, so the storage form of framebuffer fetch cannot serve a
//! fetch-and-write fragment that declares early tests at 4 samples beside a depth or stencil
//! attachment. An input attachment over the color attachment keeps the color in the render target
//! and the tests in the hardware. Vulkan leaves such a read unordered against the same draw's
//! earlier writes without `VK_EXT_rasterization_order_attachment_access`; Metal orders its own
//! framebuffer fetch (`[[color(n)]]`). This step measures what the device answers, for comparison
//! with plain Metal's answer to the same draw (our probe's `fetch` mode): byte for
//! byte, every sample.
//!
//! The draw: an `SIDE`² `B8G8R8A8_SRGB` target at `MSAA` samples cleared to (0.75, 0.5, 0.25, 1),
//! `D32_SFLOAT_S8_UINT` cleared to 1 under `LESS` with writes, and one draw of six triangles in
//! order (full screen at 0.9, the lower-left half at 0.8, full at 0.95, which fails, full at 0.7,
//! the upper-right half at 0.75, which fails, full at 0.6), each fragment running per sample
//! (`SampleId`) and returning `c · 0.5 + 0.25` of the color it reads at its sample through the
//! input attachment. The subpass holds the color as its color attachment and input attachment 0,
//! both `GENERAL`, with a by-region self-dependency. Blending is off. A compute dispatch reads every
//! sample back through a `B8G8R8A8_UNORM` view and packs its bytes (r, g, b, a, low byte first),
//! as the Metal probe does: the lower-left half takes four writes, the upper-right half three.

use super::super::pipeline::create_shader_module;
use super::{result_name, Object, Owned, VulkanContext};
use ash::vk;
use metal2vulkan::device::VkFeature;
use serde::Serialize;

/// The target's side.
const SIDE: u32 = 8;

/// The samples.
const MSAA: u32 = 4;

/// The color format, and the view the readback reads its bytes through.
const COLOR: vk::Format = vk::Format::B8G8R8A8_SRGB;
const BYTES: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// The depth/stencil format.
const DEPTH: vk::Format = vk::Format::D32_SFLOAT_S8_UINT;

/// The color the target is cleared to.
const CLEAR: [f32; 4] = [0.75, 0.5, 0.25, 1.0];

/// The triangles, in order: clip-space `x, y` of three vertices, and the depth.
const TRIANGLES: [([f32; 6], f32); 6] = [
    (FULL, 0.9),
    (LOWER_LEFT, 0.8),
    (FULL, 0.95),
    (FULL, 0.7),
    (UPPER_RIGHT, 0.75),
    (FULL, 0.6),
];
const FULL: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
const LOWER_LEFT: [f32; 6] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0];
const UPPER_RIGHT: [f32; 6] = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0];

/// The words the readback buffer holds where nothing wrote.
const UNWRITTEN: u32 = 0xDEAD_BEEF;

/// The step's report.
#[derive(Debug, Default, Serialize)]
pub(super) struct InputFetch {
    /// Why the step could not run at all.
    error: Option<String>,
    /// Per variant, what the device stored.
    variants: Vec<Variant>,
    /// What the validation layer reported during the step: a variant drawn with one is not
    /// evidence.
    pub(super) validation_messages: Vec<String>,
}

/// One variant's answer.
#[derive(Debug, Serialize)]
struct Variant {
    /// Whether the fragment shader declares `EarlyFragmentTests`.
    early_tests: bool,
    /// Every sample's bytes, `((y · SIDE) + x) · MSAA + sample`, r in the low byte.
    words: Vec<u32>,
    /// Why the variant could not run.
    error: Option<String>,
}

pub(super) fn measure(context: &VulkanContext) -> InputFetch {
    let mut out = InputFetch::default();
    match supported(context).and_then(|()| Shaders::assemble()) {
        Err(error) => out.error = Some(error),
        Ok(shaders) => {
            for early_tests in [true, false] {
                let (words, error) = match run(context, &shaders, early_tests) {
                    Ok(words) => (words, None),
                    Err(error) => (Vec::new(), Some(error)),
                };
                out.variants.push(Variant {
                    early_tests,
                    words,
                    error,
                });
            }
        }
    }
    out.validation_messages = context.take_messages();
    out
}

/// Whether the device runs the step: per-sample shading, and both formats at `MSAA` samples with
/// the usages the step gives them.
fn supported(context: &VulkanContext) -> Result<(), String> {
    if !enabled(context, VkFeature::SampleRateShading) {
        return Err("sampleRateShading is not enabled".into());
    }
    let msaa = vk::SampleCountFlags::from_raw(MSAA);
    let formats = [
        (
            "B8G8R8A8_SRGB (color, input attachment, sampled, mutable)",
            COLOR,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::INPUT_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED,
            vk::ImageCreateFlags::MUTABLE_FORMAT,
        ),
        (
            "D32_SFLOAT_S8_UINT (depth/stencil attachment)",
            DEPTH,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageCreateFlags::empty(),
        ),
    ];
    for (name, format, usage, flags) in formats {
        // SAFETY: the context's physical device belongs to its live instance.
        let properties = unsafe {
            context
                .instance
                .get_physical_device_image_format_properties(
                    context.physical,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    usage,
                    flags,
                )
        }
        .map_err(|error| format!("{name}: {}", result_name(error)))?;
        if !properties.sample_counts.contains(msaa) {
            return Err(format!("{name} at {MSAA} samples"));
        }
    }
    Ok(())
}

/// The step's SPIR-V.
struct Shaders {
    vertex: Vec<u8>,
    /// With `EarlyFragmentTests`, and without.
    early: Vec<u8>,
    late: Vec<u8>,
    readback: Vec<u8>,
}

impl Shaders {
    fn assemble() -> Result<Shaders, String> {
        let assemble = |text: &str, label: &str| {
            crate::spirv_tools::assemble_validated(text, label).map_err(String::from)
        };
        Ok(Shaders {
            vertex: assemble(VERTEX, "input-fetch-vertex")?,
            early: assemble(&fragment(true), "input-fetch-fragment-early")?,
            late: assemble(&fragment(false), "input-fetch-fragment")?,
            readback: assemble(&readback(), "input-fetch-readback")?,
        })
    }
}

/// The vertex shader: location 0 is the clip-space position.
const VERTEX: &str = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %pos %out
               OpDecorate %pos Location 0
               OpMemberDecorate %PV 0 BuiltIn Position
               OpDecorate %PV Block
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
         %PV = OpTypeStruct %v4float
     %out_PV = OpTypePointer Output %PV
    %out_v4f = OpTypePointer Output %v4float
     %in_v4f = OpTypePointer Input %v4float
     %uint_0 = OpConstant %uint 0
        %pos = OpVariable %in_v4f Input
        %out = OpVariable %out_PV Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %p = OpLoad %v4float %pos
          %o = OpAccessChain %out_v4f %out %uint_0
               OpStore %o %p
               OpReturn
               OpFunctionEnd
"#;

/// The fragment shader: input attachment 0 read at the invocation's `SampleId`, `c · 0.5 + 0.25`
/// to color 0; with `EarlyFragmentTests` where `early`.
fn fragment(early: bool) -> String {
    let mode = if early {
        "OpExecutionMode %main EarlyFragmentTests"
    } else {
        ""
    };
    format!(
        r#"
               OpCapability Shader
               OpCapability InputAttachment
               OpCapability SampleRateShading
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %color %sid %input
               OpExecutionMode %main OriginUpperLeft
               {mode}
               OpDecorate %color Location 0
               OpDecorate %sid BuiltIn SampleId
               OpDecorate %sid Flat
               OpDecorate %input DescriptorSet 0
               OpDecorate %input Binding 0
               OpDecorate %input InputAttachmentIndex 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
      %v2int = OpTypeVector %int 2
    %v4float = OpTypeVector %float 4
        %img = OpTypeImage %float SubpassData 0 0 1 2 Unknown
     %uc_img = OpTypePointer UniformConstant %img
     %in_int = OpTypePointer Input %int
    %out_v4f = OpTypePointer Output %v4float
      %int_0 = OpConstant %int 0
      %zero2 = OpConstantComposite %v2int %int_0 %int_0
       %half = OpConstant %float 0.5
    %quarter = OpConstant %float 0.25
   %quarter4 = OpConstantComposite %v4float %quarter %quarter %quarter %quarter
      %color = OpVariable %out_v4f Output
        %sid = OpVariable %in_int Input
      %input = OpVariable %uc_img UniformConstant
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %im = OpLoad %img %input
          %s = OpLoad %int %sid
          %c = OpImageRead %v4float %im %zero2 Sample %s
          %h = OpVectorTimesScalar %v4float %c %half
          %r = OpFAdd %v4float %h %quarter4
               OpStore %color %r
               OpReturn
               OpFunctionEnd
"#
    )
}

/// The readback: each invocation `(x, y)` fetches every sample of the bytes view at binding 1 and
/// writes its four channels, rounded to bytes, r low, into binding 0 at `(y · SIDE + x) · MSAA + s`.
fn readback() -> String {
    let mut constants = String::new();
    let mut body = String::new();
    for s in 0..MSAA {
        constants.push_str(&format!(
            "        %uk{s} = OpConstant %uint {s}\n        %ik{s} = OpConstant %int {s}\n"
        ));
        body.push_str(&format!(
            "        %v{s} = OpImageFetch %v4float %t %xy Sample %ik{s}\n\
             \x20       %c{s} = OpExtInst %v4float %glsl FClamp %v{s} %zero4 %one4\n\
             \x20       %m{s} = OpVectorTimesScalar %v4float %c{s} %f255\n\
             \x20       %n{s} = OpExtInst %v4float %glsl RoundEven %m{s}\n\
             \x20       %u{s} = OpConvertFToU %v4uint %n{s}\n\
             \x20      %r{s} = OpCompositeExtract %uint %u{s} 0\n\
             \x20      %g{s} = OpCompositeExtract %uint %u{s} 1\n\
             \x20      %b{s} = OpCompositeExtract %uint %u{s} 2\n\
             \x20      %a{s} = OpCompositeExtract %uint %u{s} 3\n\
             \x20     %gs{s} = OpShiftLeftLogical %uint %g{s} %uint_8\n\
             \x20     %bs{s} = OpShiftLeftLogical %uint %b{s} %uint_16\n\
             \x20     %as{s} = OpShiftLeftLogical %uint %a{s} %uint_24\n\
             \x20     %o1{s} = OpBitwiseOr %uint %r{s} %gs{s}\n\
             \x20     %o2{s} = OpBitwiseOr %uint %o1{s} %bs{s}\n\
             \x20     %o3{s} = OpBitwiseOr %uint %o2{s} %as{s}\n\
             \x20       %i{s} = OpIAdd %uint %ps %uk{s}\n\
             \x20       %p{s} = OpAccessChain %sb_uint %buf %uint_0 %i{s}\n\
             \x20               OpStore %p{s} %o3{s}\n"
        ));
    }
    format!(
        r#"
               OpCapability Shader
       %glsl = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid %buf %tex
               OpExecutionMode %main LocalSize {SIDE} {SIDE} 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %arr ArrayStride 4
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
               OpDecorate %tex DescriptorSet 0
               OpDecorate %tex Binding 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %v2int = OpTypeVector %int 2
     %v3uint = OpTypeVector %uint 3
     %v4uint = OpTypeVector %uint 4
    %v4float = OpTypeVector %float 4
        %img = OpTypeImage %float 2D 0 0 1 1 Unknown
     %uc_img = OpTypePointer UniformConstant %img
        %arr = OpTypeRuntimeArray %uint
        %Buf = OpTypeStruct %arr
     %sb_Buf = OpTypePointer StorageBuffer %Buf
    %sb_uint = OpTypePointer StorageBuffer %uint
  %in_v3uint = OpTypePointer Input %v3uint
     %uint_0 = OpConstant %uint 0
     %uint_8 = OpConstant %uint 8
    %uint_16 = OpConstant %uint 16
    %uint_24 = OpConstant %uint 24
  %uint_side = OpConstant %uint {SIDE}
%uint_samples = OpConstant %uint {MSAA}
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
       %f255 = OpConstant %float 255
      %zero4 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
       %one4 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
{constants}
        %gid = OpVariable %in_v3uint Input
        %buf = OpVariable %sb_Buf StorageBuffer
        %tex = OpVariable %uc_img UniformConstant
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %g = OpLoad %v3uint %gid
         %gx = OpCompositeExtract %uint %g 0
         %gy = OpCompositeExtract %uint %g 1
         %ix = OpBitcast %int %gx
         %iy = OpBitcast %int %gy
         %xy = OpCompositeConstruct %v2int %ix %iy
          %t = OpLoad %img %tex
        %row = OpIMul %uint %gy %uint_side
         %px = OpIAdd %uint %row %gx
         %ps = OpIMul %uint %px %uint_samples
{body}
               OpReturn
               OpFunctionEnd
"#
    )
}

/// The vertices: per triangle vertex, clip-space `(x, y, z, 1)`.
fn vertex_bytes() -> Vec<u8> {
    TRIANGLES
        .iter()
        .flat_map(|(xy, z)| (0..3).map(move |i| [xy[2 * i], xy[2 * i + 1], *z, 1.0]))
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// Draw the variant and read every sample back.
fn run(context: &VulkanContext, shaders: &Shaders, early_tests: bool) -> Result<Vec<u32>, String> {
    let mut owned = Owned::new(context);
    let words = (SIDE * SIDE * MSAA) as usize;
    let samples = vk::SampleCountFlags::from_raw(MSAA);
    let image_info = |format: vk::Format, usage: vk::ImageUsageFlags| {
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: SIDE,
                height: SIDE,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    };
    let bytes = vertex_bytes();
    let (vertices, mapped) =
        owned.host_buffer(bytes.len() as u64, vk::BufferUsageFlags::VERTEX_BUFFER)?;
    // SAFETY: the mapping holds the buffer's bytes, and nothing on the device uses it yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped, bytes.len()) };
    let (out, out_mapped) =
        owned.host_buffer((words * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let out_mapped = out_mapped.cast::<u32>();
    // SAFETY: the mapping holds `words` host-coherent words, and nothing on the device uses the
    // buffer yet.
    unsafe { std::slice::from_raw_parts_mut(out_mapped, words) }.fill(UNWRITTEN);
    let color = owned.bound_image(
        &image_info(
            COLOR,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::INPUT_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED,
        )
        .flags(vk::ImageCreateFlags::MUTABLE_FORMAT),
    )?;
    let color_view = owned.view(color, COLOR, vk::ImageAspectFlags::COLOR)?;
    let bytes_view = owned.view(color, BYTES, vk::ImageAspectFlags::COLOR)?;
    let depth = owned.bound_image(&image_info(
        DEPTH,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
    ))?;
    let depth_view = owned.view(
        depth,
        DEPTH,
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
    )?;
    let pass = render_pass(&mut owned, samples)?;
    let framebuffer = owned.framebuffer(pass, &[color_view, depth_view], SIDE)?;
    // The fragment stage's set: the input attachment.
    let input_binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let input_set_layout = owned.descriptor_set_layout(&input_binding)?;
    let input_layout = owned.plain_layout(input_set_layout)?;
    let input_set = owned.sets(
        input_set_layout,
        &[vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::INPUT_ATTACHMENT)
            .descriptor_count(1)],
        1,
    )?[0];
    // The readback's set: the words and the bytes view.
    let readback_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let readback_set_layout = owned.descriptor_set_layout(&readback_bindings)?;
    let readback_layout = owned.plain_layout(readback_set_layout)?;
    let readback_set = owned.sets(
        readback_set_layout,
        &[
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
        ],
        1,
    )?[0];
    let input_info = [vk::DescriptorImageInfo::default()
        .image_view(color_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let out_info = [vk::DescriptorBufferInfo::default()
        .buffer(out)
        .range(vk::WHOLE_SIZE)];
    let bytes_info = [vk::DescriptorImageInfo::default()
        .image_view(bytes_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(input_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
            .image_info(&input_info),
        vk::WriteDescriptorSet::default()
            .dst_set(readback_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&out_info),
        vk::WriteDescriptorSet::default()
            .dst_set(readback_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&bytes_info),
    ];
    // SAFETY: the sets, buffer and views are live objects of this device; nothing uses the sets.
    unsafe { owned.device().update_descriptor_sets(&writes, &[]) };
    let vertex = owned.module(&shaders.vertex, "input fetch vertex")?;
    let fragment_spv = if early_tests {
        &shaders.early
    } else {
        &shaders.late
    };
    let fragment = owned.module(fragment_spv, "input fetch fragment")?;
    let readback =
        owned.compute_pipeline(&shaders.readback, "input fetch readback", readback_layout)?;
    let draw = pipeline(&mut owned, (vertex, fragment), input_layout, pass, samples)?;
    owned.submit("the input fetch probe's draw", |device, command| {
        let clears = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: CLEAR },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: SIDE,
                    height: SIDE,
                },
            })
            .clear_values(&clears);
        let to_host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        // SAFETY: the command buffer is recording; every object it names is live. The pass's
        // external dependency makes its color writes visible to the readback.
        unsafe {
            device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, draw);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                input_layout,
                0,
                &[input_set],
                &[],
            );
            device.cmd_bind_vertex_buffers(command, 0, &[vertices], &[0]);
            device.cmd_draw(command, 3 * TRIANGLES.len() as u32, 1, 0, 0);
            device.cmd_end_render_pass(command);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, readback);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                readback_layout,
                0,
                &[readback_set],
                &[],
            );
            device.cmd_dispatch(command, 1, 1, 1);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &to_host,
                &[],
                &[],
            );
        }
    })?;
    // SAFETY: the mapping holds `words` host-coherent words, and the submission that wrote them
    // completed, its writes made available to the host by the barrier.
    Ok(unsafe { std::slice::from_raw_parts(out_mapped, words) }.to_vec())
}

/// The render pass: the color as color attachment and input attachment 0, both `GENERAL`, with a
/// by-region self-dependency from color writes to input reads; the depth/stencil attachment; and
/// the color's writes made visible to the readback's compute.
fn render_pass(
    owned: &mut Owned<'_>,
    samples: vk::SampleCountFlags,
) -> Result<vk::RenderPass, String> {
    let attachments = [
        vk::AttachmentDescription::default()
            .format(COLOR)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL),
        vk::AttachmentDescription::default()
            .format(DEPTH)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let general = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::GENERAL,
    };
    let colors = [general];
    let inputs = [general];
    let depth = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors)
        .input_attachments(&inputs)
        .depth_stencil_attachment(&depth)];
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
            .dependency_flags(vk::DependencyFlags::BY_REGION),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];
    owned.render_pass(
        &vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies),
    )
}

/// The draw's pipeline: the viewport y flipped (as the product draws), no culling, per-sample
/// shading, depth `LESS` with writes, blending off with every channel written.
fn pipeline(
    owned: &mut Owned<'_>,
    (vertex, fragment): (vk::ShaderModule, vk::ShaderModule),
    layout: vk::PipelineLayout,
    pass: vk::RenderPass,
    samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline, String> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let bindings = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: 16,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attributes = [vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: 0,
    }];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let side = SIDE as f32;
    let viewports = [vk::Viewport {
        x: 0.0,
        y: side,
        width: side,
        height: -side,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: SIDE,
            height: SIDE,
        },
    }];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(samples)
        .sample_shading_enable(true)
        .min_sample_shading(1.0);
    let depth_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_state)
        .color_blend_state(&blending)
        .layout(layout)
        .render_pass(pass)
        .subpass(0);
    owned.graphics_pipeline(&info, "input fetch")
}

/// Whether the probe's context enabled `feature`.
fn enabled(context: &VulkanContext, feature: VkFeature) -> bool {
    context.enabled.features.contains(&feature)
}

/// The objects one case creates, beyond [`Owned`]'s own kinds.
impl Owned<'_> {
    /// A device-local `info` image bound to memory of its own.
    fn bound_image(&mut self, info: &vk::ImageCreateInfo<'_>) -> Result<vk::Image, String> {
        let image = self
            .image(info)
            .map_err(|error| format!("create an image: {}", result_name(error)))?;
        // SAFETY: the image was just created on this device.
        let requirements = unsafe { self.device().get_image_memory_requirements(image) };
        let memory_type = device_type(self.context, requirements.memory_type_bits)
            .ok_or("no memory type for an image")?;
        let memory = self.memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
        )?;
        // SAFETY: the image and memory are this device's, unbound, and the type fits.
        unsafe { self.device().bind_image_memory(image, memory, 0) }
            .map_err(|error| format!("bind an image: {}", result_name(error)))?;
        Ok(image)
    }

    fn view(
        &mut self,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView, String> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range(aspect));
        // SAFETY: the image is a live image of this device of `format`.
        let view = unsafe { self.device().create_image_view(&info, None) }
            .map_err(|error| format!("create an image view: {}", result_name(error)))?;
        self.objects.push(Object::ImageView(view));
        Ok(view)
    }

    fn render_pass(
        &mut self,
        info: &vk::RenderPassCreateInfo<'_>,
    ) -> Result<vk::RenderPass, String> {
        // SAFETY: the device is live, and `info` is valid for the call.
        let pass = unsafe { self.device().create_render_pass(info, None) }
            .map_err(|error| format!("create a render pass: {}", result_name(error)))?;
        self.objects.push(Object::RenderPass(pass));
        Ok(pass)
    }

    fn framebuffer(
        &mut self,
        pass: vk::RenderPass,
        views: &[vk::ImageView],
        side: u32,
    ) -> Result<vk::Framebuffer, String> {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(views)
            .width(side)
            .height(side)
            .layers(1);
        // SAFETY: the pass and views are live objects of this device, compatible with each other.
        let framebuffer = unsafe { self.device().create_framebuffer(&info, None) }
            .map_err(|error| format!("create a framebuffer: {}", result_name(error)))?;
        self.objects.push(Object::Framebuffer(framebuffer));
        Ok(framebuffer)
    }

    fn module(&mut self, spv: &[u8], label: &str) -> Result<vk::ShaderModule, String> {
        let module = create_shader_module(self.context, spv, label)?;
        self.objects.push(Object::ShaderModule(module));
        Ok(module)
    }

    /// A pipeline layout of one set and no push constants.
    fn plain_layout(&mut self, set: vk::DescriptorSetLayout) -> Result<vk::PipelineLayout, String> {
        let sets = [set];
        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets);
        // SAFETY: the set layout is a live object of this device.
        let layout = unsafe { self.device().create_pipeline_layout(&info, None) }
            .map_err(|error| format!("create a pipeline layout: {}", result_name(error)))?;
        self.objects.push(Object::PipelineLayout(layout));
        Ok(layout)
    }

    /// `count` descriptor sets of `layout` from a pool of their own of `sizes`.
    fn sets(
        &mut self,
        layout: vk::DescriptorSetLayout,
        sizes: &[vk::DescriptorPoolSize],
        count: u32,
    ) -> Result<Vec<vk::DescriptorSet>, String> {
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(count)
            .pool_sizes(sizes);
        // SAFETY: the device is live, and `info` is valid for the call.
        let pool = unsafe { self.device().create_descriptor_pool(&info, None) }
            .map_err(|error| format!("create a descriptor pool: {}", result_name(error)))?;
        self.objects.push(Object::DescriptorPool(pool));
        let layouts = vec![layout; count as usize];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are live objects of this device; the pool has room.
        unsafe { self.device().allocate_descriptor_sets(&allocate) }
            .map_err(|error| format!("allocate descriptor sets: {}", result_name(error)))
    }

    fn graphics_pipeline(
        &mut self,
        info: &vk::GraphicsPipelineCreateInfo<'_>,
        label: &str,
    ) -> Result<vk::Pipeline, String> {
        // SAFETY: every object `info` names is a live object of this device.
        let pipelines = unsafe {
            self.device()
                .create_graphics_pipelines(vk::PipelineCache::null(), &[*info], None)
        }
        .map_err(|(_, error)| format!("create the {label} pipeline: {}", result_name(error)))?;
        self.objects.push(Object::Pipeline(pipelines[0]));
        Ok(pipelines[0])
    }
}

/// The first memory type among `bits` that is device-local, else the first among them.
fn device_type(context: &VulkanContext, bits: u32) -> Option<u32> {
    // SAFETY: the context's physical device belongs to its live instance.
    let properties = unsafe {
        context
            .instance
            .get_physical_device_memory_properties(context.physical)
    };
    let types = properties.memory_types_as_slice();
    let allowed = |index: &usize| bits & (1 << index) != 0;
    (0..types.len())
        .filter(allowed)
        .find(|&index| {
            types[index]
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .or_else(|| (0..types.len()).find(allowed))
        .map(|index| index as u32)
}

fn range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The step's shaders assemble and pass `spirv-val`.
    #[test]
    fn the_shaders_assemble_and_validate() {
        let shaders = Shaders::assemble().unwrap_or_else(|error| panic!("{error}"));
        for spv in [
            &shaders.vertex,
            &shaders.early,
            &shaders.late,
            &shaders.readback,
        ] {
            assert!(!spv.is_empty());
        }
    }

    /// The vertices are the six triangles in order, three vertices each of `(x, y, z, 1)`.
    #[test]
    fn the_vertices_are_the_triangles_in_order() {
        let bytes = vertex_bytes();
        assert_eq!(bytes.len(), TRIANGLES.len() * 3 * 16);
        let floats: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(&floats[..4], &[-1.0, -1.0, 0.9, 1.0]);
        assert_eq!(&floats[12..16], &[-1.0, -1.0, 0.8, 1.0]);
        assert_eq!(&floats[48..52], &[1.0, 1.0, 0.75, 1.0]);
        assert_eq!(&floats[68..72], &[-1.0, 3.0, 0.6, 1.0]);
    }
}
