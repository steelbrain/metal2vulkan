//! The routed commands' entry points, one per command and slot, and each slot's commands as its
//! device's driver gave them (`recorder`): each entry point counts its command for its slot and
//! calls the driver's.

use super::SLOTS;
use crate::generated_commands::GeneratedCommandsInfo;
use ash::vk;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// Each slot's commands recorded so far, one count per command at its index.
static COUNTS: [[AtomicU64; Command::COUNT]; SLOTS] =
    [const { [const { AtomicU64::new(0) }; Command::COUNT] }; SLOTS];

/// Slot `index`'s counts.
pub(super) fn counts(index: usize) -> CommandsReplayed {
    CommandsReplayed::read(&COUNTS[index])
}

/// Zero slot `index`'s counts, as a device takes it.
pub(super) fn clear_counts(index: usize) {
    for count in &COUNTS[index] {
        count.store(0, Ordering::Relaxed);
    }
}

/// A command whose driver's entry point was not stored for the slot whose entry point was called:
/// a device's functions routed before its commands were (a defect of the runtime's). It ends the
/// process: a command recorded otherwise would be lost.
#[cold]
fn missing(name: &CStr) -> ! {
    panic!("the routed command {name:?} has no driver's entry point");
}

/// Declare the routed commands: those of `ash`'s 1.0 table (`v1_0`), of its 1.3 table (`v1_3`),
/// and the extension commands the runtime loads itself (`extension`), each with its variant of
/// [`Command`], its field in its table (or a name of its own), its Vulkan name and its parameters
/// after the command buffer.
macro_rules! commands {
    (
        v1_0 { $($v0:ident $f0:ident $n0:literal ($($a0:ident: $t0:ty),*) $(-> $r0:ty)?;)* }
        v1_3 { $($v3:ident $f3:ident $n3:literal ($($a3:ident: $t3:ty),*) $(-> $r3:ty)?;)* }
        extension { $($ve:ident $fe:ident $ne:literal ($($ae:ident: $te:ty),*) $(-> $re:ty)?;)* }
    ) => {
        /// A command routed through the slots (`recorder`).
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub(crate) enum Command {
            $($v0,)*
            $($v3,)*
            $($ve,)*
        }

        impl Command {
            /// Every routed command.
            pub(crate) const ALL: &'static [Command] = &[
                $(Command::$v0,)*
                $(Command::$v3,)*
                $(Command::$ve,)*
            ];

            /// How many commands are routed: each one's index (`command as usize`) is below it.
            pub(crate) const COUNT: usize = Command::ALL.len();

            /// Its Vulkan name.
            pub(crate) fn name(self) -> &'static CStr {
                match self {
                    $(Command::$v0 => $n0,)*
                    $(Command::$v3 => $n3,)*
                    $(Command::$ve => $ne,)*
                }
            }
        }

        /// The commands a device recorded (`recorder`), by kind: each routed command's calls into
        /// any of its `VkCommandBuffer`s, on whichever thread recorded them, counted at the call.
        /// A field of `WorkCounts`, for the lanes: not part of the API.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct CommandsReplayed {
            $(
                #[doc = concat!("The calls of `", stringify!($v0), "`.")]
                pub $f0: u64,
            )*
            $(
                #[doc = concat!("The calls of `", stringify!($v3), "`.")]
                pub $f3: u64,
            )*
            $(
                #[doc = concat!("The calls of `", stringify!($ve), "`.")]
                pub $fe: u64,
            )*
        }

        impl CommandsReplayed {
            /// The counts `counts` holds, one per command at its index.
            fn read(counts: &[AtomicU64; Command::COUNT]) -> CommandsReplayed {
                CommandsReplayed {
                    $($f0: counts[Command::$v0 as usize].load(Ordering::Relaxed),)*
                    $($f3: counts[Command::$v3 as usize].load(Ordering::Relaxed),)*
                    $($fe: counts[Command::$ve as usize].load(Ordering::Relaxed),)*
                }
            }
        }

        /// Each slot's commands, as their devices' drivers gave them; null where none is stored.
        struct Real {
            $($f0: [AtomicPtr<c_void>; SLOTS],)*
            $($f3: [AtomicPtr<c_void>; SLOTS],)*
            $($fe: [AtomicPtr<c_void>; SLOTS],)*
        }

        static REAL: Real = Real {
            $($f0: [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS],)*
            $($f3: [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS],)*
            $($fe: [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS],)*
        };

        commands!(@thunks $($v0 $f0 $n0 ($($a0: $t0),*) $(-> $r0)?;)*);
        commands!(@thunks $($v3 $f3 $n3 ($($a3: $t3),*) $(-> $r3)?;)*);
        commands!(@thunks $($ve $fe $ne ($($ae: $te),*) $(-> $re)?;)*);

        /// Slot `S`'s entry point for `command`.
        pub(super) fn thunk<const S: usize>(command: Command) -> *const c_void {
            match command {
                $(Command::$v0 => $f0::<S> as *const c_void,)*
                $(Command::$v3 => $f3::<S> as *const c_void,)*
                $(Command::$ve => $fe::<S> as *const c_void,)*
            }
        }

        /// Store `real` as slot `index`'s `command`.
        pub(super) fn store(index: usize, command: Command, real: *const c_void) {
            let real = real.cast_mut();
            match command {
                $(Command::$v0 => REAL.$f0[index].store(real, Ordering::Release),)*
                $(Command::$v3 => REAL.$f3[index].store(real, Ordering::Release),)*
                $(Command::$ve => REAL.$fe[index].store(real, Ordering::Release),)*
            }
        }

        /// Forget slot `index`'s commands.
        pub(super) fn clear(index: usize) {
            $(REAL.$f0[index].store(std::ptr::null_mut(), Ordering::Release);)*
            $(REAL.$f3[index].store(std::ptr::null_mut(), Ordering::Release);)*
            $(REAL.$fe[index].store(std::ptr::null_mut(), Ordering::Release);)*
        }

        /// Route the tables' commands through slot `S`: each one's driver's entry point stored,
        /// the slot's in its place.
        pub(super) fn route_tables<const S: usize>(
            v1_0: &mut ash::DeviceFnV1_0,
            v1_3: &mut ash::DeviceFnV1_3,
        ) {
            $(
                store(S, Command::$v0, v1_0.$f0 as *const c_void);
                v1_0.$f0 = $f0::<S>;
            )*
            $(
                store(S, Command::$v3, v1_3.$f3 as *const c_void);
                v1_3.$f3 = $f3::<S>;
            )*
        }
    };
    (@thunks $($v:ident $f:ident $n:literal ($($a:ident: $t:ty),*) $(-> $r:ty)?;)*) => {
        $(
            /// The command through slot `S` (the module docs).
            unsafe extern "system" fn $f<const S: usize>(
                command_buffer: vk::CommandBuffer,
                $($a: $t,)*
            ) $(-> $r)? {
                COUNTS[S][Command::$v as usize].fetch_add(1, Ordering::Relaxed);
                let real = REAL.$f[S].load(Ordering::Acquire);
                if real.is_null() {
                    missing($n);
                }
                // SAFETY: the slot's device's driver gave this command (`store`), of this
                // signature; the caller's arguments are as the command takes them.
                unsafe {
                    let real = std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "system" fn(vk::CommandBuffer, $($t,)*) $(-> $r)?,
                    >(real);
                    real(command_buffer, $($a,)*)
                }
            }
        )*
    };
}

commands! {
    v1_0 {
        BeginCommandBuffer begin_command_buffer c"vkBeginCommandBuffer"
            (info: *const vk::CommandBufferBeginInfo<'_>) -> vk::Result;
        EndCommandBuffer end_command_buffer c"vkEndCommandBuffer" () -> vk::Result;
        CmdBindPipeline cmd_bind_pipeline c"vkCmdBindPipeline"
            (bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline);
        CmdSetViewport cmd_set_viewport c"vkCmdSetViewport"
            (first: u32, count: u32, viewports: *const vk::Viewport);
        CmdSetScissor cmd_set_scissor c"vkCmdSetScissor"
            (first: u32, count: u32, scissors: *const vk::Rect2D);
        CmdSetDepthBias cmd_set_depth_bias c"vkCmdSetDepthBias"
            (constant: f32, clamp: f32, slope: f32);
        CmdSetBlendConstants cmd_set_blend_constants c"vkCmdSetBlendConstants"
            (constants: *const [f32; 4]);
        CmdSetStencilCompareMask cmd_set_stencil_compare_mask c"vkCmdSetStencilCompareMask"
            (faces: vk::StencilFaceFlags, mask: u32);
        CmdSetStencilWriteMask cmd_set_stencil_write_mask c"vkCmdSetStencilWriteMask"
            (faces: vk::StencilFaceFlags, mask: u32);
        CmdSetStencilReference cmd_set_stencil_reference c"vkCmdSetStencilReference"
            (faces: vk::StencilFaceFlags, reference: u32);
        CmdBindDescriptorSets cmd_bind_descriptor_sets c"vkCmdBindDescriptorSets"
            (
                bind_point: vk::PipelineBindPoint,
                layout: vk::PipelineLayout,
                first: u32,
                count: u32,
                sets: *const vk::DescriptorSet,
                offset_count: u32,
                offsets: *const u32
            );
        CmdBindIndexBuffer cmd_bind_index_buffer c"vkCmdBindIndexBuffer"
            (buffer: vk::Buffer, offset: vk::DeviceSize, index_type: vk::IndexType);
        CmdBindVertexBuffers cmd_bind_vertex_buffers c"vkCmdBindVertexBuffers"
            (first: u32, count: u32, buffers: *const vk::Buffer, offsets: *const vk::DeviceSize);
        CmdDraw cmd_draw c"vkCmdDraw"
            (vertices: u32, instances: u32, first_vertex: u32, first_instance: u32);
        CmdDrawIndexed cmd_draw_indexed c"vkCmdDrawIndexed"
            (indices: u32, instances: u32, first_index: u32, vertex_offset: i32, first_instance: u32);
        CmdDrawIndirect cmd_draw_indirect c"vkCmdDrawIndirect"
            (buffer: vk::Buffer, offset: vk::DeviceSize, count: u32, stride: u32);
        CmdDrawIndexedIndirect cmd_draw_indexed_indirect c"vkCmdDrawIndexedIndirect"
            (buffer: vk::Buffer, offset: vk::DeviceSize, count: u32, stride: u32);
        CmdDispatch cmd_dispatch c"vkCmdDispatch" (x: u32, y: u32, z: u32);
        CmdDispatchIndirect cmd_dispatch_indirect c"vkCmdDispatchIndirect"
            (buffer: vk::Buffer, offset: vk::DeviceSize);
        CmdCopyBuffer cmd_copy_buffer c"vkCmdCopyBuffer"
            (src: vk::Buffer, dst: vk::Buffer, count: u32, regions: *const vk::BufferCopy);
        CmdCopyImage cmd_copy_image c"vkCmdCopyImage"
            (
                src: vk::Image,
                src_layout: vk::ImageLayout,
                dst: vk::Image,
                dst_layout: vk::ImageLayout,
                count: u32,
                regions: *const vk::ImageCopy
            );
        CmdBlitImage cmd_blit_image c"vkCmdBlitImage"
            (
                src: vk::Image,
                src_layout: vk::ImageLayout,
                dst: vk::Image,
                dst_layout: vk::ImageLayout,
                count: u32,
                regions: *const vk::ImageBlit,
                filter: vk::Filter
            );
        CmdCopyBufferToImage cmd_copy_buffer_to_image c"vkCmdCopyBufferToImage"
            (
                src: vk::Buffer,
                dst: vk::Image,
                dst_layout: vk::ImageLayout,
                count: u32,
                regions: *const vk::BufferImageCopy
            );
        CmdCopyImageToBuffer cmd_copy_image_to_buffer c"vkCmdCopyImageToBuffer"
            (
                src: vk::Image,
                src_layout: vk::ImageLayout,
                dst: vk::Buffer,
                count: u32,
                regions: *const vk::BufferImageCopy
            );
        CmdUpdateBuffer cmd_update_buffer c"vkCmdUpdateBuffer"
            (dst: vk::Buffer, offset: vk::DeviceSize, size: vk::DeviceSize, data: *const c_void);
        CmdFillBuffer cmd_fill_buffer c"vkCmdFillBuffer"
            (dst: vk::Buffer, offset: vk::DeviceSize, size: vk::DeviceSize, data: u32);
        CmdClearColorImage cmd_clear_color_image c"vkCmdClearColorImage"
            (
                image: vk::Image,
                layout: vk::ImageLayout,
                color: *const vk::ClearColorValue,
                count: u32,
                ranges: *const vk::ImageSubresourceRange
            );
        CmdClearDepthStencilImage cmd_clear_depth_stencil_image c"vkCmdClearDepthStencilImage"
            (
                image: vk::Image,
                layout: vk::ImageLayout,
                value: *const vk::ClearDepthStencilValue,
                count: u32,
                ranges: *const vk::ImageSubresourceRange
            );
        CmdClearAttachments cmd_clear_attachments c"vkCmdClearAttachments"
            (
                count: u32,
                attachments: *const vk::ClearAttachment,
                rect_count: u32,
                rects: *const vk::ClearRect
            );
        CmdPipelineBarrier cmd_pipeline_barrier c"vkCmdPipelineBarrier"
            (
                src: vk::PipelineStageFlags,
                dst: vk::PipelineStageFlags,
                flags: vk::DependencyFlags,
                memory_count: u32,
                memory: *const vk::MemoryBarrier<'_>,
                buffer_count: u32,
                buffers: *const vk::BufferMemoryBarrier<'_>,
                image_count: u32,
                images: *const vk::ImageMemoryBarrier<'_>
            );
        CmdBeginQuery cmd_begin_query c"vkCmdBeginQuery"
            (pool: vk::QueryPool, query: u32, flags: vk::QueryControlFlags);
        CmdEndQuery cmd_end_query c"vkCmdEndQuery" (pool: vk::QueryPool, query: u32);
        CmdResetQueryPool cmd_reset_query_pool c"vkCmdResetQueryPool"
            (pool: vk::QueryPool, first: u32, count: u32);
        CmdWriteTimestamp cmd_write_timestamp c"vkCmdWriteTimestamp"
            (stage: vk::PipelineStageFlags, pool: vk::QueryPool, query: u32);
        CmdCopyQueryPoolResults cmd_copy_query_pool_results c"vkCmdCopyQueryPoolResults"
            (
                pool: vk::QueryPool,
                first: u32,
                count: u32,
                dst: vk::Buffer,
                offset: vk::DeviceSize,
                stride: vk::DeviceSize,
                flags: vk::QueryResultFlags
            );
        CmdPushConstants cmd_push_constants c"vkCmdPushConstants"
            (
                layout: vk::PipelineLayout,
                stages: vk::ShaderStageFlags,
                offset: u32,
                size: u32,
                values: *const c_void
            );
        CmdBeginRenderPass cmd_begin_render_pass c"vkCmdBeginRenderPass"
            (begin: *const vk::RenderPassBeginInfo<'_>, contents: vk::SubpassContents);
        CmdEndRenderPass cmd_end_render_pass c"vkCmdEndRenderPass" ();
        CmdExecuteCommands cmd_execute_commands c"vkCmdExecuteCommands"
            (count: u32, buffers: *const vk::CommandBuffer);
    }
    v1_3 {
        CmdSetCullMode cmd_set_cull_mode c"vkCmdSetCullMode" (mode: vk::CullModeFlags);
        CmdSetFrontFace cmd_set_front_face c"vkCmdSetFrontFace" (face: vk::FrontFace);
        CmdSetPrimitiveTopology cmd_set_primitive_topology c"vkCmdSetPrimitiveTopology"
            (topology: vk::PrimitiveTopology);
        CmdBindVertexBuffers2 cmd_bind_vertex_buffers2 c"vkCmdBindVertexBuffers2"
            (
                first: u32,
                count: u32,
                buffers: *const vk::Buffer,
                offsets: *const vk::DeviceSize,
                sizes: *const vk::DeviceSize,
                strides: *const vk::DeviceSize
            );
        CmdSetDepthTestEnable cmd_set_depth_test_enable c"vkCmdSetDepthTestEnable"
            (enable: vk::Bool32);
        CmdSetDepthWriteEnable cmd_set_depth_write_enable c"vkCmdSetDepthWriteEnable"
            (enable: vk::Bool32);
        CmdSetDepthCompareOp cmd_set_depth_compare_op c"vkCmdSetDepthCompareOp"
            (op: vk::CompareOp);
        CmdSetStencilTestEnable cmd_set_stencil_test_enable c"vkCmdSetStencilTestEnable"
            (enable: vk::Bool32);
        CmdSetStencilOp cmd_set_stencil_op c"vkCmdSetStencilOp"
            (
                faces: vk::StencilFaceFlags,
                fail: vk::StencilOp,
                pass: vk::StencilOp,
                depth_fail: vk::StencilOp,
                compare: vk::CompareOp
            );
        CmdSetPrimitiveRestartEnable cmd_set_primitive_restart_enable
            c"vkCmdSetPrimitiveRestartEnable" (enable: vk::Bool32);
    }
    extension {
        CmdPushDescriptorSet cmd_push_descriptor_set c"vkCmdPushDescriptorSetKHR"
            (
                bind_point: vk::PipelineBindPoint,
                layout: vk::PipelineLayout,
                set: u32,
                count: u32,
                writes: *const vk::WriteDescriptorSet<'_>
            );
        CmdSetColorWriteMask cmd_set_color_write_mask c"vkCmdSetColorWriteMaskEXT"
            (first: u32, count: u32, masks: *const vk::ColorComponentFlags);
        CmdBindIndexBuffer2 cmd_bind_index_buffer2 c"vkCmdBindIndexBuffer2"
            (
                buffer: vk::Buffer,
                offset: vk::DeviceSize,
                size: vk::DeviceSize,
                index_type: vk::IndexType
            );
        CmdDrawMeshTasks cmd_draw_mesh_tasks c"vkCmdDrawMeshTasksEXT" (x: u32, y: u32, z: u32);
        CmdDrawMeshTasksIndirect cmd_draw_mesh_tasks_indirect c"vkCmdDrawMeshTasksIndirectEXT"
            (buffer: vk::Buffer, offset: vk::DeviceSize, count: u32, stride: u32);
        CmdDrawMeshTasksIndirectCount cmd_draw_mesh_tasks_indirect_count
            c"vkCmdDrawMeshTasksIndirectCountEXT"
            (
                buffer: vk::Buffer,
                offset: vk::DeviceSize,
                count_buffer: vk::Buffer,
                count_offset: vk::DeviceSize,
                max_count: u32,
                stride: u32
            );
        CmdExecuteGeneratedCommands cmd_execute_generated_commands
            c"vkCmdExecuteGeneratedCommandsEXT"
            (preprocessed: vk::Bool32, info: *const GeneratedCommandsInfo);
        CmdPreprocessGeneratedCommands cmd_preprocess_generated_commands
            c"vkCmdPreprocessGeneratedCommandsEXT"
            (info: *const GeneratedCommandsInfo, state: vk::CommandBuffer);
    }
}
