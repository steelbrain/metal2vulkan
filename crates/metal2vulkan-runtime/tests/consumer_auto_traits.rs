//! A program's crate holds the runtime's objects across threads at rustc's default recursion
//! limit: this test crate, like a program's, sets none. Each object is held as reims-vgpu holds
//! it: under its `Obj<T: Send + Sync + 'static>`'s bound (one counted reference), an encoder,
//! which Metal's program uses on one thread at a time, as a `Mutex<Option<_>>` under it.
//! rustc proves those bounds by walking each object's fields. A walk deeper than the limit fails
//! this crate's build (`E0275`), as it fails the program's: the walk from an encoder through its
//! device's render lowering slots down to the translator's reflection once did.
//!
//! rustc keeps each bound it proved for the rest of the crate, so an object checked after another
//! that reaches it would ride on the earlier, shallower proof, where a program that holds only
//! the later one walks it in full. Each object is therefore checked in a function of its own,
//! whose `where` clause names it: the environment differs, and rustc proves the object's bounds
//! afresh there.

use std::sync::Mutex;

use metal2vulkan_runtime::{
    ArgumentEncoder, BinaryArchive, BlitCommandEncoder, Buffer, CommandBuffer, CommandBufferError,
    CommandQueue, ComputeCommandEncoder, ComputePipelineState, DegradeNotice, DepthStencilState,
    Device, DynamicLibrary, Error, Event, Fence, Function, Heap, IndirectCommandBuffer,
    IndirectComputeCommand, IndirectRenderCommand, Library, MessageSink, PipelineRecord,
    RasterizationRateMap, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceStateCommandEncoder, SamplerState, SharedEvent, SharedEventHandle, SharedEventListener,
    SharedTextureHandle, Texture, VisibleFunctionTable, WorkCounts,
};

/// `T` held as a program's crate holds it: the bound of reims-vgpu's `platform::Obj<T>`, one
/// counted reference shared across threads.
fn held<T: Send + Sync + 'static>() {}

/// Each object, in a function of its own (the module docs).
macro_rules! held {
    ($($check:ident: $held:ty;)*) => {
        $(
            fn $check()
            where
                $held: Sized,
            {
                held::<$held>();
            }
        )*

        #[test]
        fn every_object_is_held_across_threads_at_the_default_recursion_limit() {
            $(
                $check();
            )*
        }
    };
}

held! {
    render_encoder: Mutex<Option<RenderCommandEncoder>>;
    compute_encoder: Mutex<Option<ComputeCommandEncoder>>;
    blit_encoder: Mutex<Option<BlitCommandEncoder>>;
    resource_state_encoder: Mutex<Option<ResourceStateCommandEncoder>>;
    device: Device;
    command_queue: CommandQueue;
    command_buffer: CommandBuffer;
    buffer: Buffer;
    texture: Texture;
    heap: Heap;
    fence: Fence;
    event: Event;
    shared_event: SharedEvent;
    shared_event_listener: SharedEventListener;
    shared_event_handle: SharedEventHandle;
    shared_texture_handle: SharedTextureHandle;
    library: Library;
    function: Function;
    dynamic_library: DynamicLibrary;
    binary_archive: BinaryArchive;
    compute_pipeline_state: ComputePipelineState;
    render_pipeline_state: RenderPipelineState;
    depth_stencil_state: DepthStencilState;
    sampler_state: SamplerState;
    indirect_command_buffer: IndirectCommandBuffer;
    indirect_render_command: IndirectRenderCommand;
    indirect_compute_command: IndirectComputeCommand;
    render_pass_descriptor: Mutex<RenderPassDescriptor>;
    argument_encoder: ArgumentEncoder;
    visible_function_table: VisibleFunctionTable;
    rasterization_rate_map: RasterizationRateMap;
    pipeline_record: PipelineRecord;
    message_sink: MessageSink;
    error: Error;
    command_buffer_error: CommandBufferError;
    degrade_notice: DegradeNotice;
    work_counts: WorkCounts;
}
