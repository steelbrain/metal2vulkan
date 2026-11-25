use super::*;

const OWNED_STATE: Inheritance = Inheritance {
    pipeline: StateSource::Command,
    buffers: StateSource::Command,
};

const TEST_DESCRIPTOR: CommandDescriptor = CommandDescriptor::Compute {
    dispatch: ComputeDispatch::Both,
    max_buffer_bind_count: 32,
    max_threadgroup_memory_bind_count: 31,
    support_dynamic_attribute_stride: true,
};

mod buffer_views;
mod records;
