use ash::vk;

mod acceptance;
mod buffers;
mod compute;
mod device;
mod enables;
mod graphics_pipeline;
mod ignored_drivers;
mod loader_word;
mod multisample;
mod pipeline;
mod plan_resources;
mod profiles;
mod render;
mod runtime;
mod runtime_acceptance;
mod runtime_bounds;
mod runtime_depth;
mod runtime_multisample;
mod runtime_probe;
mod runtime_render;
mod runtime_restriction;
mod runtime_textures;
pub use acceptance::create_acceptance_pipeline;
pub use compute::execute_engine;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub use compute::opened_worker_device;
pub use device::{capture_formats, echo_validation_messages, venue_check};
pub use profiles::{requested_restriction, restrict_devices_to};
pub use render::execute_engine_render;
pub use runtime::execute_runtime;
pub use runtime_acceptance::RuntimeKernels;
pub use runtime_probe::probe_runtime;
pub use runtime_render::execute_runtime_render;

/// Whether the profiles layer can present committed profile `stem` ([`profiles::presentable`]).
pub fn check_presentable(stem: &str) -> Result<(), String> {
    profiles::presentable(stem).map(drop)
}

// Metal render encoders define clockwise primitives as front-facing unless the application
// overrides the dynamic winding state. Authored graphics cases exercise that default contract.
pub(super) const METAL_DEFAULT_FRONT_FACE: vk::FrontFace = vk::FrontFace::CLOCKWISE;
