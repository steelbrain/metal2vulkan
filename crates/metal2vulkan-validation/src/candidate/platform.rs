use ash::vk;

mod acceleration;
mod acceptance;
mod buffers;
mod compute;
mod device;
mod enables;
mod graphics_pipeline;
mod loader_word;
mod multisample;
mod pipeline;
mod plan_resources;
mod profiles;
mod render;
mod runtime_probe;
mod sparse_images;
pub use acceptance::create_acceptance_pipeline;
pub use compute::execute_engine;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub use compute::opened_worker_device;
pub use device::{capture_formats, echo_validation_messages, venue_check};
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use device::{lane_restriction, LaneRestriction};
pub use device::{requested_leg, run_robustness_off};
pub(crate) use device::{runtime_restriction, Restricting, RuntimeRestriction};
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use loader_word::LoaderWord;
pub use profiles::{requested_restriction, restrict_devices_to};
pub use render::execute_engine_render;
pub use runtime_probe::probe_runtime;

/// Whether the profiles layer can present committed profile `stem` ([`profiles::presentable`]).
pub fn check_presentable(stem: &str) -> Result<(), String> {
    profiles::presentable(stem).map(drop)
}

// Metal render encoders define clockwise primitives as front-facing unless the application
// overrides the dynamic winding state. Authored graphics cases exercise that default contract.
pub(super) const METAL_DEFAULT_FRONT_FACE: vk::FrontFace = vk::FrontFace::CLOCKWISE;
