mod candidate_batch;
mod candidate_common;

use metal2vulkan_validation::observation::Backend;

fn main() {
    // Every pipeline this process or a run it starts creates compiles cold through Metal's
    // compiler, so the ceiling times a compile, not a hit in Metal's shader cache.
    metal2vulkan_validation::budgets::turn_driver_shader_caches_off();
    // The batch modes run this executable once per case; every other invocation, the
    // workers' included, is the single-case executor.
    if !candidate_batch::serve(Backend::Moltenvk) {
        candidate_common::main(Backend::Moltenvk);
    }
}
