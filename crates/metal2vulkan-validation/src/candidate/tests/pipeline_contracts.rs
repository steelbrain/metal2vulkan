use super::*;

#[test]
fn vulkan_candidates_run_off_macos_alone() {
    if cfg!(target_os = "macos") {
        assert!(validate_host().is_err());
    } else if cfg!(target_os = "linux") {
        assert!(validate_host().is_ok());
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn authored_vulkan_draws_match_metals_default_front_face() {
    assert!(platform::METAL_DEFAULT_FRONT_FACE == ash::vk::FrontFace::CLOCKWISE);
}
