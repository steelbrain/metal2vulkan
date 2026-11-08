//! The in-place check's draw modules: valid on every committed profile (`G-val`).

use super::*;
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::testing::assert_valid;
use std::path::PathBuf;

/// On every committed device both stages are valid and built.
#[test]
fn the_modules_are_valid_on_every_committed_device() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut built = 0;
    for entry in std::fs::read_dir(&dir).expect("profiles") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let device = Device::new(desc, DevicePolicy::default()).expect("device");
        let profile = device.profile();
        assert_valid(merge_vertex_typed_module().expect("vertex"), profile);
        assert_valid(fragment_typed_module().expect("fragment"), profile);
        let modules = check_draw_modules(&device).expect("modules");
        assert!(!modules.vertex.is_empty() && !modules.fragment.is_empty());
        built += 1;
    }
    assert!(built > 0);
}
