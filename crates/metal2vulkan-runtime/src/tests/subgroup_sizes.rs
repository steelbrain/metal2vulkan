//! Render stages at the subgroup size the translator pins.
//!
//! Metal runs every simdgroup 32 wide, and the translator lowers a vertex or fragment stage's
//! SIMD-group operations for that width, pinning it (`StageModule::required_subgroup_size`) on a
//! device that lists the stage in `requiredSubgroupSizeStages` with 32 between its
//! `minSubgroupSize` and `maxSubgroupSize`. Such a stage is created with
//! `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` of that size chained, as a kernel's is;
//! a stage the translator pins no size for has nothing chained and runs at the device's own size.
//!
//! - On the host: the create infos of a vertex stage with no pinned size and a fragment stage
//!   pinned at 32 chain nothing to the first and the one size structure, of 32, to the second.
//! - Under a device cfg, on a device whose fragment stage takes a pinned size of 32:
//!   `fragment_lane` (the owned fixture `engine/gsm2_fragment_lanes.ll`) writes each pixel's
//!   `[[thread_index_in_simdgroup]]` into an `R32Uint` `SIDE` square that two triangles of
//!   `fragment_lanes_vertex` cover, and every lane is below 32, as on Metal. A device whose
//!   fragment stage takes no such size (its stage unlisted, or 32 outside its sizes) skips
//!   visibly: the translator pins nothing there.

use crate::render::{stage_infos, subgroup_sizes, Module};
use ash::vk;
use std::ffi::CString;

/// A pinned stage's create info chains its size, and an unpinned stage's chains nothing.
#[test]
fn a_render_stage_is_made_at_the_subgroup_size_the_translator_pins() {
    let module = |stage, subgroup_size| Module {
        module: vk::ShaderModule::null(),
        entry: CString::new("main").expect("an entry point"),
        stage,
        flags: vk::PipelineShaderStageCreateFlags::empty(),
        subgroup_size,
        constants: Vec::new(),
        values: Vec::new(),
    };
    let stages = [
        module(vk::ShaderStageFlags::VERTEX, None),
        module(vk::ShaderStageFlags::FRAGMENT, Some(32)),
    ];
    let specializations = [vk::SpecializationInfo::default(); 2];
    let mut subgroups = subgroup_sizes(&stages);
    let infos = stage_infos(&stages, &specializations, &mut subgroups);
    assert_eq!(infos.len(), 2);
    assert!(infos[0].p_next.is_null(), "the vertex stage pins no size");
    let chained = infos[1]
        .p_next
        .cast::<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'_>>();
    assert!(!chained.is_null(), "the fragment stage chains its size");
    // SAFETY: the fragment stage's chain is the one structure `stage_infos` pushed, which
    // `subgroups` holds alive and nothing writes.
    let chained = unsafe { &*chained };
    assert!(
        chained.s_type
            == vk::StructureType::PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        "the chained structure is the required subgroup size's"
    );
    assert_eq!(chained.required_subgroup_size, 32);
    assert!(chained.p_next.is_null(), "one structure chained");
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        floats, library, pass, read, run, state, target, Buffers, SIDE,
    };
    use crate::{
        ColorAttachmentDescriptor, Error, FunctionDescriptor, PixelFormat, PrimitiveType,
        RenderPipelineDescriptor, StageSource, TextureUsage,
    };
    use metal2vulkan_devtools::engine_dir;

    /// On a device whose fragment stage takes a pinned size of 32, every fragment lane is below
    /// 32, as Metal's 32-wide simdgroups give (the module docs); elsewhere the test skips.
    #[test]
    fn every_device_whose_fragment_stage_pins_32_runs_its_fragments_32_wide() {
        let Some(lanes) = library(&engine_dir().join("gsm2_fragment_lanes.ll")) else {
            return;
        };
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                lanes.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("fragment_lanes_vertex"));
        d.fragment = Some(function("fragment_lane"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        each_device("fragment_subgroup_size", |name, _, device| {
            let fragment = device.core().translator.profile().subgroups().fragment;
            if !(fragment.pinnable && fragment.fits(32)) {
                println!(
                    "SKIPPED {name}: its fragment stage takes no pinned subgroup size of 32 \
                     (pinnable {}, sizes {}..={})",
                    fragment.pinnable, fragment.sizes.min, fragment.sizes.max
                );
                return;
            }
            let made = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let positions = buffers.of(
                &device,
                &floats(&[
                    -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
                ]),
            );
            let flat = target(&device, PixelFormat::R32Uint, TextureUsage::RENDER_TARGET);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut encoder = (command_buffer.render_command_encoder(&pass(&flat, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_render_pipeline_state(&made));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let bytes = read(&flat, 4);
            let lanes: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
                .map(|&word| u32::from_le_bytes(word))
                .collect();
            assert_eq!(lanes.len(), (SIDE * SIDE) as usize);
            let highest = lanes.iter().copied().max().unwrap_or(0);
            assert!(
                highest < 32,
                "{name}: a fragment ran at lane {highest}, past Metal's 32-wide simdgroups: {lanes:?}"
            );
            println!(
                "{name}: the fragment stage pinned at 32 (sizes {}..={}); every lane below 32, \
                 the highest {highest}",
                fragment.sizes.min, fragment.sizes.max
            );
        });
    }
}
