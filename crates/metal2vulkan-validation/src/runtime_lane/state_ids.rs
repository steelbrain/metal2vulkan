//! IDs and placement of pipeline and depth/stencil states on the device the cfg selects, under the
//! validation layer, as Metal answers on an M4 with macOS 27.0.1:
//!
//! - A compute or render state made with `supportIndirectCommandBuffers` takes the ID its
//!   placement names (`PipelineOptions::placement`, Metal's `resourceIndex`), in the one space
//!   compute and render states share: two live states placed at one ID both answer it. The
//!   index's low 32 bits are kept: `2^32 + 5` is 5, and `2^32`, as 0, is plain allocation.
//! - Plain allocation does not count a placed ID.
//! - A state without that support, and a tile state, take no ID: they answer 0 whatever the
//!   placement, 16384 included.
//! - A placement at 16384 or more after the cut, which ended Metal's process, is refused typed.
//! - A depth/stencil state's ID below Apple10 is plain Metal's answer, 0 (macOS 27.0.1). Metal's
//!   own API validation stops the call, which is not what plain Metal does.
//!
//! - A state answers its descriptor's label, and one made from it with additional binary
//!   functions keeps it.
//!
//! The states are of the owned fixtures `icb_replay_template` (compute),
//! `icb_render_replay_template` (render, rasterization disabled) and `cfit_tile_imageblock`
//! (tile).

use super::tile_states::on_device;
use crate::air_library::AirLibrary;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ComputePipelineState, Device, Error, GpuFamily, IdPlacement, IndirectRule, Invalid,
    PipelineOption, PipelineOptions, Problem, RenderPipelineFunctionsDescriptor,
    RenderPipelineState, ResourceId, RuntimeGap,
};

/// The placement at `id`.
fn at(id: u64) -> IdPlacement {
    IdPlacement::At(ResourceId(id))
}

/// Options that place the state at `placement`.
fn placed(placement: IdPlacement) -> PipelineOptions {
    let mut options = PipelineOptions::new(PipelineOption::NONE);
    options.placement = placement;
    options
}

/// The function `name` of the owned fixture `stem` on `device`, as a stage source.
fn source(device: &Device, stem: &str, name: &str) -> StageSource {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    StageSource::Function(function.function_descriptor())
}

/// The compute template's state, made with `supportIndirectCommandBuffers` as `support` says,
/// at `placement`.
fn compute(
    device: &Device,
    support: bool,
    placement: IdPlacement,
) -> Result<ComputePipelineState, Error> {
    let mut descriptor = ComputePipelineDescriptor::new(source(
        device,
        "icb_replay_template",
        "icb_replay_template",
    ));
    descriptor.support_indirect_command_buffers = support;
    device.new_compute_pipeline_state(&descriptor, &placed(placement))
}

/// The render template's state (rasterization disabled), made with
/// `supportIndirectCommandBuffers` as `support` says, at `placement`.
fn render(
    device: &Device,
    support: bool,
    placement: IdPlacement,
) -> Result<RenderPipelineState, Error> {
    let mut descriptor = RenderPipelineDescriptor::new(source(
        device,
        "icb_render_replay_template",
        "icb_render_replay_template",
    ));
    descriptor.rasterization_enabled = false;
    descriptor.support_indirect_command_buffers = support;
    device.new_render_pipeline_state(&descriptor, &placed(placement))
}

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// Compute and render states made with `supportIndirectCommandBuffers` answer the ID their
/// placement names, in one space, two live states at one ID included; the index's low 32 bits
/// are kept; plain allocation does not count placed IDs; states without the support, and tile
/// states, answer 0 (probes 6, 6b, 6d and 6e).
#[test]
fn pipeline_states_take_the_id_their_placement_names() {
    on_device(|device| {
        let states = [
            compute(device, true, at(400)).unwrap(),
            compute(device, true, at(420)).unwrap(),
            compute(device, true, at(420)).unwrap(),
            compute(device, true, at(16383)).unwrap(),
            compute(device, true, at((1 << 32) + 5)).unwrap(),
            compute(device, true, at((1 << 33) + 7)).unwrap(),
        ];
        assert_eq!(
            states.each_ref().map(ComputePipelineState::gpu_resource_id),
            [400, 420, 420, 16383, 5, 7].map(ResourceId)
        );
        let renders = [
            render(device, true, at(402)).unwrap(),
            render(device, true, at(420)).unwrap(),
        ];
        assert_eq!(
            renders.each_ref().map(RenderPipelineState::gpu_resource_id),
            [402, 420].map(ResourceId)
        );
        // 0 and `2^32` are plain allocation: the lowest free ID, from 1.
        let plain = [
            compute(device, true, at(0)).unwrap(),
            compute(device, true, at(1 << 32)).unwrap(),
        ];
        assert_eq!(
            plain.each_ref().map(ComputePipelineState::gpu_resource_id),
            [1, 2].map(ResourceId)
        );
        let third = render(device, true, at(1 << 32)).unwrap();
        assert_eq!(third.gpu_resource_id(), ResourceId(3));
        // Plain allocation does not count a placed ID: it hands out 4 while a state placed at 4
        // lives (probe 6e).
        let fourth = compute(device, true, at(4)).unwrap();
        assert_eq!(
            [
                fourth.gpu_resource_id(),
                render(device, true, IdPlacement::Any)
                    .unwrap()
                    .gpu_resource_id()
            ],
            [4, 4].map(ResourceId)
        );

        for placement in [at(401), at(16384)] {
            assert_eq!(
                compute(device, false, placement).unwrap().gpu_resource_id(),
                ResourceId(0)
            );
            assert_eq!(
                render(device, false, placement).unwrap().gpu_resource_id(),
                ResourceId(0)
            );
        }
        let mut tile = TileRenderPipelineDescriptor::new(source(
            device,
            "cfit_tile_imageblock",
            "cfit_ib_tile_clear",
        ));
        tile.color_attachment_formats[0] = Some(PixelFormat::R32Uint);
        let tile = device
            .new_tile_render_pipeline_state(&tile, &placed(at(404)))
            .unwrap();
        assert_eq!(tile.gpu_resource_id(), ResourceId(0));
    });
}

/// A pipeline state answers the label its creation was given (`PipelineOptions::label`, the
/// descriptor's), nil without one and an empty one as given, and a state made from it with
/// additional binary functions keeps it, as Metal's compute, render and tile states do.
#[test]
fn pipeline_states_answer_their_descriptors_label() {
    on_device(|device| {
        let labelled = |label: Option<&str>| {
            let mut options = PipelineOptions::new(PipelineOption::NONE);
            options.label = label.map(str::to_owned);
            options
        };
        let kernel = ComputePipelineDescriptor::new(source(
            device,
            "icb_replay_template",
            "icb_replay_template",
        ));
        let compute = |label| {
            device
                .new_compute_pipeline_state(&kernel, &labelled(label))
                .unwrap()
        };
        assert_eq!(compute(Some("cp-label")).label(), Some("cp-label"));
        assert_eq!(compute(Some("")).label(), Some(""));
        assert_eq!(compute(None).label(), None);
        let added = compute(Some("base-label"))
            .new_compute_pipeline_state_with_additional_binary_functions(&[])
            .unwrap();
        assert_eq!(added.label(), Some("base-label"));

        let mut descriptor = RenderPipelineDescriptor::new(source(
            device,
            "icb_render_replay_template",
            "icb_render_replay_template",
        ));
        descriptor.rasterization_enabled = false;
        let render = device
            .new_render_pipeline_state(&descriptor, &labelled(Some("rp-label")))
            .unwrap();
        assert_eq!(render.label(), Some("rp-label"));
        let added = render
            .new_render_pipeline_state_with_additional_binary_functions(
                &RenderPipelineFunctionsDescriptor::default(),
            )
            .unwrap();
        assert_eq!(added.label(), Some("rp-label"));
        let unlabelled = device
            .new_render_pipeline_state(&descriptor, &labelled(None))
            .unwrap();
        assert_eq!(unlabelled.label(), None);

        let mut tile = TileRenderPipelineDescriptor::new(source(
            device,
            "cfit_tile_imageblock",
            "cfit_ib_tile_clear",
        ));
        tile.color_attachment_formats[0] = Some(PixelFormat::R32Uint);
        let tile = device
            .new_tile_render_pipeline_state(&tile, &labelled(Some("tile-label")))
            .unwrap();
        assert_eq!(tile.label(), Some("tile-label"));
    });
}

/// A placement at 16384 or more after its index's cut to 32 bits ended Metal's process for a
/// compute state made with `supportIndirectCommandBuffers` (probes 6c and 6d): it is refused
/// typed, naming the kept index, for compute and render states alike.
#[test]
fn placements_past_the_pipeline_limit_are_refused_typed() {
    on_device(|device| {
        let refused = |id: u64| Problem::ResourceId { id, limit: 16384 };
        for (index, kept) in [
            (16384, 16384),
            (65536, 65536),
            (1 << 31, 1 << 31),
            ((1 << 32) - 1, (1 << 32) - 1),
            ((1 << 32) + 16384, 16384),
        ] {
            assert_eq!(
                compute(device, true, at(index)).unwrap_err(),
                invalid("new_compute_pipeline_state", refused(kept))
            );
        }
        assert_eq!(
            render(device, true, at(16384)).unwrap_err(),
            invalid("new_render_pipeline_state", refused(16384))
        );
    });
}

/// A depth/stencil state's `gpu_resource_id` below Apple10 is plain Metal's answer: 0 (our probe on
/// an M3 Max under macOS 27.0.1; Metal's own API validation stops the call, "Obtaining the
/// resourceID of a DepthStencilState object requires support for MTLGPUFamilyApple10", probe 6,
/// which is not what plain Metal does). It is a gap on a device that claims Apple10. Every
/// descriptor answers alike, a depth one and a depth-and-stencil one among them.
#[test]
fn a_depth_stencil_states_id_below_apple10_is_plain_metal_s_answer() {
    on_device(|device| {
        let mut stencilled = DepthStencilDescriptor::default();
        stencilled.front_face_stencil.stencil_compare_function = CompareFunction::Equal;
        for descriptor in [DepthStencilDescriptor::default(), stencilled] {
            let state = device.new_depth_stencil_state(&descriptor).unwrap();
            let expected = if device.supports_family(GpuFamily::Apple10.raw()) {
                Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands(
                    IndirectRule::CommandRenderStates,
                )))
            } else {
                Ok(ResourceId(0))
            };
            assert_eq!(state.gpu_resource_id(), expected, "{descriptor:?}");
        }
    });
}
