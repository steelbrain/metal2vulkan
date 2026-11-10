//! Whether a device's `Depth32Float` depth image holds a denormal and its depth test compares
//! one as Metal's (the parity rule: Metal keeps one), and how its depth test compares a stored
//! negative value or NaN, on the device the cfg selects, under the validation layer.
//!
//! **Metal's answer**: an Apple GPU's `Depth32Float` texture keeps a denormal through a blit in
//! and out and a shader's read (an M4 under macOS 27.0.1); MoltenVK on the M4, whose clears,
//! copies, draws and depth tests are Metal's, keeps one a clear, a copy or the CPU stores, stores 0
//! for a draw at a denormal depth, and its depth test compares a positive one above zero and a
//! negative one not below zero (`DriverFact::DepthTestKeepsDenormals`).
//!
//! Each of four denormals (the smallest and largest of each sign) is stored into a 4x4
//! single-sample `Depth32Float` render target by each way a program stores depth: a pass's clear,
//! `replace_region`, a blit from a buffer, and a draw at that depth (compare `Always`, depth
//! writes on; the owned fixture `rt6_render_state`'s fullscreen triangle); the negative ones only
//! by the CPU and the blit (a draw's is clipped, a clear's outside Vulkan's clear range). Each
//! stored texture is then read two ways that do not share a path:
//!
//! - **read**: `get_bytes`, the texel's bits as a transfer reads them;
//! - **tested**: a draw at depth 0 that loads the depth and tests it, `Less` for a positive
//!   denormal and `Greater` for a negative one, writing white where it passes: Metal's passes
//!   against a positive one and fails against a negative one.
//!
//! Each prints a `DEPTH-DENORMAL` line. Checked: the two controls, the smallest normal and 0.5,
//! read and pass as given on every device; a draw's denormal stores zero, as Metal's, but where
//! the drawing stage leaves a `float` denormal to the device (`device-denormals`), which may keep
//! it; where the driver has `DriverFact::DepthImageKeepsDenormals` the clear's, the CPU's and
//! the blit's read back as given, but a negative one the upload clamps to 0 on a device whose
//! uploads clamp (`degrade_notices::clamps_depth_uploads`); where it has
//! `DriverFact::DepthTestKeepsDenormals` too, each one read back as given tests as Metal's; and `depth-denormals` is announced exactly where the
//! driver lacks either fact.
//!
//! **Signs**: Metal's depth test compares the fragment's depth with the stored value's magnitude,
//! as the integers of their low 31 bits: a stored -0.5 passes `Equal` for a fragment at 0.5, and a
//! stored NaN orders above +inf (our probe `depthkeep`, the M3 Max under macOS 27.0.1: 5 compare
//! functions, fragments at 0, 0.5 and 1, 18 stored values, 270 of 270 outcomes; its texel keeps
//! every value's bits through copies, reads, samples and a pass's load and store without depth
//! writes, and `sample_compare` is the IEEE compare of the reference with the texel, neither
//! clamped, 720 of 720). Vulkan's compares the stored value as it is: a device's depth test is
//! Metal's only where `DriverFact::DepthTestComparesMagnitudes` holds (MoltenVK), and elsewhere a
//! write that may store such a value in a render target's depth announces `signed-depth-test`.
//! Each of the probe's 18 values is stored by `replace_region` and by a blit from a buffer, read by
//! `get_bytes`, a kernel's read and a nearest sample, compared by `sample_compare` against five
//! references under the eight functions, and depth-tested by the probe's 15 draws, each printing a
//! `PROBED-DEPTH` line where it differs from Metal's. Checked on a device that keeps such a value:
//! every read, sample and comparison Metal's wherever the value was stored as given, and every
//! depth test too where the device's test is Metal's own; there, a pass with depth writes over a
//! texture of such values stores the draw's depth where it passed and every other texel without
//! its sign, as Metal's (our probe `depthstore`, the M3 Max under macOS 27.0.1), and a pass that
//! sets a state with depth writes but draws under none, which Metal's stores without the signs,
//! announces `signed-depth-test`; on every such device a pass without depth writes keeps every
//! value. An ordinary depth loads in place: no kept image or stand-in made (`ORDINARY-DEPTH`
//! prints the command buffers its passes begin and submit).
//!
//! **A new target's pattern**: a new `Private` `Depth32Float` or `Depth32Float_Stencil8` render
//! target of 16² reads Metal's pattern, a NaN with the sign set (all ones); a pass that loads it
//! before anything wrote it and draws at 0.5 with depth writes draws everywhere under `Less` and
//! `NotEqual` and nowhere under `Greater` and `Equal`, and stores every texel no fragment wrote as
//! `0x7fffffff` (our probe `fd32`, the M3 Max under macOS 27.0.1, 64² drawn over half). Checked on
//! a device that writes the pattern: where the device's test is Metal's own, every draw and store
//! Metal's; elsewhere each texel no fragment wrote keeps the pattern, with or without its sign,
//! and each pass that loads it announces `signed-depth-test`. On a device that may clamp the
//! pattern, each target reads zero and draws and stores as over a zero depth, announced
//! (`fresh-pattern-zeroed`) at its first load. A new target cleared first loads its clear,
//! announcing neither (`FRESH-SIGNED-DEPTH` lines). On a device without Metal's compression
//! (`DeviceProperties::lossless_compression` false) a new target holds what its memory held, so
//! its draws and stores are unchecked and its passes announce neither.

use super::degrade_notices::{assert_made_once, assert_once_allowing, bound_value_kinds};
use super::degrade_notices::{clamps_depth_uploads, forced, loses_depth_denormals};
use super::degrade_notices::{counts, signs_depth_tests};
use super::degrade_notices::{made_compute_kinds, made_render_kinds, shared_in_system_memory};
use super::depth_compares::run;
use super::render_state::library;
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device,
    validated_device_with,
};
use super::{Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::{DriverFact, VkFeature};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandBufferStatus, CommandQueue, ComputePipelineState, CpuCacheMode,
    Degradation, Device, DeviceOptions, Error, HazardTrackingMode, Library, LoadAction, Origin,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, Size, StorageMode, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// The kind a depth image that may store a denormal otherwise than Metal's takes.
pub(super) const DEPTH_DENORMALS: [Degradation; 1] = [Degradation::DepthDenormals];

/// The kind a render target's depth that may hold a negative value or a NaN takes where the
/// device's depth test compares it as it is, or where a pass that sets a state with depth writes
/// but draws under none keeps its sign.
pub(super) const SIGNED_DEPTH_TESTS: [Degradation; 1] = [Degradation::SignedDepthTest];

/// The 18 stored values of the probe `depthkeep`: ±2, ±0.5, the smallest and largest denormal of
/// each sign, ±0, the smallest normal, 0.25, 0.5, 1, the `float` after 1, 1.5, ±inf, and the quiet
/// NaN.
const PROBED: [u32; 18] = [
    0x4000_0000,
    0xc000_0000,
    0x3f00_0000,
    0xbf00_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x0000_0000,
    0x8000_0000,
    0x0080_0000,
    0x3e80_0000,
    0x3f80_0000,
    0x3f80_0001,
    0x3fc0_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
];

/// The probe's depth tests: each of these compare functions at each of [`FRAGMENTS`].
const PROBE_TESTS: [CompareFunction; 5] = [
    CompareFunction::Less,
    CompareFunction::LessEqual,
    CompareFunction::Equal,
    CompareFunction::GreaterEqual,
    CompareFunction::Greater,
];

/// The fragment depths of the probe's tests.
const FRAGMENTS: [f32; 3] = [0.0, 0.5, 1.0];

/// The eight compare functions, each a runtime sampler's for `sample_compare`.
const FUNCTIONS: [CompareFunction; 8] = [
    CompareFunction::Never,
    CompareFunction::Less,
    CompareFunction::Equal,
    CompareFunction::LessEqual,
    CompareFunction::Greater,
    CompareFunction::NotEqual,
    CompareFunction::GreaterEqual,
    CompareFunction::Always,
];

/// The references each stored value is compared against.
const REFERENCES: [f32; 5] = [-1.0, 0.0, 0.5, 1.0, 2.0];

/// The merge's texels, row by row: negative values, -0, NaNs, the infinities, and values on both
/// sides of 0.5 (a draw at 0.5 under `Less` writes where the magnitude lies above it).
const MERGED: [u32; 16] = [
    0xbf00_0000,
    0x7fc0_0000,
    0xc000_0000,
    0x3f40_0000,
    0x3e80_0000,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x3f00_0000,
    0x3fc0_0000,
    0x3e00_0000,
    0xbe00_0000,
    0x4000_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7fc0_0001,
];

/// Synthetic AIR written for this test: `sd_read` reads texel `(0, 0)` of the `depth2d<float>` at
/// slot 0, `sd_sample` samples it through sampler slot 0 at buffer 0's lane `tid` (a `float4`: the
/// coordinate in `xy`), and `sd_compare` compares it there through sampler slot 0 against the
/// reference in `z`; each writes its `float` to `x` of buffer 1's lane `tid` (a `float4`).
const READS: &str = r#"define void @sd_read(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {
  %i = zext i32 %tid to i64
  %rs = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { float, i8 } @air.read_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, i32 1, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { float, i8 } %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}

define void @sd_sample(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid) {
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %r = tail call { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 1, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { float, i8 } %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}

define void @sd_compare(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid) {
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %r = call { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 1, <2 x float> %uv, float %ref, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue { float, i8 } %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { float, i8 } @air.read_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x i32>, <2 x i32>, i32, i32)
declare { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!0, !10, !20}
!air.version = !{!30}
!air.language_version = !{!31}
!0 = !{ptr @sd_read, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"depth2d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!8 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!9 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"t"}
!10 = !{ptr @sd_sample, !1, !11}
!11 = !{!9, !4, !5, !6, !8}
!20 = !{ptr @sd_compare, !1, !11}
!30 = !{i32 2, i32 8, i32 0}
!31 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The textures' side.
const SIDE: u64 = 4;

/// The denormals: the smallest and largest of each sign.
const DENORMALS: [u32; 4] = [0x0000_0001, 0x007f_ffff, 0x8000_0001, 0x807f_ffff];

/// The controls, which every device stores and tests as given: the smallest normal, and 0.5.
const CONTROLS: [u32; 2] = [0x0080_0000, 0x3f00_0000];

/// The ways a program stores depth.
#[derive(Clone, Copy, Debug)]
enum Route {
    Clear,
    ReplaceRegion,
    BlitFromBuffer,
    Draw,
}

const ROUTES: [Route; 4] = [
    Route::Clear,
    Route::ReplaceRegion,
    Route::BlitFromBuffer,
    Route::Draw,
];

/// The lane's device, queue, and the fixture's pipeline drawing white over a depth attachment.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
    white: RenderPipelineState,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let library = library(&device);
        let white = device
            .new_render_pipeline_state(&white_descriptor(&library), &PipelineOptions::default())
            .unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            device,
            library,
            white,
        }
    }

    /// A 4x4 texture of `pf` the passes render to and the CPU reads and writes.
    fn texture(&self, pf: PixelFormat) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::new(
            StorageMode::Shared,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        self.device.new_texture(&d).unwrap()
    }

    /// A pass over `depth` (cleared to `clear` or loaded) and a color target cleared to zero,
    /// drawing white at depth `z` under `compare`, with depth writes where `write`, once; the
    /// color target's texels.
    fn pass(
        &self,
        depth: &Texture,
        clear: Option<f64>,
        draw: Option<(f32, CompareFunction, bool)>,
    ) -> Vec<u8> {
        let color = self.texture(PixelFormat::RGBA8Unorm);
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        target.store_action = StoreAction::Store;
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = match clear {
            Some(_) => LoadAction::Clear,
            None => LoadAction::Load,
        };
        pass.depth_attachment.clear_depth = clear.unwrap_or(1.0);
        pass.depth_attachment.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let mut keep = Vec::new();
        if let Some((z, compare, write)) = draw {
            let mut state = DepthStencilDescriptor::default();
            state.depth_compare_function = compare;
            state.depth_write_enabled = write;
            let state = self.device.new_depth_stencil_state(&state).unwrap();
            let words: Vec<u8> = [z, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
            let at = buffer_with(&self.device, &words).unwrap();
            encoder.set_render_pipeline_state(&self.white).unwrap();
            encoder.set_depth_stencil_state(&state).unwrap();
            encoder.set_vertex_buffer(Some(&at), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            keep.push(at);
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        texels(&color)
    }

    /// A pass that loads and stores `depth` and sets a depth/stencil state (`Less`, depth writes
    /// where `write`), then does `then`.
    fn state_pass(&self, depth: &Texture, write: bool, then: Then) {
        let color = self.texture(PixelFormat::RGBA8Unorm);
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color);
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::Store;
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Load;
        pass.depth_attachment.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Less;
        state.depth_write_enabled = write;
        let state = self.device.new_depth_stencil_state(&state).unwrap();
        encoder.set_render_pipeline_state(&self.white).unwrap();
        encoder.set_depth_stencil_state(&state).unwrap();
        let words: Vec<u8> = [0.5f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let at = buffer_with(&self.device, &words).unwrap();
        encoder.set_vertex_buffer(Some(&at), 0, 0).unwrap();
        match then {
            Then::Nothing => {}
            Then::NoInstances => encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 0, 0)
                .unwrap(),
            Then::WritesOffDraw => {
                let mut off = DepthStencilDescriptor::default();
                off.depth_compare_function = CompareFunction::Less;
                let off = self.device.new_depth_stencil_state(&off).unwrap();
                encoder.set_depth_stencil_state(&off).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// The words of `depth`, by `get_bytes`.
    fn words(&self, depth: &Texture) -> Vec<u32> {
        texels(depth)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    }

    /// A pass that clears `depth` to `clear` and stores it, and draws nothing.
    fn clearing_pass(&self, depth: &Texture, clear: f64) {
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = clear;
        pass.depth_attachment.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// A private 4x4 render target of `pf` at `samples` holding `words` (each texel's samples in
    /// turn, row by row) by a blit from a buffer with `option`, or the runtime's refusal of it.
    fn blitted(
        &self,
        pf: PixelFormat,
        samples: u64,
        words: &[u32],
        option: BlitOption,
    ) -> Result<Texture, Error> {
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = samples;
        }
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let texture = self.device.new_texture(&d)?;
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let source = buffer_with(&self.device, &bytes).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            SIDE * samples * 4,
            bytes.len() as u64,
            Size::new(SIDE, SIDE, 1),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            option,
        )?;
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(texture)
    }

    /// The depth words of `depth`, a 4x4 texture at `samples` (each texel's samples in turn, row
    /// by row), by a blit into a buffer with `option`.
    fn blitted_words(&self, depth: &Texture, samples: u64, option: BlitOption) -> Vec<u32> {
        let bytes = SIDE * SIDE * samples * 4;
        let out = buffer_with(&self.device, &vec![0xee; bytes as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            depth,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &out,
            0,
            SIDE * samples * 4,
            bytes,
            option,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        read_buffer(&out, 0, bytes)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    }

    /// A depth texture holding `bits` everywhere, stored by `route`.
    fn stored(&self, route: Route, bits: u32) -> Texture {
        let depth = self.texture(PixelFormat::Depth32Float);
        let value = f32::from_bits(bits);
        let bytes: Vec<u8> = (0..SIDE * SIDE).flat_map(|_| bits.to_le_bytes()).collect();
        match route {
            Route::Clear => {
                self.pass(&depth, Some(f64::from(value)), None);
            }
            Route::ReplaceRegion => depth
                .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &bytes, SIDE * 4, 0)
                .unwrap(),
            Route::BlitFromBuffer => {
                let source = buffer_with(&self.device, &bytes).unwrap();
                let command_buffer = self.queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_texture(
                    &source,
                    0,
                    SIDE * 4,
                    SIDE * SIDE * 4,
                    Size::new(SIDE, SIDE, 1),
                    &depth,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            Route::Draw => {
                self.pass(
                    &depth,
                    Some(0.5),
                    Some((value, CompareFunction::Always, true)),
                );
            }
        }
        depth
    }
}

/// The descriptor of the lane's pipeline drawing white over a depth attachment, of `library`.
fn white_descriptor(library: &Library) -> RenderPipelineDescriptor {
    let function = |name| library.new_function_with_name(name).unwrap();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
        function("rt6_full").function_descriptor(),
    ));
    descriptor.fragment = Some(StageSource::Function(
        function("rt6_white").function_descriptor(),
    ));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    descriptor
}

/// Every texel of `texture`, 4 bytes each.
fn texels(texture: &Texture) -> Vec<u8> {
    let mut out = vec![0u8; (SIDE * SIDE * 4) as usize];
    texture
        .get_bytes(
            &mut out,
            SIDE * 4,
            SIDE * SIDE * 4,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    out
}

/// Each denormal by each route, read and tested (the module docs).
#[test]
fn depth_images_hold_denormals_as_their_drivers_claim() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let facts = &device.properties().driver_facts;
        let keeps = facts.contains(&DriverFact::DepthImageKeepsDenormals);
        let tests = facts.contains(&DriverFact::DepthTestKeepsDenormals);
        let clamps = clamps_depth_uploads(&device);
        // The uploads into the depth images (the blit's, into a render target, may store a
        // negative value), the draws' `Shared` buffers, and the white pipeline's making.
        let mut first = forced(&[
            (Degradation::RestrictedDepthUpload, clamps),
            (Degradation::SignedDepthTest, signs_depth_tests(&device)),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
            (Degradation::DepthDenormals, !(keeps && tests)),
        ]);
        let lane = Lane::new(device);
        let drawn_kinds = made_render_kinds(&lane.device, &white_descriptor(&lane.library));
        // Where the drawing stage leaves a `float` denormal to the device (`device-denormals`),
        // a draw at a denormal depth stores the device's: zero, as Metal's rasterizer stores, or
        // the denormal kept.
        let device_denormals = drawn_kinds.contains(&Degradation::DeviceDenormals);
        first.extend(drawn_kinds);
        for route in ROUTES {
            for bits in CONTROLS.into_iter().chain(DENORMALS) {
                let control = CONTROLS.contains(&bits);
                let negative = bits & 0x8000_0000 != 0;
                // A clear and a draw store no negative depth: a draw's is clipped, and a clear's
                // is outside the range Vulkan clears to without `VK_EXT_depth_range_unrestricted`.
                if negative && matches!(route, Route::Clear | Route::Draw) {
                    continue;
                }
                let depth = lane.stored(route, bits);
                let read: Vec<u32> = texels(&depth)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect();
                let compare = match negative {
                    true => CompareFunction::Greater,
                    false => CompareFunction::Less,
                };
                let color = lane.pass(&depth, None, Some((0.0, compare, false)));
                let passed = color
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|t| t[0] == 255)
                    .count();
                let held = read.iter().filter(|&&w| w == bits).count();
                println!(
                    "DEPTH-DENORMAL {name} {route:?} {bits:08x}: read {:08x} ({held} of 16 kept), \
                     tested {passed} of 16 passed",
                    read[0]
                );
                let at = format!("{name} {route:?} {bits:08x}");
                if control {
                    assert_eq!(held, 16, "{at}: read {read:08x?}");
                    assert_eq!(passed, 16, "{at}: tested");
                    continue;
                }
                // A draw at a denormal depth stores zero, as Metal's rasterizer does, where the
                // stage does not leave the denormal to the device.
                if matches!(route, Route::Draw) {
                    let kept = device_denormals && read.iter().all(|&w| w == bits);
                    assert!(
                        kept || read.iter().all(|&w| w == 0),
                        "{at}: read {read:08x?}"
                    );
                    continue;
                }
                // What the upload gives the image: a negative denormal clamped to 0 where uploads
                // clamp (`bind::clamped_depth`), each other value as it is.
                let given = match clamps {
                    true => metal2vulkan::bind::clamped_depth(bits),
                    false => bits,
                };
                if keeps {
                    let stored = read.iter().filter(|&&w| w == given).count();
                    assert_eq!(stored, 16, "{at}: given {given:08x}, read {read:08x?}");
                }
                if held == 16 && tests {
                    let metals = if negative { 0 } else { 16 };
                    assert_eq!(passed, metals, "{at}: tested");
                }
            }
        }
        let counts = assert_made_once(&messages, &[], &first);
        let announced = counts.contains_key(&Degradation::DepthDenormals);
        println!("DEPTH-DENORMAL {name} announced depth-denormals: {announced}");
        assert_eq!(
            announced,
            !(keeps && tests),
            "{name}: depth-denormals announced where the driver lacks a fact (keeps {keeps}, \
             tests {tests})"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// Whether Metal's depth test passes a fragment at depth `z` under `compare` against the stored
/// value of bits `stored`: the two magnitudes compared as the integers of their low 31 bits (the
/// module docs).
fn metal_passes(z: f32, compare: CompareFunction, stored: u32) -> bool {
    let (fragment, held) = (z.to_bits() & 0x7fff_ffff, stored & 0x7fff_ffff);
    match compare {
        CompareFunction::Never => false,
        CompareFunction::Less => fragment < held,
        CompareFunction::Equal => fragment == held,
        CompareFunction::LessEqual => fragment <= held,
        CompareFunction::Greater => fragment > held,
        CompareFunction::NotEqual => fragment != held,
        CompareFunction::GreaterEqual => fragment >= held,
        CompareFunction::Always => true,
        other => panic!("{other:?} is no compare function Metal names"),
    }
}

/// Whether Metal's `sample_compare` under `f` passes the reference `r` against the texel `texel`:
/// the IEEE order of the two, neither clamped, `-0` equal to `+0`; the function's raw value's bits
/// 0, 1 and 2 pass a reference below, equal to and above the texel, and an unordered pair passes
/// `not_equal` and `always` alone (`depth_compares`' rule, the probe's 720 of 720).
fn metal_compares(f: CompareFunction, r: f32, texel: f32) -> bool {
    let bits = f.as_raw();
    match r.partial_cmp(&texel) {
        Some(Ordering::Less) => bits & 1 != 0,
        Some(Ordering::Equal) => bits & 2 != 0,
        Some(Ordering::Greater) => bits & 4 != 0,
        None => bits & 5 == 5,
    }
}

/// Whether `a` and `b` are one read of a texel: the same bits, or both NaN (a sample or a
/// kernel's read need not keep a NaN's payload).
fn same_read(a: u32, b: u32) -> bool {
    let nan = |w: u32| f32::from_bits(w).is_nan();
    a == b || (nan(a) && nan(b))
}

/// What a test of the probe's values reads of one stored texture: `get_bytes`' first word, a
/// kernel's read and nearest sample of texel `(0, 0)`, every `sample_compare` of [`REFERENCES`]
/// under each of [`FUNCTIONS`] (in that order, the references inner),
/// and how many of the 16 texels each depth test of [`PROBE_TESTS`] at each of [`FRAGMENTS`]
/// passed.
struct Observed {
    read: u32,
    kernel_read: u32,
    sampled: u32,
    compared: Vec<bool>,
    tested: Vec<usize>,
}

/// The kernels and samplers that read a stored texture ([`READS`]).
struct Readers {
    read: ComputePipelineState,
    sample: ComputePipelineState,
    compare: ComputePipelineState,
    nearest: SamplerState,
    comparing: Vec<SamplerState>,
}

impl Readers {
    fn new(device: &Device, library: &Library) -> Readers {
        let kernel = |name| {
            let function = library.new_function_with_name(name).unwrap();
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(StageSource::Function(
                        function.function_descriptor(),
                    )),
                    &PipelineOptions::default(),
                )
                .unwrap()
        };
        let sampler = |compare: Option<CompareFunction>| {
            let mut d = SamplerDescriptor::new();
            d.min_filter = SamplerMinMagFilter::Nearest;
            d.mag_filter = SamplerMinMagFilter::Nearest;
            if let Some(f) = compare {
                d.compare_function = f;
            }
            device.new_sampler_state(&d).unwrap()
        };
        Readers {
            read: kernel("sd_read"),
            sample: kernel("sd_sample"),
            compare: kernel("sd_compare"),
            nearest: sampler(None),
            comparing: FUNCTIONS.iter().map(|&f| sampler(Some(f))).collect(),
        }
    }
}

impl Lane {
    /// What `depth`, a texture of one stored value, reads and tests as (the module docs).
    fn observed(&self, depth: &Texture, readers: &Readers) -> Observed {
        let read = texels(depth);
        let read = u32::from_le_bytes([read[0], read[1], read[2], read[3]]);
        let centre = 0.5 / SIDE as f32;
        let at = [[centre, centre, 0.0, 0.0]];
        let word = |state: &ComputePipelineState, sampler: Option<&SamplerState>| {
            run(&self.device, (depth, sampler), state, &at, 1)[0].to_bits()
        };
        let kernel_read = word(&readers.read, None);
        let sampled = word(&readers.sample, Some(&readers.nearest));
        let jobs: Vec<[f32; 4]> = REFERENCES
            .iter()
            .map(|&r| [centre, centre, r, 0.0])
            .collect();
        let mut compared = Vec::new();
        for sampler in &readers.comparing {
            let out = run(
                &self.device,
                (depth, Some(sampler)),
                &readers.compare,
                &jobs,
                1,
            );
            compared.extend((0..jobs.len()).map(|i| out[4 * i] == 1.0));
        }
        let mut tested = Vec::new();
        for compare in PROBE_TESTS {
            for z in FRAGMENTS {
                let color = self.pass(depth, None, Some((z, compare, false)));
                tested.push(passed(&color));
            }
        }
        Observed {
            read,
            kernel_read,
            sampled,
            compared,
            tested,
        }
    }
}

/// What a [`Lane::state_pass`] does after it sets its state.
#[derive(Clone, Copy, Debug)]
enum Then {
    /// Nothing: the pass draws nothing.
    Nothing,
    /// A draw of no instances, which records nothing.
    NoInstances,
    /// A draw at 0.5 under a state without depth writes, set over the first.
    WritesOffDraw,
}

/// How many of a color target's texels a pass drew white.
fn passed(color: &[u8]) -> usize {
    color
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|t| t[0] == 255)
        .count()
}

/// Each of the probe's 18 values stored by `replace_region` and by a blit into a render target's
/// `Depth32Float`, then read, sampled, compared and depth-tested (the module docs): on a device
/// that keeps such a value in a depth image, each read, sample and comparison is Metal's wherever
/// the value was stored as given (a blit's denormal is a zero on a driver without
/// `DriverFact::DepthImageKeepsDenormals`, and `get_bytes` reads it so; a host copy's is read as a
/// zero by such a driver's GPU copies and reads, announced as `depth-denormals`, and a sample or
/// comparison whose making records `device-sampling`, the device's sampler's answer, may read it
/// so), and each depth test too where the device's test is Metal's own
/// (`DriverFact::DepthTestComparesMagnitudes`). There, a pass with depth writes over a texture
/// holding such values stores its draw's depth where it passed and every other texel without its
/// sign, as Metal's (probe `depthstore`), and one that sets a state with depth writes but draws
/// under none announces `signed-depth-test`; elsewhere the writes announce it. On every device
/// that keeps such a value, a pass whose state has no depth writes keeps every value. A device
/// that may clamp such a value announces `restricted-depth-upload` instead.
#[test]
fn a_stored_depth_reads_samples_and_compares_as_metal_s() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        // The device keeps such a value, and its depth test is Metal's own or compares it as it
        // is.
        let keeps = !clamps_depth_uploads(&device);
        let signs = signs_depth_tests(&device);
        let magnitudes = keeps && !signs;
        let facts = &device.properties().driver_facts;
        let library = device
            .new_library_with_air(&llvm_assemble_text(READS).unwrap())
            .unwrap();
        let kernels = ["sd_read", "sd_sample", "sd_compare"];
        let function = |name| library.new_function_with_name(name).unwrap();
        let mut allowed = BTreeSet::new();
        let mut first = forced(&[
            (Degradation::RestrictedDepthUpload, !keeps),
            // The writes where the device's test is not Metal's; the passes that draw under no
            // state with depth writes where it is.
            (Degradation::SignedDepthTest, keeps),
            (
                Degradation::DepthDenormals,
                loses_depth_denormals(&device, false),
            ),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
        ]);
        for kernel in kernels {
            first.extend(made_compute_kinds(
                &device,
                function(kernel).function_descriptor(),
            ));
            allowed.extend(bound_value_kinds(
                &device,
                function(kernel).function_descriptor(),
            ));
        }
        // Where a denormal is Metal's: a driver whose GPU copies and reads keep one in a depth
        // image, and of those, a sample or comparison whose making records no note that it may
        // read one otherwise.
        let kinds = |kernel| {
            let descriptor = || function(kernel).function_descriptor();
            let mut kinds = made_compute_kinds(&device, descriptor());
            kinds.extend(bound_value_kinds(&device, descriptor()));
            kinds
        };
        let denormals = facts.contains(&DriverFact::DepthImageKeepsDenormals);
        let sampled_denormals =
            denormals && !kinds("sd_sample").contains(&Degradation::DeviceSampling);
        let compared_denormals =
            denormals && !kinds("sd_compare").contains(&Degradation::DeviceSampling);
        let readers = Readers::new(&device, &library);
        let lane = Lane::new(device);
        first.extend(made_render_kinds(
            &lane.device,
            &white_descriptor(&lane.library),
        ));
        let (mut asked, mut metals) = (0, 0);
        for route in [Route::ReplaceRegion, Route::BlitFromBuffer] {
            for bits in PROBED {
                let depth = lane.stored(route, bits);
                let seen = lane.observed(&depth, &readers);
                let value = f32::from_bits(bits);
                // Metal's answers stand for the value as it was stored.
                let kept = keeps && seen.read == bits;
                let at = format!("{name} {route:?} {bits:08x}");
                println!(
                    "PROBED-DEPTH {at}: read {:08x}, kernel read {:08x}, sampled {:08x}",
                    seen.read, seen.kernel_read, seen.sampled
                );
                let compares = FUNCTIONS
                    .iter()
                    .flat_map(|&f| REFERENCES.iter().map(move |&r| metal_compares(f, r, value)));
                let tests = PROBE_TESTS.iter().flat_map(|&compare| {
                    FRAGMENTS
                        .iter()
                        .map(move |&z| match metal_passes(z, compare, bits) {
                            true => 16,
                            false => 0,
                        })
                });
                let mut wrong_compares = Vec::new();
                for ((f, r), (got, want)) in FUNCTIONS
                    .iter()
                    .flat_map(|f| REFERENCES.iter().map(move |r| (f, r)))
                    .zip(seen.compared.iter().zip(compares))
                {
                    asked += 1;
                    metals += usize::from(*got == want);
                    if *got != want {
                        wrong_compares
                            .push(format!("compare {f:?} ref {r}: {got}, Metal's {want}"));
                    }
                }
                let mut wrong_tests = Vec::new();
                for ((compare, z), (&got, want)) in PROBE_TESTS
                    .iter()
                    .flat_map(|c| FRAGMENTS.iter().map(move |z| (c, z)))
                    .zip(seen.tested.iter().zip(tests))
                {
                    asked += 1;
                    metals += usize::from(got == want);
                    if got != want {
                        wrong_tests.push(format!(
                            "test {compare:?} at {z}: {got} of 16, Metal's {want}"
                        ));
                    }
                }
                for line in wrong_compares.iter().chain(&wrong_tests) {
                    println!("PROBED-DEPTH {at}: {line}");
                }
                let denormal = bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0;
                if kept && (!denormal || denormals) {
                    assert!(same_read(seen.kernel_read, bits), "{at}: kernel read");
                    if magnitudes {
                        assert!(wrong_tests.is_empty(), "{at}: {wrong_tests:#?}");
                    }
                }
                if kept && (!denormal || sampled_denormals) {
                    assert!(same_read(seen.sampled, bits), "{at}: nearest sample");
                }
                if kept && (!denormal || compared_denormals) {
                    assert!(wrong_compares.is_empty(), "{at}: {wrong_compares:#?}");
                }
            }
        }
        // The merge: values a draw at 0.5 under `Less`, writing its depth, passes against where
        // their magnitude lies above 0.5 under Metal's test. Metal stores the draw's depth there
        // and every other texel without its sign (probe `depthstore`); a pass that sets a state
        // with depth writes and draws nothing stores every texel so, and one whose state has none
        // keeps every bit.
        let held = || {
            let depth = lane.texture(PixelFormat::Depth32Float);
            let bytes: Vec<u8> = MERGED.iter().flat_map(|w| w.to_le_bytes()).collect();
            depth
                .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &bytes, SIDE * 4, 0)
                .unwrap();
            depth
        };
        let merged = held();
        let color = lane.pass(&merged, None, Some((0.5, CompareFunction::Less, true)));
        // A state with depth writes and no draw, one with a draw of no instances, and one replaced
        // by a state without depth writes before its draw: Metal strips each (probe
        // `depthstore`); a device whose pass is Metal's own is handed no state with depth writes.
        let unstored: Vec<(&str, Texture)> = [
            ("no draw", Then::Nothing),
            ("no instances", Then::NoInstances),
            ("replaced", Then::WritesOffDraw),
        ]
        .into_iter()
        .map(|(case, then)| {
            let depth = held();
            lane.state_pass(&depth, true, then);
            (case, depth)
        })
        .collect();
        let kept = held();
        lane.state_pass(&kept, false, Then::Nothing);
        let drawn = 0.5f32.to_bits();
        let want: Vec<u32> = MERGED
            .iter()
            .map(|&w| match metal_passes(0.5, CompareFunction::Less, w) {
                true => drawn,
                false => w & 0x7fff_ffff,
            })
            .collect();
        let read = lane.words(&merged);
        let read_kept = lane.words(&kept);
        println!("PROBED-DEPTH {name} no depth writes: read {read_kept:08x?}");
        for (case, depth) in &unstored {
            let read = lane.words(depth);
            println!("PROBED-DEPTH {name} depth writes, {case}: read {read:08x?}");
            if keeps {
                let either = (read.iter().zip(MERGED))
                    .all(|(&w, m)| same_read(w, m) || same_read(w, m & 0x7fff_ffff));
                assert!(either, "{name}: {case}: each word with or without its sign");
            }
        }
        // The texels Metal's test passes: not those whose stored value equals the draw's (0.5).
        let wins = MERGED
            .iter()
            .filter(|&&w| metal_passes(0.5, CompareFunction::Less, w))
            .count();
        println!(
            "PROBED-DEPTH {name} merged: {} of 16 drawn (Metal's {wins}), read {read:08x?}",
            passed(&color)
        );
        println!(
            "PROBED-DEPTH {name}: {metals} of {asked} answers Metal's; keeps {keeps}; the \
             device's test is Metal's {magnitudes}"
        );
        if magnitudes {
            assert_eq!(passed(&color), wins, "{name}: the merge pass drew");
            assert_eq!(
                read, want,
                "{name}: the merge stored the draw's depth and the others without their signs"
            );
        }
        if keeps {
            let same = read_kept.iter().zip(MERGED).all(|(&a, b)| same_read(a, b));
            assert!(same, "{name}: a pass without depth writes kept every value");
        }
        let taken = messages.taken_kinds();
        let expected: Vec<Degradation> = first.difference(&taken).copied().collect();
        assert_once_allowing(&messages.take_notices(), &expected, &allowed);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A NaN clear depth stores as Metal's (our probe `nanclear`, the M3 Max under macOS 27.0.1): a
/// `Depth32Float` or `Depth32Float_Stencil8` attachment, 1x and 4x, cleared to a NaN of either sign
/// and stored, holds `0x7fc00000` in every texel and sample; a `Depth16Unorm` one holds 0. Where
/// the device makes no write of the word (a multisampled depth with neither maintenance8's colour
/// copies nor an exact `FragDepth`, a device that may clamp a depth), the pass clears to 1.0,
/// announced. A draw at 0.5 in the clearing pass passes `Less` and fails `Greater`, as Metal's
/// test orders a NaN above every value, where the device's test is Metal's own or the pass clears
/// to 1.0; elsewhere the device compares the NaN as it is, announced (`signed-depth-test`). Each
/// case prints a `NAN-CLEAR-DEPTH` line.
#[test]
fn a_nan_clear_depth_stores_as_metal_s() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let keeps = !clamps_depth_uploads(&device);
        let signs = signs_depth_tests(&device);
        let colour = device.enabled().features.contains(&VkFeature::Maintenance8)
            && !device
                .properties()
                .driver_defects
                .contains(&RuntimeDriverDefect::MultisampledDepthStencilColourCopies);
        let drawn = device
            .properties()
            .driver_facts
            .contains(&DriverFact::FragmentDepthStoresUnchanged);
        let lane = Lane::new(device);
        let (depth, combined) = (BlitOption::None, BlitOption::DepthFromDepthStencil);
        for (case, pf, samples, option) in [
            ("Depth32Float", PixelFormat::Depth32Float, 1, depth),
            ("Depth32Float 4x", PixelFormat::Depth32Float, 4, depth),
            (
                "Depth32Float_Stencil8",
                PixelFormat::Depth32FloatStencil8,
                1,
                combined,
            ),
            (
                "Depth32Float_Stencil8 4x",
                PixelFormat::Depth32FloatStencil8,
                4,
                combined,
            ),
        ] {
            for clear in [f64::NAN, -f64::NAN] {
                let held = vec![0x3e80_0000; (SIDE * SIDE * samples) as usize];
                let texture = lane.blitted(pf, samples, &held, option).unwrap();
                lane.clearing_pass(&texture, clear);
                let read = lane.blitted_words(&texture, samples, option);
                println!(
                    "NAN-CLEAR-DEPTH {name} {case} {clear}: read {:08x?}",
                    &read[..4]
                );
                let written = keeps && (samples == 1 || colour || drawn);
                let want = match written {
                    true => 0x7fc0_0000,
                    false => 1.0f32.to_bits(),
                };
                assert!(
                    read.iter().all(|&w| w == want),
                    "{name}: {case} cleared to {clear}: {want:08x} everywhere"
                );
            }
        }
        // A 16-bit depth's NaN clear stores 0.
        let depth16 = lane.texture(PixelFormat::Depth16Unorm);
        lane.clearing_pass(&depth16, 0.5);
        lane.clearing_pass(&depth16, f64::NAN);
        let codes = texels(&depth16);
        println!(
            "NAN-CLEAR-DEPTH {name} Depth16Unorm: read {:02x?}",
            &codes[..8]
        );
        assert!(
            codes.iter().all(|&b| b == 0),
            "{name}: Depth16Unorm's NaN clear stores 0"
        );
        // In the clearing pass, a draw at 0.5 passes `Less` and fails `Greater` against the NaN,
        // where the device's test is Metal's own or the pass cleared to 1.0.
        for (compare, want) in [(CompareFunction::Less, 16), (CompareFunction::Greater, 0)] {
            let depth = lane.texture(PixelFormat::Depth32Float);
            let color = lane.pass(&depth, Some(f64::NAN), Some((0.5, compare, false)));
            println!(
                "NAN-CLEAR-DEPTH {name} draw at 0.5 under {compare:?}: {} of 16 drawn",
                passed(&color)
            );
            if !signs {
                assert_eq!(
                    passed(&color),
                    want,
                    "{name}: {compare:?} against a NaN clear"
                );
            }
        }
        messages.take_notices();
    }
    layer_reported_nothing_at_all(&messages);
}

/// The side of [`fresh_target`]'s textures, which Metal compresses: each texel reads its pattern.
const FRESH_SIDE: u64 = 16;

/// A new `Private` render target of `pf` of [`FRESH_SIDE`]², sampled too, which nothing wrote.
fn fresh_target(lane: &Lane, pf: PixelFormat) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, FRESH_SIDE, FRESH_SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    lane.device.new_texture(&d).unwrap()
}

/// A pass that loads and stores `depth` (of `pf`, its stencil too) and draws white at 0.5 through
/// `white` under `compare` with depth writes over a [`FRESH_SIDE`]² color target cleared to zero:
/// how many texels it drew.
fn fresh_pass(
    lane: &Lane,
    white: &RenderPipelineState,
    (depth, pf): (&Texture, PixelFormat),
    compare: CompareFunction,
) -> usize {
    let mut d =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, FRESH_SIDE, FRESH_SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::shared();
    let color = lane.device.new_texture(&d).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let target = &mut pass.color_attachments[0];
    target.texture = Some(color.clone());
    target.load_action = LoadAction::Clear;
    target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    target.store_action = StoreAction::Store;
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = LoadAction::Load;
    pass.depth_attachment.store_action = StoreAction::Store;
    if pf == PixelFormat::Depth32FloatStencil8 {
        pass.stencil_attachment.texture = Some(depth.clone());
        pass.stencil_attachment.load_action = LoadAction::Load;
        pass.stencil_attachment.store_action = StoreAction::Store;
    }
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = compare;
    state.depth_write_enabled = true;
    let state = lane.device.new_depth_stencil_state(&state).unwrap();
    let words: Vec<u8> = [0.5f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let at = buffer_with(&lane.device, &words).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(white).unwrap();
    encoder.set_depth_stencil_state(&state).unwrap();
    encoder.set_vertex_buffer(Some(&at), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let side = FRESH_SIDE;
    let mut out = vec![0u8; (side * side * 4) as usize];
    let region = Region::new_2d(0, 0, side, side);
    color
        .get_bytes(&mut out, side * 4, side * side * 4, region, 0, 0)
        .unwrap();
    passed(&out)
}

/// The depth words of [`fresh_target`]'s `depth`, by a blit into a buffer with `option`.
fn fresh_words(lane: &Lane, depth: &Texture, option: BlitOption) -> Vec<u32> {
    let side = FRESH_SIDE;
    let bytes = side * side * 4;
    let out = buffer_with(&lane.device, &vec![0xee; bytes as usize]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let size = Size::new(side, side, 1);
    let origin = Origin::new(0, 0, 0);
    blit.copy_from_texture_to_buffer(depth, 0, 0, origin, size, &out, 0, side * 4, bytes, option)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, bytes)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// The kinds [`a_new_target_s_pattern_loaded_unwritten_draws_as_the_device_tests_it`]'s device
/// announces at each occurrence, so that each format's targets announce their own.
const FRESH_SIGNED_EACH: [Degradation; 2] = [
    Degradation::SignedDepthTest,
    Degradation::FreshPatternZeroed,
];

/// A new target's pattern, a NaN with the sign set, loaded by a pass before anything wrote it,
/// draws and stores as the device's test and pass take it (the module docs): as Metal's where the
/// device's test is Metal's own; elsewhere each pass that loads it announces `signed-depth-test`
/// and each texel no fragment wrote keeps the pattern, with or without its sign; on a device that
/// may clamp the pattern, each target reads and stores zero, announced at its first load; a
/// target cleared first announces neither. The device announces [`FRESH_SIGNED_EACH`] at each
/// occurrence, so each format's targets announce their own.
#[test]
fn a_new_target_s_pattern_loaded_unwritten_draws_as_the_device_tests_it() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let each = |options: &mut DeviceOptions| {
            options.announce_each = FRESH_SIGNED_EACH.into_iter().collect();
        };
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), each).unwrap();
        let name = device.name().to_owned();
        let keeps = !clamps_depth_uploads(&device);
        // The device keeps the pattern and its test is not Metal's own.
        let signs = signs_depth_tests(&device);
        // No pattern at all: a new target holds what its memory held.
        let unfilled = super::fresh_contents::unfilled(&device);
        let lane = Lane::new(device);
        let white = |pf: PixelFormat| {
            let mut d = white_descriptor(&lane.library);
            d.depth_format = Some(pf);
            if pf == PixelFormat::Depth32FloatStencil8 {
                d.stencil_format = Some(pf);
            }
            let options = PipelineOptions::default();
            lane.device.new_render_pipeline_state(&d, &options).unwrap()
        };
        let all = (FRESH_SIDE * FRESH_SIDE) as usize;
        // Each function, whether a draw at 0.5 passes over Metal's pattern and over a zero depth.
        let compares = [
            (CompareFunction::Less, true, false),
            (CompareFunction::NotEqual, true, true),
            (CompareFunction::Greater, false, true),
            (CompareFunction::Equal, false, false),
        ];
        let targets = compares.len();
        let (depth, combined) = (BlitOption::None, BlitOption::DepthFromDepthStencil);
        for (case, pf, option) in [
            ("Depth32Float", PixelFormat::Depth32Float, depth),
            (
                "Depth32Float_Stencil8",
                PixelFormat::Depth32FloatStencil8,
                combined,
            ),
        ] {
            let white = white(pf);
            let mut notices = Vec::new();
            for (compare, over_pattern, over_zero) in compares {
                let target = fresh_target(&lane, pf);
                let drawn = fresh_pass(&lane, &white, (&target, pf), compare);
                let words = fresh_words(&lane, &target, option);
                notices.extend(messages.take_notices());
                println!(
                    "FRESH-SIGNED-DEPTH {name} {case} {compare:?}: drawn {drawn} of {all}, \
                     stored {:08x?}",
                    &words[..2]
                );
                // Where the device keeps the pattern and its test is Metal's own, each texel a
                // fragment wrote holds 0.5 and every other the pattern without its sign, as
                // Metal's; where its test is not, every other the pattern, with or without its
                // sign; elsewhere the target reads zero, announced, and draws and stores as over a
                // zero depth.
                if unfilled {
                    continue;
                }
                let (draws, unwritten) = match keeps {
                    true => (over_pattern, 0x7fff_ffff),
                    false => (over_zero, 0),
                };
                if !signs {
                    let want = if draws { all } else { 0 };
                    assert_eq!(drawn, want, "{name}: {case} under {compare:?}");
                }
                let stored = |w: u32| match drawn == all {
                    true => w == 0x3f00_0000,
                    false => w == unwritten || (signs && w == 0xffff_ffff),
                };
                assert!(
                    words.iter().all(|&w| stored(w)),
                    "{name}: {case} under {compare:?} stores {unwritten:08x}: {:08x?}",
                    &words[..4]
                );
            }
            let announced = counts(&notices);
            let of = |kind: Degradation| announced.get(&kind).copied().unwrap_or(0);
            println!(
                "FRESH-SIGNED-DEPTH {name} {case}: signed-depth-test {}, fresh-pattern-zeroed {}",
                of(Degradation::SignedDepthTest),
                of(Degradation::FreshPatternZeroed),
            );
            // A device whose test is not Metal's announces its test at each pass that loads a
            // target's pattern.
            let signed = if signs && !unfilled { targets } else { 0 };
            assert_eq!(
                of(Degradation::SignedDepthTest),
                signed,
                "{name}: {case}: signed-depth-test"
            );
            let at_load = notices
                .iter()
                .filter_map(|m| m.degradation.as_ref())
                .filter(|n| n.degradation == Degradation::SignedDepthTest)
                .all(|n| n.reason.starts_with("new_texture "));
            assert!(
                at_load,
                "{name}: {case}: its test, at the pass that loads it"
            );
            // A device that may clamp the pattern writes none: zero, announced where each target's
            // pass first loads it.
            let zeroed = if keeps || unfilled { 0 } else { targets };
            assert_eq!(
                of(Degradation::FreshPatternZeroed),
                zeroed,
                "{name}: {case}: fresh-pattern-zeroed"
            );
        }
        // A new target cleared first loads its clear.
        let white = white(PixelFormat::Depth32Float);
        let target = fresh_target(&lane, PixelFormat::Depth32Float);
        lane.clearing_pass(&target, 0.75);
        let pair = (&target, PixelFormat::Depth32Float);
        let drawn = fresh_pass(&lane, &white, pair, CompareFunction::Less);
        println!("FRESH-SIGNED-DEPTH {name} cleared first: drawn {drawn}");
        assert_eq!(
            drawn, all,
            "{name}: a draw at 0.5 under Less over a clear of 0.75"
        );
        let announced = counts(&messages.take_notices());
        assert!(
            FRESH_SIGNED_EACH
                .iter()
                .all(|kind| !announced.contains_key(kind)),
            "{name}: a cleared target announces nothing: {announced:?}"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// The depth of an ordinary pass loads in place: no kept image or stand-in made, and the command
/// buffers each pass begins and submits are printed (`WorkCounts`), the batch's to compare with
/// the base's.
#[test]
fn an_ordinary_depth_loads_in_place() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let before = lane.device.work_counts();
        let depth = lane.stored(Route::ReplaceRegion, 0x3f00_0000);
        const PASSES: u64 = 8;
        for _ in 0..PASSES {
            let at = (0.25, CompareFunction::LessEqual, true);
            let color = lane.pass(&depth, None, Some(at));
            assert_eq!(passed(&color), 16, "{name}");
        }
        let after = lane.device.work_counts();
        let moved = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
        println!(
            "ORDINARY-DEPTH {name}: {PASSES} passes began {} and submitted {} command buffers, \
             made {} kept images and {} stand-ins",
            moved(|c| c.command_buffers_begun),
            moved(|c| c.command_buffers_submitted),
            moved(|c| c.kept_backings_made),
            moved(|c| c.stand_ins_made),
        );
        assert_eq!(moved(|c| c.kept_backings_made), 0, "{name}");
        assert_eq!(moved(|c| c.stand_ins_made), 0, "{name}");
        // 0.5, then each pass's 0.25.
        assert_eq!(texels(&depth)[..4], 0.25f32.to_le_bytes(), "{name}");
        messages.take_notices();
    }
    layer_reported_nothing_at_all(&messages);
}
