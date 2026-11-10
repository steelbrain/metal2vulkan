//! Render passes of more color attachments than the device draws at once, drawn in color groups
//! (the runtime's `render/color_groups.rs`), on the device the cfg selects under the validation
//! layer: restricted to four color attachments and four fragment outputs (Vulkan's floor) through
//! `VK_LAYER_KHRONOS_profiles` and a profile of those two limits alone, which the test writes, and
//! as it is, where nothing groups. The owned fixtures `color_groups` and `color_groups_fetch`, each
//! scene over 8x8 `RGBA16Float` colors:
//!
//! - **five**, **seven**, **sparse**: `cg_five` (colors 0 to 4, two groups), `cg_seven` (0 to 6,
//!   two groups), `cg_sparse` (0 and 5, one group renumbered), each color cleared to its own value,
//!   one fullscreen triangle between a wait for a fence and its update.
//! - **depth**: five colors and a `Depth32FloatStencil8` attachment cleared to 1 and 0, counted
//!   by a visibility buffer. `cg_depth` (depth `Less` with writes, stencil replaced by 1) in the left
//!   half's scissor, then `cg_five` blending one plus one (depth `Less` without writes, stencil
//!   `Equal` 1) over the whole target: the second group's color 4 doubles exactly where the first
//!   draw's depth and stencil let the second pass, on its twin of them.
//! - **load**: a pass that clears and stores every attachment (depth 0.15, stencil 5), then a pass
//!   that loads them, draws `cg_depth` (depth `Less` with writes, stencil `Equal` 5 incremented on
//!   a pass) in the top half's scissor, and stores color 4 and the depth by late store actions:
//!   the second group's twin is loaded by its copy.
//! - **fetch**: `cg_fetch_sparse` in a pass of colors 0 and 5, reading color 5's clear.
//! - **dead read**: `dfk_dead_group`'s `dfg_branch` and `dfg_gated` specialized with their read of
//!   color 4 dead (function constant 0 false), and `dfk_dead_rog`'s `dfr_color` (a color 0 input in
//!   raster order group 0) and `dfr_buffer` (a buffer in group 0) with theirs dead, in a pass of
//!   colors 0 to 4: each writes `tint` swizzled to colors 0 to 3 and `tint` to color 4, and reads
//!   nothing, so on the restricted device it draws in two groups as `cg_five` does, its group
//!   ordering nothing. Specialized live, each is the pipeline's limit refusal there (`dfg_branch`
//!   reads color 4 from group 0's variant; the others order a read in each group's draw).
//!   `dfr_lacks`, a color 1 input beside a pipeline of color 0 alone, is the pipeline's
//!   `InvalidDescriptor(ColorAttachment)` dead and live, as plain Metal refuses it both ways
//!   (`metal2vulkan-conformance/probes/dead_ordered_and_unattached_reads.m`).
//!
//! Metal's answers (a probe on the Apple M3 Max, macOS 27.0.1, of these scenes; the dead read's of
//! `dfk_dead_rog` by `dead_ordered_and_unattached_reads.m`, `dfg_branch` and `dfg_gated` writing
//! `tint` and its swizzles only): every texel of
//! every color, depth and stencil is the closed form the checks compute ([`tint`] at each pixel
//! center), and the visibility counts are 32 and 22. Where the driver counts each sample once per
//! color its fragment shader writes (ANV: `RuntimeDriverDefect::OcclusionCountPerColorOutput`),
//! the counted draws run announced (`visibility-count-per-color`), and the counts are only checked
//! nonzero where Metal's are.
//!
//! On the restricted device, the passes of two groups draw a pass past their first and make one
//! twin of the depth and stencil where they have one ([`WorkCounts::color_group_passes`] and
//! `color_group_twins`); on the device as it is (eight), none. Refused there with the pipeline's
//! limit refusal: `cg_counts`, whose fragment function writes memory, and `cg_fetch`, which reads a
//! color of another group. A pipeline of four colors in the five-color pass, which the device
//! draws in groups, is refused `NotImplemented(ColorAttachmentGroups)`.

use super::framebuffer_fetch::fetches;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use crate::observation::PROFILES_LAYER;
use metal2vulkan::device::VkLimit;
use metal2vulkan::error::{
    Capability, DescriptorError, Error as CompileError, Feature, MetalAnswer, RuntimeDriverDefect,
    Unsupported,
};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, DataType, PixelFormat, PrimitiveType, RenderStages,
    StencilOperation, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Degradation, DepthStencilState, Device, Error,
    LayerSetting, LayerSettingValue, Library, LoadAction, Origin, PipelineOptions,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap,
    ScissorRect, Size, StoreAction, Texture, TextureDescriptor, VisibilityResultMode, WorkCounts,
};
use std::collections::BTreeSet;
use std::ffi::CString;
use std::time::{Duration, Instant};

/// Every target's side.
const SIDE: u64 = 8;

/// The kind [`scenes`] takes, in both widths, where the driver counts each sample per color.
pub(super) const PER_COLOR_COUNTS: [Degradation; 1] = [Degradation::VisibilityCountPerColor];

/// The device's color attachments: restricted to four, or as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    Four,
    Device,
}

struct Lane {
    device: Device,
    queue: CommandQueue,
    groups: Library,
    fetch: Library,
    dead: Library,
    rog: Library,
    width: Width,
}

/// The profile of four color attachments and four fragment outputs alone: the profiles layer
/// reports the device's own value for every member it omits.
const FOUR_COLORS: &str = r#"{
  "$schema": "https://schema.khronos.org/vulkan/profiles-0.8-latest.json#",
  "capabilities": {
    "four_colors": {
      "properties": {
        "VkPhysicalDeviceProperties": {
          "limits": { "maxColorAttachments": 4, "maxFragmentOutputAttachments": 4 }
        }
      }
    }
  },
  "profiles": {
    "VP_M2V_four_colors": {
      "version": 1,
      "api-version": "1.2.0",
      "label": "Four color attachments",
      "description": "The device's own capabilities but four color attachments and fragment outputs.",
      "contributors": {},
      "history": [
        { "revision": 1, "date": "2026-10-16", "author": "metal2vulkan", "comment": "A lane's." }
      ],
      "capabilities": ["four_colors"]
    }
  }
}"#;

/// The lane's device at `width`, under the validation layer where `validated`, its messages and
/// shader cache, and the directory of the profile that restricts it; `None` (skipped, visibly)
/// where the profiles layer is not installed or the device as it is draws fewer than eight.
fn lane(width: Width, validated: bool) -> Option<(Messages, ScratchCache, ScratchCache, Lane)> {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let profile = ScratchCache::new().unwrap();
    let file = profile.path().join("four_colors.json");
    if width == Width::Four {
        if !crate::candidate::profiles_layer_installed().unwrap() {
            println!("SKIPPED: VK_LAYER_KHRONOS_profiles is not installed");
            return None;
        }
        std::fs::create_dir_all(profile.path()).unwrap();
        std::fs::write(&file, FOUR_COLORS).unwrap();
    }
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        if !validated {
            options.layers.clear();
        }
        if width == Width::Four {
            let setting = |name: &str, value| {
                LayerSetting::new(
                    PROFILES_LAYER.to_owned(),
                    CString::new(name).unwrap(),
                    value,
                )
            };
            let strings = |values: &[&str]| {
                LayerSettingValue::Strings(
                    values.iter().map(|v| CString::new(*v).unwrap()).collect(),
                )
            };
            let path = file.to_string_lossy().into_owned();
            options.layers.push(PROFILES_LAYER.to_owned());
            options.layer_settings.extend([
                setting("profile_emulation", LayerSettingValue::Bool(true)),
                setting("profile_file", strings(&[&path])),
                setting("profile_name", strings(&["VP_M2V_four_colors"])),
                setting(
                    "simulate_capabilities",
                    strings(&["SIMULATE_PROPERTIES_BIT"]),
                ),
                setting("emulate_portability", LayerSettingValue::Bool(false)),
                setting("debug_fail_on_error", LayerSettingValue::Bool(false)),
            ]);
        }
    })
    .unwrap();
    let limits = device.compile_device().profile().limits();
    let held = limits
        .max_color_attachments
        .min(limits.max_fragment_output_attachments);
    match width {
        Width::Four => assert_eq!(held, 4, "the restricted {}", device.name()),
        Width::Device => {
            if held < 8 {
                println!("SKIPPED {}: draws {held} colors at once", device.name());
                return None;
            }
        }
    }
    let library = |stem: &str| {
        let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
        device.new_library_with_air(&bitcode).unwrap()
    };
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        groups: library("color_groups"),
        fetch: library("color_groups_fetch"),
        dead: library("dfk_dead_group"),
        rog: library("dfk_dead_rog"),
        width,
        device,
    };
    Some((messages, cache, profile, lane))
}

/// The fixture's `tint` at pixel `(x, y)`'s center: the fullscreen triangle's `a` is
/// `((2x + 1) / 32, (15 - 2y) / 32)` there, and `tint` is `(a.x, a.y, 1 - a.x, 0.25)`.
fn tint(x: u64, y: u64) -> [f32; 4] {
    let ax = (2 * x + 1) as f32 / 32.0;
    let ay = (15 - 2 * y) as f32 / 32.0;
    [ax, ay, 1.0 - ax, 0.25]
}

/// `t` swizzled by `order` (indices into `xyzw`).
fn swizzle(t: [f32; 4], order: [usize; 4]) -> [f32; 4] {
    order.map(|i| t[i])
}

fn times(t: [f32; 4], k: f32) -> [f32; 4] {
    t.map(|v| v * k)
}

/// `cg_seven`'s colors of `t`; `cg_five`'s are the first five.
fn seven(t: [f32; 4]) -> [[f32; 4]; 7] {
    [
        t,
        swizzle(t, [1, 2, 3, 0]),
        swizzle(t, [2, 3, 0, 1]),
        swizzle(t, [3, 0, 1, 2]),
        times(t, 0.5),
        times(t, 0.25),
        t.map(|v| 1.0 - v),
    ]
}

/// Color `index`'s clear in every scene.
fn clear_of(index: usize) -> [f32; 4] {
    [0.0625 * (index + 1) as f32, 0.5, 0.125 * index as f32, 1.0]
}

impl Lane {
    /// A pipeline of `cg_cover` and `fragment` of `library` drawing `RGBA16Float` at `colors`
    /// (blending each one plus one where `blend`), with `Depth32FloatStencil8` where
    /// `depth_stencil`.
    fn pipeline(
        &self,
        library: &Library,
        fragment: &str,
        colors: &[usize],
        (depth_stencil, blend): (bool, bool),
    ) -> Result<RenderPipelineState, Error> {
        let vertex = self.groups.new_function_with_name("cg_cover").unwrap();
        let fragment = library.new_function_with_name(fragment).unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        for &index in colors {
            let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
            if blend {
                color.blending_enabled = true;
                color.source_rgb_blend_factor = BlendFactor::One;
                color.destination_rgb_blend_factor = BlendFactor::One;
                color.source_alpha_blend_factor = BlendFactor::One;
                color.destination_alpha_blend_factor = BlendFactor::One;
            }
            descriptor.color_attachments[index] = Some(color);
        }
        if depth_stencil {
            descriptor.depth_format = Some(PixelFormat::Depth32FloatStencil8);
            descriptor.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        }
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    fn texture(&self, format: PixelFormat) -> Texture {
        self.sized(format, SIDE)
    }

    fn sized(&self, format: PixelFormat, side: u64) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(format, side, side, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        self.device.new_texture(&descriptor).unwrap()
    }

    fn depth_state(
        &self,
        (compare, writes): (CompareFunction, bool),
        (stencil, pass): (CompareFunction, StencilOperation),
    ) -> DepthStencilState {
        let mut face = StencilDescriptor::default();
        face.stencil_compare_function = stencil;
        face.depth_stencil_pass_operation = pass;
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = compare;
        descriptor.depth_write_enabled = writes;
        descriptor.front_face_stencil = face;
        descriptor.back_face_stencil = face;
        self.device.new_depth_stencil_state(&descriptor).unwrap()
    }

    /// `texture`'s bytes, `bytes` a texel, as `option` copies them out.
    fn read(&self, texture: &Texture, bytes: u64, option: BlitOption) -> Vec<u8> {
        let length = bytes * SIDE * SIDE;
        let buffer = super::buffer_with(&self.device, &vec![0; length as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &buffer,
            0,
            bytes * SIDE,
            length,
            option,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        super::read_buffer(&buffer, 0, length).unwrap()
    }

    /// An `RGBA16Float` texture's texels.
    fn colors(&self, texture: &Texture) -> Vec<[f32; 4]> {
        self.read(texture, 8, BlitOption::None)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|t| std::array::from_fn(|c| half(u16::from_le_bytes([t[2 * c], t[2 * c + 1]]))))
            .collect()
    }

    fn depths(&self, texture: &Texture) -> Vec<f32> {
        self.read(texture, 4, BlitOption::DepthFromDepthStencil)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|d| f32::from_le_bytes(*d))
            .collect()
    }

    fn stencils(&self, texture: &Texture) -> Vec<u8> {
        self.read(texture, 1, BlitOption::StencilFromDepthStencil)
    }

    /// Run `body` in a pass of `descriptor`, committed and waited for.
    fn run(&self, descriptor: &RenderPassDescriptor, body: impl FnOnce(&mut RenderCommandEncoder)) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
        body(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{}", self.device.name());
    }

    /// Whether the device draws `colors` colors in more than one group.
    fn splits(&self, colors: usize) -> bool {
        self.width == Width::Four && colors > 4
    }
}

/// An IEEE half's value.
fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => fraction * 2f32.powi(-24),
        31 if fraction == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    }
}

fn clear_color(index: usize) -> ClearColor {
    let [r, g, b, a] = clear_of(index).map(f64::from);
    ClearColor::new(r, g, b, a)
}

/// A pass of `RGBA16Float` `targets` at `indices`, each cleared to its value and stored.
fn cleared_pass(targets: &[Texture], indices: &[usize]) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    for (texture, &index) in targets.iter().zip(indices) {
        let color = &mut pass.color_attachments[index];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = clear_color(index);
        color.store_action = StoreAction::Store;
    }
    pass
}

/// Each pixel's expected color of each target against what `got` read.
fn check_colors(
    what: &str,
    lane: &Lane,
    got: &[(usize, Vec<[f32; 4]>)],
    expected: impl Fn(usize, u64, u64) -> [f32; 4],
) {
    for (index, texels) in got {
        for (at, texel) in texels.iter().enumerate() {
            let (x, y) = (at as u64 % SIDE, at as u64 / SIDE);
            let want = expected(*index, x, y);
            assert_eq!(
                *texel,
                want,
                "{what} color {index} at ({x}, {y}) on {} ({:?})",
                lane.device.name(),
                lane.width
            );
        }
    }
}

/// The work counts' color group fields moved by `scene`.
fn group_counts(lane: &Lane, scene: impl FnOnce()) -> (u64, u64) {
    let before: WorkCounts = lane.device.work_counts();
    scene();
    let after = lane.device.work_counts();
    (
        after.color_group_passes - before.color_group_passes,
        after.color_group_twins - before.color_group_twins,
    )
}

/// One fullscreen triangle of `fragment` in a pass of `indices` cleared to their values.
fn plain(
    lane: &Lane,
    fragment: &str,
    indices: &[usize],
    expected: impl Fn(usize, [f32; 4]) -> [f32; 4],
) {
    let pipeline = lane
        .pipeline(&lane.groups, fragment, indices, (false, false))
        .unwrap();
    let targets: Vec<Texture> = indices
        .iter()
        .map(|_| lane.texture(PixelFormat::RGBA16Float))
        .collect();
    // A fence the pass waits for and updates: encoder-level, once for every group.
    let fence = lane.device.new_fence();
    let (passes, twins) = group_counts(lane, || {
        lane.run(&cleared_pass(&targets, indices), |encoder| {
            encoder
                .wait_for_fence(&fence, RenderStages::VERTEX)
                .unwrap();
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder
                .update_fence(&fence, RenderStages::FRAGMENT)
                .unwrap();
        });
    });
    let groups = u64::from(lane.splits(indices.len()));
    assert_eq!(
        (passes, twins),
        (groups, 0),
        "{fragment} on {}",
        lane.device.name()
    );
    let got: Vec<_> = (indices.iter().copied())
        .zip(targets.iter().map(|t| lane.colors(t)))
        .collect();
    check_colors(fragment, lane, &got, |index, x, y| {
        expected(index, tint(x, y))
    });
}

fn scenes(width: Width) {
    let Some((messages, _cache, _profile, lane)) = lane(width, true) else {
        return;
    };
    // Announced once, as the first state of groups is made; and where the driver counts per
    // color, once, at the depth scene's first counted draw.
    let groups = Degradation::ColorAttachmentGroups;
    let [per_color] = PER_COLOR_COUNTS;
    let five = [0, 1, 2, 3, 4];
    plain(&lane, "cg_five", &five, |index, t| seven(t)[index]);
    plain(&lane, "cg_seven", &[0, 1, 2, 3, 4, 5, 6], |index, t| {
        seven(t)[index]
    });
    plain(&lane, "cg_sparse", &[0, 5], |index, t| match index {
        0 => t,
        _ => swizzle(t, [3, 2, 1, 0]),
    });
    depth_scene(&lane, &five);
    load_scene(&lane, &five);
    dead_scene(&lane, &five);
    // A device without a fetch form refuses the fetching pipelines, typed ([`fetches`]).
    let fetching = fetches(&lane.device);
    if fetching {
        fetch_scene(&lane);
    }
    refusals(&lane, &five, fetching);
    let announced: BTreeSet<Degradation> = (messages.take_notices().iter())
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| notice.degradation)
        .filter(|kind| [groups, per_color].contains(kind))
        .collect();
    let mut expected = BTreeSet::new();
    if width == Width::Four {
        expected.insert(groups);
    }
    if counts_per_color(&lane) {
        expected.insert(per_color);
    }
    assert_eq!(announced, expected, "on {}", lane.device.name());
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// The depth scene (the module docs).
fn depth_scene(lane: &Lane, five: &[usize]) {
    let name = lane.device.name().to_owned();
    let first = lane
        .pipeline(&lane.groups, "cg_depth", five, (true, false))
        .unwrap();
    let second = lane
        .pipeline(&lane.groups, "cg_five", five, (true, true))
        .unwrap();
    let writes = lane.depth_state(
        (CompareFunction::Less, true),
        (CompareFunction::Always, StencilOperation::Replace),
    );
    let tests = lane.depth_state(
        (CompareFunction::Less, false),
        (CompareFunction::Equal, StencilOperation::Keep),
    );
    let targets: Vec<Texture> = five
        .iter()
        .map(|_| lane.texture(PixelFormat::RGBA16Float))
        .collect();
    let ds = lane.texture(PixelFormat::Depth32FloatStencil8);
    let visibility: Buffer = super::buffer_with(&lane.device, &[0xee; 16]).unwrap();
    let mut pass = cleared_pass(&targets, five);
    pass.depth_attachment.texture = Some(ds.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = StoreAction::Store;
    pass.stencil_attachment.texture = Some(ds.clone());
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.clear_stencil = 0;
    pass.stencil_attachment.store_action = StoreAction::Store;
    pass.visibility_result_buffer = Some(visibility.clone());
    let (passes, twins) = group_counts(lane, || {
        lane.run(&pass, |encoder| {
            encoder.set_render_pipeline_state(&first).unwrap();
            encoder.set_depth_stencil_state(&writes).unwrap();
            encoder.set_stencil_reference_value(1);
            let half = ScissorRect {
                x: 0,
                y: 0,
                width: SIDE / 2,
                height: SIDE,
            };
            encoder.set_scissor_rect(half).unwrap();
            encoder
                .set_visibility_result_mode(VisibilityResultMode::Counting, 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder.set_render_pipeline_state(&second).unwrap();
            encoder.set_depth_stencil_state(&tests).unwrap();
            let whole = ScissorRect {
                width: SIDE,
                ..half
            };
            encoder.set_scissor_rect(whole).unwrap();
            encoder
                .set_visibility_result_mode(VisibilityResultMode::Counting, 8)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder
                .set_visibility_result_mode(VisibilityResultMode::Disabled, 0)
                .unwrap();
        });
    });
    let split = u64::from(lane.splits(five.len()));
    assert_eq!((passes, twins), (split, split), "depth on {name}");
    let got: Vec<_> = (five.iter().copied())
        .zip(targets.iter().map(|t| lane.colors(t)))
        .collect();
    check_colors("depth", lane, &got, |index, x, y| {
        let t = tint(x, y);
        match (x < SIDE / 2, t[0] < t[1]) {
            (true, true) => times(seven(t)[index], 2.0),
            (true, false) => seven(t)[index],
            (false, _) => clear_of(index),
        }
    });
    let depths = lane.depths(&ds);
    let stencils = lane.stencils(&ds);
    for (at, (&depth, &stencil)) in depths.iter().zip(&stencils).enumerate() {
        let (x, y) = (at as u64 % SIDE, at as u64 / SIDE);
        let t = tint(x, y);
        let left = x < SIDE / 2;
        let want = match left {
            true => (0.25 * (t[0] + t[1]), 1),
            false => (1.0, 0),
        };
        assert_eq!((depth, stencil), want, "depth at ({x}, {y}) on {name}");
    }
    let counts: Vec<u64> = super::read_buffer(&visibility, 0, 16)
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    // Counted once, in the first group's pass (the M3 Max's 32 and 22). Where the driver counts
    // each sample once per color its fragment shader writes, the draws, of four colors or five,
    // run announced (`scenes`), and each count is nonzero where Metal's is.
    match counts_per_color(lane) {
        false => assert_eq!(counts, [32, 22], "visibility on {name}"),
        true => assert!(
            counts.iter().all(|&c| c > 0),
            "visibility on {name}: {counts:?}"
        ),
    }
}

/// Whether the lane's device counts a visibility result's samples once per color its fragment
/// shader writes (`RuntimeDriverDefect::OcclusionCountPerColorOutput`).
fn counts_per_color(lane: &Lane) -> bool {
    let defect = RuntimeDriverDefect::OcclusionCountPerColorOutput;
    lane.device.properties().driver_defects.contains(&defect)
}

/// The load scene (the module docs).
fn load_scene(lane: &Lane, five: &[usize]) {
    let name = lane.device.name().to_owned();
    let pipeline = lane
        .pipeline(&lane.groups, "cg_depth", five, (true, false))
        .unwrap();
    let state = lane.depth_state(
        (CompareFunction::Less, true),
        (CompareFunction::Equal, StencilOperation::IncrementClamp),
    );
    let targets: Vec<Texture> = five
        .iter()
        .map(|_| lane.texture(PixelFormat::RGBA16Float))
        .collect();
    let ds = lane.texture(PixelFormat::Depth32FloatStencil8);
    let mut cleared = cleared_pass(&targets, five);
    cleared.depth_attachment.texture = Some(ds.clone());
    cleared.depth_attachment.load_action = LoadAction::Clear;
    cleared.depth_attachment.clear_depth = 0.15;
    cleared.depth_attachment.store_action = StoreAction::Store;
    cleared.stencil_attachment.texture = Some(ds.clone());
    cleared.stencil_attachment.load_action = LoadAction::Clear;
    cleared.stencil_attachment.clear_stencil = 5;
    cleared.stencil_attachment.store_action = StoreAction::Store;
    lane.run(&cleared, |_| {});
    let mut loaded = RenderPassDescriptor::new();
    for (texture, &index) in targets.iter().zip(five) {
        let color = &mut loaded.color_attachments[index];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = match index {
            4 => StoreAction::Unknown,
            _ => StoreAction::Store,
        };
    }
    loaded.depth_attachment.texture = Some(ds.clone());
    loaded.depth_attachment.load_action = LoadAction::Load;
    loaded.depth_attachment.store_action = StoreAction::Unknown;
    loaded.stencil_attachment.texture = Some(ds.clone());
    loaded.stencil_attachment.load_action = LoadAction::Load;
    loaded.stencil_attachment.store_action = StoreAction::Store;
    let (passes, twins) = group_counts(lane, || {
        lane.run(&loaded, |encoder| {
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder.set_depth_stencil_state(&state).unwrap();
            encoder.set_stencil_reference_value(5);
            let top = ScissorRect {
                x: 0,
                y: 0,
                width: SIDE,
                height: SIDE / 2,
            };
            encoder.set_scissor_rect(top).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder
                .set_color_store_action(StoreAction::Store, 4)
                .unwrap();
            encoder.set_depth_store_action(StoreAction::Store).unwrap();
        });
    });
    let split = u64::from(lane.splits(five.len()));
    assert_eq!((passes, twins), (split, split), "load on {name}");
    let cleared_depth = 0.15f32;
    let passes_at = |x: u64, y: u64| {
        let t = tint(x, y);
        y < SIDE / 2 && 0.25 * (t[0] + t[1]) < cleared_depth
    };
    let got: Vec<_> = (five.iter().copied())
        .zip(targets.iter().map(|t| lane.colors(t)))
        .collect();
    check_colors("load", lane, &got, |index, x, y| match passes_at(x, y) {
        true => seven(tint(x, y))[index],
        false => clear_of(index),
    });
    let depths = lane.depths(&ds);
    let stencils = lane.stencils(&ds);
    for (at, (&depth, &stencil)) in depths.iter().zip(&stencils).enumerate() {
        let (x, y) = (at as u64 % SIDE, at as u64 / SIDE);
        let t = tint(x, y);
        let want = match passes_at(x, y) {
            true => (0.25 * (t[0] + t[1]), 6),
            false => (cleared_depth, 5),
        };
        assert_eq!((depth, stencil), want, "load depth at ({x}, {y}) on {name}");
    }
}

/// The fetch scene (the module docs).
fn fetch_scene(lane: &Lane) {
    let sparse = [0, 5];
    let pipeline = lane
        .pipeline(&lane.fetch, "cg_fetch_sparse", &sparse, (false, false))
        .unwrap();
    let targets: Vec<Texture> = sparse
        .iter()
        .map(|_| lane.texture(PixelFormat::RGBA16Float))
        .collect();
    lane.run(&cleared_pass(&targets, &sparse), |encoder| {
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    });
    let got: Vec<_> = (sparse.iter().copied())
        .zip(targets.iter().map(|t| lane.colors(t)))
        .collect();
    check_colors("fetch", lane, &got, |index, x, y| {
        let t = tint(x, y);
        match index {
            0 => t,
            _ => {
                let last = clear_of(5);
                let w = swizzle(t, [3, 2, 1, 0]);
                std::array::from_fn(|c| last[c] * 0.5 + w[c])
            }
        }
    });
}

/// The pipeline of `cg_cover` and `fragment` of `library` drawing `RGBA16Float` at `colors`, the
/// fragment function specialized with its `bool` function constant 0 `reads`.
fn dead_pipeline(
    lane: &Lane,
    (library, fragment): (&Library, &str),
    colors: &[usize],
    reads: bool,
) -> Result<RenderPipelineState, Error> {
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::Bool, [u8::from(reads)]),
    );
    let vertex = lane.groups.new_function_with_name("cg_cover").unwrap();
    let fragment = library
        .new_function_with_name_constant_values(fragment, &constants)
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    for &index in colors {
        descriptor.color_attachments[index] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    }
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// The dead read scene (the module docs).
fn dead_scene(lane: &Lane, five: &[usize]) {
    let name = lane.device.name().to_owned();
    // What `dfr_buffer` reads where its read is live.
    let bytes: Vec<u8> = [0.125f32, 0.25, 0.375, 0.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let buffer = lane
        .device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let functions = [
        (&lane.dead, "dfg_branch"),
        (&lane.dead, "dfg_gated"),
        (&lane.rog, "dfr_color"),
        (&lane.rog, "dfr_buffer"),
    ];
    for (library, fragment) in functions {
        let pipeline = dead_pipeline(lane, (library, fragment), five, false)
            .unwrap_or_else(|e| panic!("{fragment} dead on {name}: {e}"));
        let targets: Vec<Texture> = five
            .iter()
            .map(|_| lane.texture(PixelFormat::RGBA16Float))
            .collect();
        let (passes, twins) = group_counts(lane, || {
            lane.run(&cleared_pass(&targets, five), |encoder| {
                encoder.set_render_pipeline_state(&pipeline).unwrap();
                encoder.set_fragment_buffer(Some(&buffer), 0, 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            });
        });
        let groups = u64::from(lane.splits(five.len()));
        assert_eq!((passes, twins), (groups, 0), "{fragment} dead on {name}");
        let got: Vec<_> = (five.iter().copied())
            .zip(targets.iter().map(|t| lane.colors(t)))
            .collect();
        check_colors(fragment, lane, &got, |index, x, y| {
            let t = tint(x, y);
            match index {
                4 => t,
                _ => seven(t)[index],
            }
        });
        eprintln!(
            "COLOR-GROUPS {fragment} dead on {name} ({:?}): Metal's colors, {passes} group passes \
             past the first",
            lane.width
        );
    }
    if lane.width == Width::Four {
        let limit = Error::Unsupported(Unsupported::needs(
            Feature::InterfaceComponents,
            Capability::Limit {
                limit: VkLimit::MaxColorAttachments,
                required: 5,
                available: 4,
            },
        ));
        for (library, fragment) in [functions[0], functions[2], functions[3]] {
            let live = dead_pipeline(lane, (library, fragment), five, true);
            assert_eq!(live.unwrap_err(), limit, "{fragment} live on {name}");
        }
    }
    // A color input without an attachment, read or not: Metal's pipeline error.
    let unattached = Error::Compile(CompileError::InvalidDescriptor(
        DescriptorError::ColorAttachment {
            index: 1,
            metal: MetalAnswer::NilWithError,
        },
    ));
    for reads in [false, true] {
        let made = dead_pipeline(lane, (&lane.rog, "dfr_lacks"), &[0], reads);
        assert_eq!(
            made.unwrap_err(),
            unattached,
            "dfr_lacks ({reads}) on {name}"
        );
    }
    eprintln!("COLOR-GROUPS dfr_lacks on {name}: refused dead and live, as Metal");
}

/// What the restricted device refuses, and the device as it is draws (the module docs); on a
/// device without a fetch form (not `fetching`), `cg_fetch` is refused there as `FramebufferFetch`.
fn refusals(lane: &Lane, five: &[usize], fetching: bool) {
    let name = lane.device.name().to_owned();
    // The pipeline's limit refusal: its attachments reach index 4 on a device of four.
    let limit = Error::Unsupported(Unsupported::needs(
        Feature::InterfaceComponents,
        Capability::Limit {
            limit: VkLimit::MaxColorAttachments,
            required: 5,
            available: 4,
        },
    ));
    let counts = lane.pipeline(&lane.groups, "cg_counts", five, (false, false));
    let fetch = lane.pipeline(&lane.fetch, "cg_fetch", five, (false, false));
    let four = lane
        .pipeline(&lane.groups, "cg_five", &five[..4], (false, false))
        .unwrap();
    match lane.width {
        Width::Four => {
            assert_eq!(counts.unwrap_err(), limit, "cg_counts on {name}");
            assert_eq!(fetch.unwrap_err(), limit, "cg_fetch on {name}");
            let targets: Vec<Texture> = five
                .iter()
                .map(|_| lane.texture(PixelFormat::RGBA16Float))
                .collect();
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut encoder = command_buffer
                .render_command_encoder(&cleared_pass(&targets, five))
                .unwrap();
            let refused = encoder.set_render_pipeline_state(&four);
            assert!(
                matches!(
                    refused,
                    Err(Error::NotImplemented(RuntimeGap::ColorAttachmentGroups))
                ),
                "{refused:?} on {name}"
            );
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        Width::Device => {
            counts.unwrap();
            match fetching {
                true => {
                    fetch.unwrap();
                }
                false => match fetch {
                    Err(Error::Unsupported(refusal))
                        if refusal.feature == Feature::FramebufferFetch => {}
                    Err(error) => panic!("cg_fetch on {name} without a fetch form: {error}"),
                    Ok(_) => panic!("cg_fetch on {name}: made without a fetch form"),
                },
            }
        }
    }
}

/// The scenes on the device restricted to four color attachments (the module docs).
#[test]
fn passes_of_more_colors_than_the_device_draws_at_once_draw_metal_s_answers_in_groups() {
    scenes(Width::Four);
}

/// The same scenes on the device as it is, where nothing groups.
#[test]
fn the_same_passes_draw_metal_s_answers_where_the_device_draws_every_color_at_once() {
    scenes(Width::Device);
}

/// The cost run's target side, draws a pass, and command buffers untimed and timed.
const COST_SIDE: u64 = 512;
const COST_DRAWS: usize = 16;
const COST_WARM: usize = 3;
const COST_TIMED: usize = 15;

/// The medians over [`COST_TIMED`] command buffers of one pass of [`COST_DRAWS`] fullscreen
/// `cg_five` triangles into `RGBA16Float` `colors` of [`COST_SIDE`], each cleared and stored, and
/// where `depth` a `Depth32FloatStencil8` attachment loaded and stored, which each draw writes:
/// the encoding (the encoder's creation through `end_encoding`) and the commit to completion.
fn cost(lane: &Lane, colors: &[usize], depth: bool) -> (Duration, Duration) {
    let pipeline = lane
        .pipeline(&lane.groups, "cg_five", colors, (depth, false))
        .unwrap();
    let writes = lane.depth_state(
        (CompareFunction::Always, true),
        (CompareFunction::Always, StencilOperation::Keep),
    );
    let targets: Vec<Texture> = colors
        .iter()
        .map(|_| lane.sized(PixelFormat::RGBA16Float, COST_SIDE))
        .collect();
    let mut pass = cleared_pass(&targets, colors);
    if depth {
        let ds = lane.sized(PixelFormat::Depth32FloatStencil8, COST_SIDE);
        pass.depth_attachment.texture = Some(ds.clone());
        pass.depth_attachment.load_action = LoadAction::Load;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.stencil_attachment.texture = Some(ds);
        pass.stencil_attachment.load_action = LoadAction::Load;
        pass.stencil_attachment.store_action = StoreAction::Store;
    }
    let mut encoding = Vec::with_capacity(COST_TIMED);
    let mut completion = Vec::with_capacity(COST_TIMED);
    for run in 0..COST_WARM + COST_TIMED {
        let command_buffer = lane.queue.command_buffer().unwrap();
        let start = Instant::now();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        if depth {
            encoder.set_depth_stencil_state(&writes).unwrap();
        }
        for _ in 0..COST_DRAWS {
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        let encoded = start.elapsed();
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{}", lane.device.name());
        if run >= COST_WARM {
            encoding.push(encoded);
            completion.push(committed.elapsed());
        }
    }
    let median = |mut times: Vec<Duration>| {
        times.sort();
        times[times.len() / 2]
    };
    (median(encoding), median(completion))
}

/// What drawing a pass in two color groups costs on the device restricted to four color
/// attachments, without the validation layer, against a pass of four colors of the same draws
/// (the fifth color dropped): a `COLOR-GROUP-COST` line each, with and without a loaded depth and
/// stencil attachment, whose second group's twin is copied from it. Report only, in release
/// builds.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn color_group_costs() {
    let Some((_messages, _cache, _profile, lane)) = lane(Width::Four, false) else {
        return;
    };
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    for depth in [false, true] {
        let (four_encode, four_done) = cost(&lane, &[0, 1, 2, 3], depth);
        let (five_encode, five_done) = cost(&lane, &[0, 1, 2, 3, 4], depth);
        eprintln!(
            "COLOR-GROUP-COST {} {COST_SIDE}x{COST_SIDE}, {COST_DRAWS} draws{}: two groups of \
             five colors encode {:.3} ms, commit to completion {:.3} ms; four colors encode \
             {:.3} ms, commit to completion {:.3} ms ({:.2}x, {:.2}x)",
            lane.device.name(),
            if depth {
                ", depth and stencil loaded"
            } else {
                ""
            },
            ms(five_encode),
            ms(five_done),
            ms(four_encode),
            ms(four_done),
            ms(five_encode) / ms(four_encode),
            ms(five_done) / ms(four_done),
        );
    }
}
