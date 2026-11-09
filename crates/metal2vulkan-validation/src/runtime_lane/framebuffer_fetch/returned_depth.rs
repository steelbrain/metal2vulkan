//! A fetching function that returns its depth, beside a depth attachment, on the device the cfg
//! selects, under the validation layer.
//!
//! `fetch_returned_depth` (`dsx_fetch_returned_depth`) returns half of the color it fetched plus a
//! quarter, and the depth its buffer holds. Each case is a pass of one draw of the fullscreen
//! triangle over a fresh 4² `R32Float` target cleared to 0.75, beside a `Depth32Float_Stencil8`
//! attachment cleared to depth 1 and stencil 0, no stencil test. Metal tests the depth the
//! function returns, clamped by the depth clip mode (a probe): in clip mode to `[0, 1]`, in clamp
//! mode to the viewport's range; any value with its sign bit set as +0, a negative NaN staying 0 in
//! clamp mode, past the range; a positive NaN fails, `Always` included. A fragment that passes
//! stores 0.625 and, under a written depth state, its clamped depth; one that fails keeps 0.75 and
//! depth 1.
//!
//! In the pixel interlock (RADV, ANV, NVIDIA, MoltenVK) the function stores its color before the
//! device's tests, so a draw whose depth test can drop a fragment runs the variant that tests the
//! returned depth in the shader; through input attachments (lavapipe) the device's own test takes
//! the depth the function writes, clamped as the runtime clamps it. Each case prints its outcome
//! and form.

use super::{form, library};
use crate::runtime_lane::buffer_with;
use crate::runtime_lane::resolves::{lane, quiet, FULL};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, DepthClipMode, PixelFormat, PrimitiveType};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, LoadAction, PipelineOptions, RenderPassDescriptor, StoreAction,
    Viewport,
};

/// The target's side.
const SIDE: usize = 4;

/// The value each target is cleared to, and what a fragment that passes stores.
const CLEAR: f32 = 0.75;
const STORED: f32 = 0.625;

/// The depth each depth attachment is cleared to.
const CLEAR_DEPTH: f32 = 1.0;

/// A positive and a negative quiet NaN.
const NAN: u32 = 0x7FC0_0000;
const NEGATIVE_NAN: u32 = 0xFFC0_0000;

/// One case: its name, the depth compare function and whether it writes, the clip mode and the
/// viewport's depth range, the returned depth's bits, and Metal's answer: whether the fragment
/// passes, and the depth's bits it stores where it passes and the state writes.
struct Case {
    name: &'static str,
    compare: CompareFunction,
    write: bool,
    mode: DepthClipMode,
    range: (f64, f64),
    returned: u32,
    passes: bool,
    stored: u32,
}

/// Metal's answer for each case (the module docs).
fn cases() -> Vec<Case> {
    use CompareFunction::{Always, Greater, Less, LessEqual};
    use DepthClipMode::{Clamp, Clip};
    let unit = (0.0, 1.0);
    let range = (0.25, 0.75);
    let case = |name, compare, write, mode, range, returned: u32, passes, stored: u32| Case {
        name,
        compare,
        write,
        mode,
        range,
        returned,
        passes,
        stored,
    };
    let f = f32::to_bits;
    vec![
        case(
            "0.25, less, written",
            Less,
            true,
            Clip,
            unit,
            f(0.25),
            true,
            f(0.25),
        ),
        case(
            "0.25, greater, unwritten",
            Greater,
            false,
            Clip,
            unit,
            f(0.25),
            false,
            0,
        ),
        case(
            "1.5 as 1, less-equal, written",
            LessEqual,
            true,
            Clip,
            unit,
            f(1.5),
            true,
            f(1.0),
        ),
        case(
            "1.5 as 1, less, written",
            Less,
            true,
            Clip,
            unit,
            f(1.5),
            false,
            0,
        ),
        case(
            "-0.5 as +0, less, written",
            Less,
            true,
            Clip,
            unit,
            f(-0.5),
            true,
            0,
        ),
        case(
            "-0 as +0, always, written",
            Always,
            true,
            Clip,
            unit,
            0x8000_0000,
            true,
            0,
        ),
        case(
            "a positive NaN, always, written",
            Always,
            true,
            Clip,
            unit,
            NAN,
            false,
            0,
        ),
        case(
            "a negative NaN as 0, less, written",
            Less,
            true,
            Clip,
            unit,
            NEGATIVE_NAN,
            true,
            0,
        ),
        case(
            "clamp mode, 0.1 to 0.25",
            Less,
            true,
            Clamp,
            range,
            f(0.1),
            true,
            f(0.25),
        ),
        case(
            "clamp mode, 1.5 to 0.75",
            Less,
            true,
            Clamp,
            range,
            f(1.5),
            true,
            f(0.75),
        ),
        case(
            "clamp mode, -0.5 to 0.25",
            Less,
            true,
            Clamp,
            range,
            f(-0.5),
            true,
            f(0.25),
        ),
        case(
            "clamp mode, a negative NaN as 0",
            Less,
            true,
            Clamp,
            range,
            NEGATIVE_NAN,
            true,
            0,
        ),
    ]
}

/// Metal's answer in every case (the module docs), on every device.
#[test]
fn a_fetch_that_returns_its_depth_is_tested_by_the_depth_it_returns() {
    let (messages, _cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let vertices = library(&lane, "rt6_resolves");
    let returning = library(&lane, "dsx_fetch_returned_depth");
    let device = lane.device.name();
    let function = |library: &metal2vulkan_runtime::Library, name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let ds = PixelFormat::Depth32FloatStencil8;
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function(&vertices, FULL)));
    descriptor.fragment = Some(StageSource::Function(function(
        &returning,
        "fetch_returned_depth",
    )));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
    descriptor.depth_format = Some(ds);
    descriptor.stencil_format = Some(ds);
    let state = lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap_or_else(|error| panic!("the pipeline on {device}: {error}"));
    let floats = |bits: u32| bits.to_le_bytes().repeat(SIDE * SIDE);
    for case in cases() {
        let name = case.name;
        let target = lane.single(PixelFormat::R32Float);
        let depth_texture = lane.single(ds);
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(f64::from(CLEAR), 0.0, 0.0, 0.0);
        color.store_action = StoreAction::Store;
        pass.depth_attachment.texture = Some(depth_texture.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = f64::from(CLEAR_DEPTH);
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.stencil_attachment.texture = Some(depth_texture.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.clear_stencil = 0;
        pass.stencil_attachment.store_action = StoreAction::Store;
        let mut d = DepthStencilDescriptor::default();
        d.depth_compare_function = case.compare;
        d.depth_write_enabled = case.write;
        let ds_state = lane.device.new_depth_stencil_state(&d).unwrap();
        lane.run(&pass, |lane, encoder, held: &mut Vec<Buffer>| {
            let z = buffer_with(&lane.device, &super::Z.to_le_bytes()).unwrap();
            let returned = buffer_with(&lane.device, &case.returned.to_le_bytes()).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&returned), 0, 0).unwrap();
            encoder.set_depth_stencil_state(&ds_state).unwrap();
            let (near, far) = case.range;
            encoder
                .set_viewport(Viewport::new(0.0, 0.0, SIDE as f64, SIDE as f64, near, far))
                .unwrap();
            encoder.set_depth_clip_mode(case.mode).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap_or_else(|error| panic!("{name}: the draw on {device}: {error}"));
            held.push(z);
            held.push(returned);
        })
        .unwrap_or_else(|error| panic!("{name}: the pass on {device}: {error}"));
        let color = lane.read(&target, (0, 0), 4, BlitOption::None);
        let want_color = floats(f32::to_bits(if case.passes { STORED } else { CLEAR }));
        assert_eq!(color, want_color, "{name}: the target on {device}");
        let depth_read = lane.read(&depth_texture, (0, 0), 4, BlitOption::DepthFromDepthStencil);
        let want_depth = floats(match case.passes && case.write {
            true => case.stored,
            false => CLEAR_DEPTH.to_bits(),
        });
        assert_eq!(depth_read, want_depth, "{name}: the depth on {device}");
        eprintln!(
            "framebuffer fetch returning its depth, {name}, on {device} through {}: Metal's answer",
            form(&lane)
        );
    }
    quiet(lane, &messages);
}
