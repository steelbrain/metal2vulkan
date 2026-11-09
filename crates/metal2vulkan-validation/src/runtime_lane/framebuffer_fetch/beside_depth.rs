//! A fetching function that also writes other memory, beside a depth and stencil attachment, on the
//! device the cfg selects, under the validation layer.
//!
//! `fetch_coherent` (`dvp_rog_coherence`) adds 1 to `log[y · 8 + x]` and returns half of what it
//! fetched plus a quarter. Each case is a pass of one draw of the fullscreen triangle at depth 0.5
//! over a fresh 4² `R32Float` target cleared to 0.75, beside a `Depth32Float_Stencil8` attachment
//! cleared to depth 1 and stencil 0, the counters zero. Metal runs the function for every
//! fragment and tests after it (device writes for every fragment, those the tests
//! drop included, 4,096 of 4,096): every covered pixel's counter is 1; a fragment that passes the
//! tests stores 0.625 and, under a written depth state, depth 0.5; one that fails keeps 0.75 and
//! depth 1. The states:
//!
//! - none, and depth `Always` written: no test drops a fragment (`FragmentTests::Pass`);
//! - depth `Less` unwritten (every fragment passes) and depth `Greater` unwritten (every fragment
//!   fails): a depth test that can drop a fragment and writes nothing (`Depth`, unwritten);
//! - a stencil `Equal` of reference 1 (every fragment fails): a stencil test that can drop a
//!   fragment (`Drop`);
//! - depth `Less` written (`Depth`, written).
//!
//! Through input attachments (lavapipe) the function runs without early tests, its color written
//! through the attachment after the device's tests: every state draws Metal's answer. In the
//! pixel interlock (RADV, ANV, NVIDIA, MoltenVK) the color is stored in the shader before the
//! device's tests: the two `Pass` states draw Metal's answer; the three depth states
//! (`FragmentTests::Depth`) draw it too, the function testing the depth itself against the
//! depth's stand-in; and so does the stencil state, the function testing the stencil first against
//! the stencil's stand-in (Refused at the draw before). Each case's outcome and form is printed.

use super::{form, library, Z};
use crate::runtime_lane::resolves::{lane, quiet, FULL};
use crate::runtime_lane::{buffer_with, read_buffer};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, Library, LoadAction, PipelineOptions, RenderPassDescriptor,
    StoreAction,
};

/// The target's side, and the stride `fetch_coherent` indexes its counters by.
const SIDE: usize = 4;
const STRIDE: usize = 8;

/// The value each target is cleared to, and what a fragment that passes stores.
const CLEAR: f32 = 0.75;
const STORED: f32 = 0.625;

/// Each covered pixel's counter: its index in the counters `fetch_coherent` writes.
fn covered(index: usize) -> bool {
    index % STRIDE < SIDE && index / STRIDE < SIDE
}

/// Metal's answer for every state (the module docs), on every device.
#[test]
fn a_fetch_that_writes_memory_beside_depth_runs_metals_draws() {
    let (messages, _cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let vertices = library(&lane, "rt6_resolves");
    let coherent = library(&lane, "dvp_rog_coherence");
    let device = lane.device.name();
    let function = |library: &Library, name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let ds = PixelFormat::Depth32FloatStencil8;
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function(&vertices, FULL)));
    descriptor.fragment = Some(StageSource::Function(function(&coherent, "fetch_coherent")));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
    descriptor.depth_format = Some(ds);
    descriptor.stencil_format = Some(ds);
    let state = lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap_or_else(|error| panic!("the pipeline on {device}: {error}"));
    let depth = |compare: CompareFunction, write: bool| {
        let mut d = DepthStencilDescriptor::default();
        d.depth_compare_function = compare;
        d.depth_write_enabled = write;
        d
    };
    let mut equal = DepthStencilDescriptor::default();
    for face in [&mut equal.front_face_stencil, &mut equal.back_face_stencil] {
        face.stencil_compare_function = CompareFunction::Equal;
    }
    // (name, state, stencil reference, whether the fragments pass, whether they write depth)
    let cases = [
        ("no state", None, 0, true, false),
        (
            "depth always, written",
            Some(depth(CompareFunction::Always, true)),
            0,
            true,
            true,
        ),
        // The depth states run in the interlock: the function tests the depth itself.
        (
            "depth less, unwritten",
            Some(depth(CompareFunction::Less, false)),
            0,
            true,
            false,
        ),
        (
            "depth greater, unwritten",
            Some(depth(CompareFunction::Greater, false)),
            0,
            false,
            false,
        ),
        // The stencil state runs in the interlock too: the function tests the stencil itself.
        ("stencil equal, failing", Some(equal), 1, false, false),
        (
            "depth less, written",
            Some(depth(CompareFunction::Less, true)),
            0,
            true,
            true,
        ),
    ];
    let floats = |value: f32| value.to_le_bytes().repeat(SIDE * SIDE);
    for (case, ds_state, reference, passes, writes) in cases {
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
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.stencil_attachment.texture = Some(depth_texture.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.clear_stencil = 0;
        pass.stencil_attachment.store_action = StoreAction::Store;
        let log = buffer_with(&lane.device, &[0u8; 4 * STRIDE * STRIDE]).unwrap();
        let ds_state = ds_state.map(|d| lane.device.new_depth_stencil_state(&d).unwrap());
        let mut drawn = None;
        lane.run(&pass, |lane, encoder, held: &mut Vec<Buffer>| {
            let z = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&log), 0, 0).unwrap();
            if let Some(ds_state) = &ds_state {
                encoder.set_depth_stencil_state(ds_state).unwrap();
            }
            encoder.set_stencil_reference_value(reference);
            drawn = Some(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            held.push(z);
        })
        .unwrap_or_else(|error| panic!("{case}: the pass on {device}: {error}"));
        drawn
            .expect("the draw")
            .unwrap_or_else(|error| panic!("{case} on {device}: {error}"));
        let counters: Vec<u32> = read_buffer(&log, 0, (4 * STRIDE * STRIDE) as u64)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let want_counters: Vec<u32> = (0..STRIDE * STRIDE)
            .map(|i| u32::from(covered(i)))
            .collect();
        assert_eq!(counters, want_counters, "{case}: the counters on {device}");
        let color = lane.read(&target, (0, 0), 4, BlitOption::None);
        let want_color = floats(if passes { STORED } else { CLEAR });
        assert_eq!(color, want_color, "{case}: the target on {device}");
        let depth_read = lane.read(&depth_texture, (0, 0), 4, BlitOption::DepthFromDepthStencil);
        let want_depth = floats(if passes && writes { Z } else { 1.0 });
        assert_eq!(depth_read, want_depth, "{case}: the depth on {device}");
        eprintln!(
            "framebuffer fetch beside depth, a memory-writing function, {case}, on {device} \
             through {}: Metal's answer",
            form(&lane)
        );
    }
    quiet(lane, &messages);
}
