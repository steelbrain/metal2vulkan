//! Rate-mapped render passes through the runtime on the device the cfg selects, under the
//! validation layer. The committed public rate-mapped cases run in `renders`' census; this module
//! draws the public case `fragment-positions-in-a-half-quality- rate-mapped-pass`'s pipeline (its
//! generated fullscreen triangle and `fragment_rate_map_position`, which writes each fragment's
//! `[[position]]`) in passes of its own, over a map of a 64-pixel screen at quality 0.5, whose
//! physical size is 33 by 33, and over the identity of that screen, as our probes drew them on an
//! M3 Max and an M4:
//!
//! - Into attachments of 33, 48 and 64 pixels, the default viewport (the screen) shades the 33 by
//!   33 physical pixels and none past them, each at its centre; a viewport over the screen's left
//!   half shades physical columns 0 to 15; a scissor of the whole screen clips nothing, and one
//!   past the screen is clipped to it (probe 16's `sc_past_att_screen`, which Metal's validation
//!   stops: "render pass width(64)"); over the identity, a 16-pixel scissor keeps 16 columns.
//! - Plain Metal runs two passes its validation stops: attachments of 32 and 16 pixels, smaller
//!   than the physical size, take the physical pixels in them (probe 16's `att_32` and
//!   `att_small`), and an array length of 2 past the map's one layer rasterizes by that layer, a
//!   draw of slice 0 shading slice 0 alone (`map1_array_flat`).
//! - The pass is refused as Metal's validation refuses it: an array length past the layers of a
//!   map whose layers differ, and a pass without attachments larger than the screen. A scissor
//!   that cuts into the viewport over a map that is not the identity is `NotImplemented`, as the
//!   region path does not draw it.
//! - A pass over a map of no layers runs its clear and store, and its draws write nothing, as our
//!   probe's cases drew on an M4 wherever the map was made while no map of layers had been
//!   released. Made after one was, Metal draws physical rows 0 to 15 (an M4 and an M3 Max alike),
//!   the runtime nothing, and announces it.

use super::degrade_notices::{
    assert_made_once, counts, made_render_kinds, shared_in_system_memory,
};
use super::rate_maps::descriptor;
use super::renders::map_render;
use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device, Messages, ScratchCache, WAIT,
};
use crate::candidate::runtime_case::{render_pipeline_descriptor, render_pipeline_state};
use crate::gpu::TEST_DEVICE;
use crate::store::CorpusStore;
use metal2vulkan::metal::Mutability;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    Degradation, Device, Error, EventRef, Invalid, LoadAction, Problem, RasterizationRateMap,
    Region, RegionRefusal, RenderCommandEncoder, RenderPassDescriptor, RenderPassProblem,
    RenderPipelineState, ResourceOptions, RuntimeGap, ScissorRect, StoreAction, Texture,
    TextureDescriptor, Viewport,
};
use std::collections::BTreeSet;
use std::path::Path;

/// The public case whose pipeline the passes draw.
const CASE: &str = "fragment-positions-in-a-half-quality-rate-mapped-pass";

/// The maps' screen side.
const SCREEN: u32 = 64;

/// The case's pipeline and its fragment buffer (the screen) on one device.
struct Fixture {
    device: Device,
    state: RenderPipelineState,
    screen: Buffer,
    /// The kinds its state's making announced where the device had not
    /// (`made_render_kinds`), and its `Shared` buffer's system memory.
    made: std::collections::BTreeSet<Degradation>,
}

impl Fixture {
    fn new(device: &Device) -> Fixture {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
        let store = CorpusStore::new(&root);
        let case = (0..64)
            .flat_map(|shard| store.read_case_shard(shard).unwrap())
            .find(|case| case.name == CASE)
            .expect("the rate-mapped position case");
        let case = map_render(&root, &store, case).unwrap();
        let screen = [SCREEN as f32; 2]
            .iter()
            .flat_map(|side| side.to_le_bytes())
            .collect::<Vec<u8>>();
        let descriptor = render_pipeline_descriptor(device, &case.inputs()).unwrap();
        let mut made = made_render_kinds(device, &descriptor);
        if shared_in_system_memory(device) {
            made.insert(Degradation::SharedInSystemMemory);
        }
        Fixture {
            state: render_pipeline_state(device, &case.inputs()).unwrap(),
            screen: buffer_with(device, &screen).unwrap(),
            device: device.clone(),
            made,
        }
    }

    /// The map of a screen of `side` pixels a side and these layers' qualities.
    fn map_of(&self, side: u32, layers: &[(f32, f32)]) -> RasterizationRateMap {
        self.device
            .new_rasterization_rate_map(&descriptor(side, side, layers))
            .unwrap()
    }

    /// The map of the screen at `quality` on both axes.
    fn map(&self, quality: f32) -> RasterizationRateMap {
        self.device
            .new_rasterization_rate_map(&descriptor(SCREEN, SCREEN, &[(quality, quality)]))
            .unwrap()
    }

    /// An `RGBA32Float` attachment of `side` pixels a side, the CPU reading it.
    fn target(&self, side: u64) -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, side, side, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        self.device.new_texture(&descriptor).unwrap()
    }

    /// An `RGBA32Float` 2D array attachment of `side` pixels a side and `slices` slices, the CPU
    /// reading it.
    fn target_array(&self, side: u64, slices: u64) -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, side, side, false);
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = slices;
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        self.device.new_texture(&descriptor).unwrap()
    }

    /// Run a pass over `map` into `target`, cleared to -1, its encoder's state set by `state`,
    /// with one draw of the case's triangle.
    fn draw(
        &self,
        map: &RasterizationRateMap,
        target: &Texture,
        state: impl FnOnce(&mut RenderCommandEncoder),
    ) -> Result<(), Error> {
        self.draw_in(&pass(map, target), state)
    }

    /// Run `pass` with its encoder's state set by `state` and one draw of the case's triangle.
    fn draw_in(
        &self,
        pass: &RenderPassDescriptor,
        state: impl FnOnce(&mut RenderCommandEncoder),
    ) -> Result<(), Error> {
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(pass)?;
        encoder.set_render_pipeline_state(&self.state)?;
        encoder.set_fragment_buffer(Some(&self.screen), 0, 0)?;
        state(&mut encoder);
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(())
    }
}

/// A pass over `map` into `target`, cleared to -1 and stored.
fn pass(map: &RasterizationRateMap, target: &Texture) -> RenderPassDescriptor {
    let mut pass = plain(target);
    pass.rasterization_rate_map = Some(map.clone());
    pass
}

/// A pass into `target` without a map, cleared to -1 and stored.
fn plain(target: &Texture) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.clear_color = ClearColor::new(-1.0, -1.0, -1.0, -1.0);
    attachment.store_action = StoreAction::Store;
    pass
}

/// `fixture`'s attachment of `side` pixels a side holding 0.25 in every channel, so that a
/// pass's clear shows.
fn filled(fixture: &Fixture, side: u64) -> Texture {
    let target = fixture.target(side);
    let bytes: Vec<u8> = (0..side * side * 4)
        .flat_map(|_| 0.25f32.to_le_bytes())
        .collect();
    target
        .replace_region(
            Region::new_2d(0, 0, side, side),
            0,
            0,
            &bytes,
            16 * side,
            16 * side * side,
        )
        .unwrap();
    target
}

/// Whether every channel of every texel of `target` is the passes' clear, -1.
fn cleared(target: &Texture) -> bool {
    let side = target.width();
    let mut bytes = vec![0u8; (side * side * 16) as usize];
    target
        .get_bytes(
            &mut bytes,
            side * 16,
            0,
            Region::new_2d(0, 0, side, side),
            0,
            0,
        )
        .unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .all(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) == -1.0)
}

/// What a draw shaded into `target`'s first slice (`shaded_in`).
fn shaded(target: &Texture) -> (usize, [u64; 4], [[f32; 2]; 2]) {
    shaded_in(target, 0)
}

/// What a draw shaded into `slice` of `target`: the count of pixels it wrote, their bounding box
/// `[x0, y0, x1, y1]`, and the positions written at its first and last corner; zeros where it
/// wrote none.
fn shaded_in(target: &Texture, slice: u64) -> (usize, [u64; 4], [[f32; 2]; 2]) {
    let side = target.width();
    let mut bytes = vec![0u8; (side * side * 16) as usize];
    target
        .get_bytes(
            &mut bytes,
            side * 16,
            0,
            Region::new_2d(0, 0, side, side),
            0,
            slice,
        )
        .unwrap();
    let texel = |x: u64, y: u64| {
        let at = ((y * side + x) * 16) as usize;
        let float =
            |k: usize| f32::from_le_bytes(bytes[at + 4 * k..at + 4 * k + 4].try_into().unwrap());
        [float(0), float(1)]
    };
    let mut count = 0;
    let mut bounds = [u64::MAX, u64::MAX, 0, 0];
    for y in 0..side {
        for x in 0..side {
            if texel(x, y)[0] >= 0.0 {
                count += 1;
                bounds = [
                    bounds[0].min(x),
                    bounds[1].min(y),
                    bounds[2].max(x),
                    bounds[3].max(y),
                ];
            }
        }
    }
    if count == 0 {
        return (0, [0; 4], [[0.0; 2]; 2]);
    }
    let positions = [texel(bounds[0], bounds[1]), texel(bounds[2], bounds[3])];
    (count, bounds, positions)
}

/// Probes 16 and 16b's draws: what a rate-mapped pass shades, and where (the module docs).
#[test]
fn a_rate_mapped_pass_shades_its_physical_pixels_region_by_region() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let fixture = Fixture::new(&device);
    let half = fixture.map(0.5);
    let whole = (33 * 33, [0, 0, 32, 32], [[0.5, 0.5], [32.5, 32.5]]);
    for side in [33, 48, 64] {
        let target = fixture.target(side);
        fixture.draw(&half, &target, |_| {}).unwrap();
        assert_eq!(shaded(&target), whole, "attachments of {side}");
    }
    let target = fixture.target(33);
    fixture
        .draw(&half, &target, |encoder| {
            encoder
                .set_viewport(Viewport::new(0.0, 0.0, 32.0, 64.0, 0.0, 1.0))
                .unwrap();
        })
        .unwrap();
    assert_eq!(
        shaded(&target),
        (16 * 33, [0, 0, 15, 32], [[0.5, 0.5], [15.5, 32.5]]),
        "the left half's viewport"
    );
    let target = fixture.target(33);
    fixture
        .draw(&half, &target, |encoder| {
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, 64, 64))
                .unwrap();
        })
        .unwrap();
    assert_eq!(shaded(&target), whole, "the screen's scissor");
    // A scissor past the screen is clipped to it (probe 16's `sc_past_att_screen`).
    let target = fixture.target(64);
    fixture
        .draw(&half, &target, |encoder| {
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, 100, 64))
                .unwrap();
        })
        .unwrap();
    assert_eq!(shaded(&target), whole, "a scissor past the screen");
    let identity = fixture.map(1.0);
    let target = fixture.target(64);
    fixture
        .draw(&identity, &target, |encoder| {
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, 16, 64))
                .unwrap();
        })
        .unwrap();
    assert_eq!(
        shaded(&target),
        (16 * 64, [0, 0, 15, 63], [[0.5, 0.5], [15.5, 63.5]]),
        "a scissor over the identity"
    );
    // Attachments smaller than the physical size take the physical pixels in them (probe 16's
    // `att_32` and `att_small`).
    for side in [32, 16] {
        let target = fixture.target(side);
        fixture.draw(&half, &target, |_| {}).unwrap();
        let last = side - 1;
        let centre = last as f32 + 0.5;
        assert_eq!(
            shaded(&target),
            (
                (side * side) as usize,
                [0, 0, last, last],
                [[0.5, 0.5], [centre, centre]]
            ),
            "attachments of {side}"
        );
    }
    // An array length past the map's one layer rasterizes by it, a draw of slice 0 shading that
    // slice alone (`map1_array_flat`).
    let target = fixture.target_array(33, 2);
    let mut layered = pass(&half, &target);
    layered.render_target_array_length = 2;
    fixture.draw_in(&layered, |_| {}).unwrap();
    assert_eq!(shaded_in(&target, 0), whole, "slice 0 past the map's layer");
    assert_eq!(shaded_in(&target, 1).0, 0, "slice 1 past the map's layer");
    layer_reported_nothing_at_all(&messages);
}

/// The kind [`a_pass_over_a_map_of_no_layers_draws_nothing`] takes: a draw over a map of no
/// layers made after the device released a map of layers.
pub(super) const ZERO_LAYER_AFTER_RELEASE: [Degradation; 1] =
    [Degradation::ZeroLayerMapAfterRelease];

/// Run `test` with a fixture on a device of its own, as each of the probes' cases ran in a process
/// of its own (a device's released maps of layers decide what its later maps of no layers draw),
/// then fail on any validation message.
fn fresh(test: impl FnOnce(&Fixture, &Messages)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let fixture = Fixture::new(&device);
    test(&fixture, &messages);
    layer_reported_nothing_at_all(&messages);
}

/// Whether the device announced [`ZERO_LAYER_AFTER_RELEASE`] since `messages` was last read.
fn announced_after_release(messages: &Messages) -> bool {
    counts(&messages.take_notices()).contains_key(&Degradation::ZeroLayerMapAfterRelease)
}

/// Our probe's eleven cases and its release before the map (an M4 with macOS
/// 27.0.1, each in a process of its own, three times, the same bytes each time), each on a device
/// of its own: a pass over a map of no layers made while no map of layers was released runs its
/// clear and store and its draw writes nothing, at 8 pixels with and without a scissor, at 64, a
/// map of 256 pixels over an attachment of 8, beside a live map of one layer, between passes over
/// one, and for a map made before a map of layers was released; and announces nothing. One made
/// after a map of layers was released draws on Metal (rm0draw1's `after_half64`: physical rows 0
/// to 15 at full width), where the runtime draws nothing and announces
/// [`ZERO_LAYER_AFTER_RELEASE`] once. The probe's controls: without a map the draw shades the
/// scissor's 3 by 5 pixels, and all 64 by 64; a map of 8 pixels at half quality shades those 3 by 5
/// pixels too, and one of 64 its 33 by 33 physical pixels.
#[test]
fn a_pass_over_a_map_of_no_layers_draws_nothing() {
    let corner = |encoder: &mut RenderCommandEncoder| {
        encoder
            .set_scissor_rect(ScissorRect::new(0, 0, 3, 5))
            .unwrap();
    };
    let whole = (33 * 33, [0, 0, 32, 32], [[0.5, 0.5], [32.5, 32.5]]);
    // `none8` and `half8`.
    fresh(|fixture, _| {
        let target = filled(fixture, 8);
        fixture.draw_in(&plain(&target), corner).unwrap();
        assert_eq!(
            shaded(&target),
            (15, [0, 0, 2, 4], [[0.5, 0.5], [2.5, 4.5]]),
            "none8"
        );
    });
    fresh(|fixture, _| {
        let target = filled(fixture, 8);
        fixture
            .draw(&fixture.map_of(8, &[(0.5, 0.5)]), &target, corner)
            .unwrap();
        let (count, bounds, _) = shaded(&target);
        assert_eq!((count, bounds), (15, [0, 0, 2, 4]), "half8");
    });
    // `zero8`, `zero8full`, `zero64` and `zero256on8`.
    for (name, screen, side, scissor) in [
        ("zero8", 8, 8, true),
        ("zero8full", 8, 8, false),
        ("zero64", 64, 64, false),
        ("zero256on8", 256, 8, true),
    ] {
        fresh(|fixture, messages| {
            let target = filled(fixture, side);
            fixture
                .draw(&fixture.map_of(screen, &[]), &target, |encoder| {
                    if scissor {
                        corner(encoder);
                    }
                })
                .unwrap();
            assert!(cleared(&target), "{name}");
            assert!(!announced_after_release(messages), "{name}");
        });
    }
    // `none64` and `half64`.
    fresh(|fixture, _| {
        let target = filled(fixture, 64);
        fixture.draw_in(&plain(&target), |_| {}).unwrap();
        assert_eq!(
            shaded(&target),
            (64 * 64, [0, 0, 63, 63], [[0.5, 0.5], [63.5, 63.5]]),
            "none64"
        );
        let target = filled(fixture, 64);
        fixture
            .draw(&fixture.map_of(64, &[(0.5, 0.5)]), &target, |_| {})
            .unwrap();
        assert_eq!(shaded(&target), whole, "half64");
    });
    // `after_half64`: a pass over a map of one layer, the map released, then one of none, which
    // Metal draws (its rows 0 to 15) and the runtime does not: announced, once.
    fresh(|fixture, messages| {
        {
            let half = fixture.map_of(64, &[(0.5, 0.5)]);
            let target = filled(fixture, 64);
            fixture.draw(&half, &target, |_| {}).unwrap();
            assert_eq!(shaded(&target), whole, "after_half64_first");
        }
        for _ in 0..2 {
            let target = filled(fixture, 64);
            fixture
                .draw(&fixture.map_of(64, &[]), &target, |_| {})
                .unwrap();
            assert!(cleared(&target), "after_half64_zero");
        }
        // Beside it, the fixture's making and buffer, and the half map's pass drawn by regions
        // (`rasterization-rate-map`: its axes are not the identity).
        let mut first = fixture.made.clone();
        first.insert(Degradation::RasterizationRateMap);
        assert_made_once(messages, &ZERO_LAYER_AFTER_RELEASE, &first);
    });
    // `beside_half64`: a map of one layer live beside one of none.
    fresh(|fixture, messages| {
        let half = fixture.map_of(64, &[(0.5, 0.5)]);
        let empty = fixture.map_of(64, &[]);
        let target = filled(fixture, 64);
        fixture.draw(&empty, &target, |_| {}).unwrap();
        assert!(cleared(&target), "beside_half64_zero");
        let target = filled(fixture, 64);
        fixture.draw(&half, &target, |_| {}).unwrap();
        assert_eq!(shaded(&target), whole, "beside_half64_half");
        assert!(!announced_after_release(messages), "beside_half64");
    });
    // `seq`: passes over a map of none, one of one layer, and the first again.
    fresh(|fixture, messages| {
        let empty = fixture.map_of(64, &[]);
        let half = fixture.map_of(64, &[(0.5, 0.5)]);
        for (name, over, zero) in [
            ("seq_zero1", &empty, true),
            ("seq_half", &half, false),
            ("seq_zero2", &empty, true),
        ] {
            let target = filled(fixture, 64);
            fixture.draw(over, &target, |_| {}).unwrap();
            match zero {
                true => assert!(cleared(&target), "{name}"),
                false => assert_eq!(shaded(&target), whole, "{name}"),
            }
        }
        assert!(!announced_after_release(messages), "seq");
    });
    // `zero_before_release`: a map of none made before a map of one layer is
    // released draws nothing on Metal after the release too, and announces nothing here.
    fresh(|fixture, messages| {
        let empty = fixture.map_of(64, &[]);
        {
            let half = fixture.map_of(64, &[(0.5, 0.5)]);
            fixture.draw(&half, &filled(fixture, 64), |_| {}).unwrap();
        }
        let target = filled(fixture, 64);
        fixture.draw(&empty, &target, |_| {}).unwrap();
        assert!(cleared(&target), "zero_before_release_zero2");
        assert!(!announced_after_release(messages), "zero_before_release");
    });
}

/// When a test resets the map from the identity to half quality (our probe `rm`'s cases).
#[derive(Clone, Copy, Debug, PartialEq)]
enum ResetAt {
    /// After the encoder opened.
    Opened,
    /// After its draw.
    Drawn,
    /// After `end_encoding`.
    Ended,
    /// After `commit`, while the command buffer waits for an event the CPU signals after the
    /// reset.
    Held,
    /// After the command buffer completed.
    Completed,
}

/// A map of the screen at `quality` (`Mutable` where `mutable`, reset to that quality).
fn reset_map(fixture: &Fixture, quality: f32, mutable: bool) -> RasterizationRateMap {
    let mut made = descriptor(SCREEN, SCREEN, &[(quality, quality)]);
    if mutable {
        made.mutability = Mutability::Mutable;
    }
    let map = fixture.device.new_rasterization_rate_map(&made).unwrap();
    if mutable {
        map.reset_using_descriptor(&descriptor(SCREEN, SCREEN, &[(quality, quality)]))
            .unwrap();
    }
    map
}

/// A pass over `map` into `target` with one draw, `map` reset to `to` at `at`; the command
/// buffer completed. Before the reset where `before`, run on the command buffer before the pass.
fn reset_in_pass(
    fixture: &Fixture,
    map: &RasterizationRateMap,
    target: &Texture,
    to: &metal2vulkan::bind::RateMapDescriptor,
    at: ResetAt,
) {
    let device = &fixture.device;
    let reset = || map.reset_using_descriptor(to).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let event = device.new_shared_event().unwrap();
    if at == ResetAt::Held {
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
    }
    let mut encoder = command_buffer
        .render_command_encoder(&pass(map, target))
        .unwrap();
    if at == ResetAt::Opened {
        reset();
    }
    encoder.set_render_pipeline_state(&fixture.state).unwrap();
    encoder
        .set_fragment_buffer(Some(&fixture.screen), 0, 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    if at == ResetAt::Drawn {
        reset();
    }
    encoder.end_encoding().unwrap();
    if at == ResetAt::Ended {
        reset();
    }
    command_buffer.commit().unwrap();
    if at == ResetAt::Held {
        reset();
        event.set_signaled_value(1);
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    if at == ResetAt::Completed {
        reset();
    }
}

/// Our probe `rm` (the M3 Max under macOS 27.0.1, each case in a process of its own): Metal's
/// pass reads the map as the GPU runs it, so a reset from the identity to half quality after the
/// encoder opened, after its draw, after `end_encoding`, and after `commit` while the command
/// buffer waits for an event the CPU signals after the reset each shades the half-quality map's
/// 33 by 33 physical pixels; a reset after the command buffer completed leaves the identity's 64
/// by 64. The runtime encodes the pass again for the reset, once, from its log: at the commit for
/// a reset before it, in the reset's call after it (`WorkCounts::encoders_reencoded`). A mutable
/// map and an immutable one alike (Metal resets either), and without a notice.
#[test]
fn a_reset_shows_in_a_pass_not_yet_submitted() {
    let half = (33 * 33, [0, 0, 32, 32], [[0.5, 0.5], [32.5, 32.5]]);
    let identity = (64 * 64, [0, 0, 63, 63], [[0.5, 0.5], [63.5, 63.5]]);
    let to = descriptor(SCREEN, SCREEN, &[(0.5, 0.5)]);
    for at in [
        ResetAt::Opened,
        ResetAt::Drawn,
        ResetAt::Ended,
        ResetAt::Held,
        ResetAt::Completed,
    ] {
        for mutable in [true, false] {
            fresh(|fixture, messages| {
                let map = reset_map(fixture, 1.0, mutable);
                let target = fixture.target(64);
                let before = fixture.device.work_counts().encoders_reencoded;
                reset_in_pass(fixture, &map, &target, &to, at);
                let again = fixture.device.work_counts().encoders_reencoded - before;
                let case = format!("{at:?}, mutable {mutable}");
                match at {
                    ResetAt::Completed => {
                        assert_eq!(shaded(&target), identity, "{case}");
                        assert_eq!(again, 0, "{case}");
                    }
                    _ => {
                        assert_eq!(shaded(&target), half, "{case}");
                        assert_eq!(again, 1, "{case}");
                    }
                }
                let kinds = counts(&messages.take_notices());
                assert!(
                    !kinds.contains_key(&Degradation::RateMapResetUnencoded),
                    "{case}: {kinds:?}"
                );
            });
        }
    }
}

/// Our probe `rm`'s `held_two`: two passes of one command buffer waiting for an event, the first
/// ended before a reset to half quality and the second opened after it, then a reset to a quarter
/// before the CPU signals the event: both shade the quarter's 32 by 32 physical pixels, as Metal's
/// GPU reads the map as it runs them.
#[test]
fn a_reset_shows_in_every_pass_not_yet_submitted() {
    fresh(|fixture, _| {
        let device = &fixture.device;
        let map = reset_map(fixture, 1.0, true);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let event = device.new_shared_event().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let targets = [fixture.target(64), fixture.target(64)];
        for (index, target) in targets.iter().enumerate() {
            if index == 1 {
                map.reset_using_descriptor(&descriptor(SCREEN, SCREEN, &[(0.5, 0.5)]))
                    .unwrap();
            }
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&map, target))
                .unwrap();
            encoder.set_render_pipeline_state(&fixture.state).unwrap();
            encoder
                .set_fragment_buffer(Some(&fixture.screen), 0, 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        map.reset_using_descriptor(&descriptor(SCREEN, SCREEN, &[(0.25, 0.25)]))
            .unwrap();
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let quarter = (32 * 32, [0, 0, 31, 31], [[0.5, 0.5], [31.5, 31.5]]);
        for (index, target) in targets.iter().enumerate() {
            assert_eq!(shaded(target), quarter, "pass {index}");
        }
    });
}

/// Our probe `rs`'s resets to another screen (the M3 Max under macOS 27.0.1, each case in a
/// process of its own, repeated alike): a pass over the identity of a mutable map of `taken`
/// pixels a side, reset to the identity of `now` pixels after its encoder opened, after
/// `end_encoding` or after `commit` while the command buffer waits, keeps the screen it took and
/// shades its physical pixels to the 32-pixel tiles of `now` where `now` spans one tile fewer than
/// `taken`, else to `taken` (`(taken, now, target, shaded)`); reset before the encoder opened, the
/// pass is the new map's, shading `now`. Each pixel at its centre, and without a notice.
#[test]
fn a_reset_to_another_screen_keeps_the_screen_the_pass_took() {
    const AFTER: [(u32, u32, u64, u64); 10] = [
        (16, 8, 64, 16),
        (8, 16, 64, 8),
        (64, 32, 64, 32),
        (48, 24, 64, 32),
        (40, 8, 64, 32),
        (100, 50, 128, 100),
        (128, 80, 128, 96),
        (128, 40, 128, 128),
        (200, 161, 256, 192),
        (256, 200, 256, 224),
    ];
    const BEFORE: [(u32, u32, u64, u64); 3] = [(16, 8, 64, 8), (8, 16, 16, 16), (100, 50, 128, 50)];
    let shades = |side: u64| {
        let edge = side as f32 - 0.5;
        (
            (side * side) as usize,
            [0, 0, side - 1, side - 1],
            [[0.5, 0.5], [edge, edge]],
        )
    };
    fresh(|fixture, messages| {
        let device = &fixture.device;
        let identity = |side: u32| descriptor(side, side, &[(1.0, 1.0)]);
        let map_of = |side: u32| {
            let mut made = identity(side);
            made.mutability = Mutability::Mutable;
            let map = device.new_rasterization_rate_map(&made).unwrap();
            map.reset_using_descriptor(&identity(side)).unwrap();
            map
        };
        for (taken, now, side, shaded_to) in AFTER {
            for at in [ResetAt::Opened, ResetAt::Ended, ResetAt::Held] {
                let map = map_of(taken);
                let target = fixture.target(side);
                reset_in_pass(fixture, &map, &target, &identity(now), at);
                let case = format!("{taken} reset to {now} at {at:?}, target {side}");
                assert_eq!(shaded(&target), shades(shaded_to), "{case}");
            }
        }
        for (taken, now, side, shaded_to) in BEFORE {
            let map = map_of(taken);
            map.reset_using_descriptor(&identity(now)).unwrap();
            let target = fixture.target(side);
            fixture.draw(&map, &target, |_| {}).unwrap();
            let case = format!("{taken} reset to {now} before the pass, target {side}");
            assert_eq!(shaded(&target), shades(shaded_to), "{case}");
        }
        // Beyond the fixture's making and buffer, nothing: every map is the identity.
        let kinds = counts(&messages.take_notices());
        assert!(
            kinds.keys().all(|kind| fixture.made.contains(kind)),
            "{kinds:?}"
        );
    });
}

/// A command buffer of `queue` with a pass over `map` into a target of its own for each of
/// `draws` (a draw of a triangle where true, no draw call where false), committed and completed.
fn passes(
    fixture: &Fixture,
    queue: &CommandQueue,
    map: &RasterizationRateMap,
    draws: &[bool],
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    for &drawn in draws {
        let target = fixture.target(64);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(map, &target))
            .unwrap();
        if drawn {
            encoder.set_render_pipeline_state(&fixture.state).unwrap();
            encoder
                .set_fragment_buffer(Some(&fixture.screen), 0, 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    command_buffer
}

/// Our probe `rs`'s drawless cases (the M3 Max under macOS 27.0.1, each in a process of its own):
/// Metal's GPU hangs at a pass of no draw call over a `Mutable` map, of layers or of none, failing
/// its command buffer with `GpuHang`, until a command buffer with a draw call over a mutable map
/// was committed on the device: a drawn pass before it in its own command buffer does not keep
/// it from hanging, nor does a drawn pass over an immutable map; a pass over an immutable map
/// runs. The runtime answers so, each hang on a queue of its own (a queue's second revokes it).
#[test]
fn a_pass_of_no_draw_call_over_a_mutable_map_hangs_until_one_drew() {
    fresh(|fixture, _| {
        let device = &fixture.device;
        let hangs = |command_buffer: &CommandBuffer, case: &str| {
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Error,
                "{case}"
            );
            assert_eq!(
                command_buffer.error(),
                Some(CommandBufferError::GpuHang),
                "{case}"
            );
        };
        let runs = |command_buffer: &CommandBuffer, case: &str| {
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "{case}"
            );
        };
        let queue = device.new_command_queue().unwrap();
        let immutable = reset_map(fixture, 0.5, false);
        runs(
            &passes(fixture, &queue, &immutable, &[true]),
            "immutable, drawn",
        );
        runs(&passes(fixture, &queue, &immutable, &[false]), "immutable");
        let mutable = reset_map(fixture, 0.5, true);
        hangs(
            &passes(
                fixture,
                &device.new_command_queue().unwrap(),
                &mutable,
                &[false],
            ),
            "mutable, before any drew",
        );
        let mut none = descriptor(SCREEN, SCREEN, &[]);
        none.mutability = Mutability::Mutable;
        let made_none = device.new_rasterization_rate_map(&none).unwrap();
        hangs(
            &passes(
                fixture,
                &device.new_command_queue().unwrap(),
                &made_none,
                &[false],
            ),
            "mutable of no layers, before any drew",
        );
        hangs(
            &passes(
                fixture,
                &device.new_command_queue().unwrap(),
                &mutable,
                &[true, false],
            ),
            "mutable, after a drawn pass of its own command buffer",
        );
        // That command buffer was committed: now none hangs.
        let queue = device.new_command_queue().unwrap();
        runs(
            &passes(fixture, &queue, &mutable, &[false]),
            "mutable, after one drew",
        );
        mutable
            .reset_using_descriptor(&descriptor(SCREEN, SCREEN, &[]))
            .unwrap();
        runs(
            &passes(fixture, &queue, &mutable, &[false]),
            "reset to no layers, after one drew",
        );
        runs(
            &passes(fixture, &queue, &made_none, &[false]),
            "of no layers, after one drew",
        );
    });
}

/// Our probe `rm`'s `unset_pass`, `none_pass` and `none_mut_pass` (the M3 Max under macOS
/// 27.0.1): a pass over a mutable map not yet reset, and over a map of no layers made while no map
/// of layers was released, mutable or not, runs its clear and store and each draw's vertex
/// function (Metal's counted 3 runs for a triangle), and shades nothing. The runtime records the
/// draw through an empty scissor: it records `vkCmdDraw` once (`commands_recorded`).
#[test]
fn a_pass_over_a_map_of_no_compiled_layers_runs_its_vertex_function_alone() {
    let unset = |screen: u32, layers: &[(f32, f32)]| {
        let mut made = descriptor(screen, screen, layers);
        made.mutability = Mutability::Mutable;
        made
    };
    for (name, made) in [
        ("unset", unset(SCREEN, &[(0.5, 0.5)])),
        ("none", descriptor(SCREEN, SCREEN, &[])),
        ("none_mutable", unset(SCREEN, &[])),
    ] {
        fresh(|fixture, messages| {
            let device = &fixture.device;
            let map = device.new_rasterization_rate_map(&made).unwrap();
            let target = filled(fixture, 64);
            let before = device.work_counts();
            fixture.draw(&map, &target, |_| {}).unwrap();
            let after = device.work_counts();
            assert!(cleared(&target), "{name}");
            let draws = after.commands_recorded.cmd_draw - before.commands_recorded.cmd_draw;
            assert_eq!(draws, 1, "{name}: the vertex function's draw");
            assert!(!announced_after_release(messages), "{name}");
        });
    }
}

/// The kind [`a_pass_the_runtime_cannot_encode_again_draws_by_the_map_it_took`] takes.
pub(super) const RESET_UNENCODED: [Degradation; 1] = [Degradation::RateMapResetUnencoded];

/// A pass whose encoding again is refused keeps its first recording: its draw's scissor, which
/// cuts into the viewport, the runtime plans over the identity map the pass took, and refuses over
/// a map that is not the identity (`RegionRefusal::Scissor`), so a reset of its map to half quality
/// before the commit cannot encode it again: it draws by the map as it took it, the scissor's 16
/// by 64 pixels of the identity, where Metal's GPU draws by the map as reset, and the commit
/// announces [`RESET_UNENCODED`], once for two such command buffers.
#[test]
fn a_pass_the_runtime_cannot_encode_again_draws_by_the_map_it_took() {
    fresh(|fixture, messages| {
        let device = &fixture.device;
        let identity = (16 * 64, [0, 0, 15, 63], [[0.5, 0.5], [15.5, 63.5]]);
        for _ in 0..2 {
            let map = reset_map(fixture, 1.0, true);
            let target = fixture.target(64);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&map, &target))
                .unwrap();
            encoder.set_render_pipeline_state(&fixture.state).unwrap();
            encoder
                .set_fragment_buffer(Some(&fixture.screen), 0, 0)
                .unwrap();
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, 16, 64))
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            map.reset_using_descriptor(&descriptor(SCREEN, SCREEN, &[(0.5, 0.5)]))
                .unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(shaded(&target), identity);
        }
        // Beside it, the fixture's making and buffer, and the reset's map drawn by regions.
        let mut made: BTreeSet<Degradation> = fixture.made.clone();
        made.insert(Degradation::RasterizationRateMap);
        assert_made_once(messages, &RESET_UNENCODED, &made);
    });
}

/// Probes 16, 16b and 16c's refusals, and the region path's (the module docs).
#[test]
fn rate_mapped_passes_are_refused_where_metal_or_the_runtime_refuses_them() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let fixture = Fixture::new(&device);
    let half = fixture.map(0.5);
    let queue = device.new_command_queue().unwrap();
    let refused = |pass: &RenderPassDescriptor| {
        let command_buffer = queue.command_buffer().unwrap();
        match command_buffer.render_command_encoder(pass) {
            Err(Error::Invalid(Invalid {
                problem: Problem::RenderPass(problem),
                ..
            })) => problem,
            other => panic!("a refused pass made {other:?}"),
        }
    };
    // Past the layers of a map whose layers differ.
    let differing = device
        .new_rasterization_rate_map(&descriptor(SCREEN, SCREEN, &[(0.5, 0.5), (1.0, 1.0)]))
        .unwrap();
    let mut layered = pass(&differing, &fixture.target_array(33, 3));
    layered.render_target_array_length = 3;
    assert_eq!(
        refused(&layered),
        RenderPassProblem::RateMapArrayLength {
            length: 3,
            layers: 2,
        }
    );
    let mut targetless = RenderPassDescriptor::new();
    targetless.rasterization_rate_map = Some(half.clone());
    targetless.render_target_width = 128;
    targetless.render_target_height = 128;
    targetless.default_raster_sample_count = 1;
    assert_eq!(
        refused(&targetless),
        RenderPassProblem::RateMapTargetSize {
            target: [128, 128],
            screen: [SCREEN; 2],
        }
    );
    targetless.render_target_width = 33;
    targetless.render_target_height = 33;
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&targetless)
        .unwrap()
        .end_encoding()
        .unwrap();

    // A scissor that cuts into the viewport over a map that is not the identity.
    let target = fixture.target(33);
    let refusal = fixture.draw(&half, &target, |encoder| {
        encoder
            .set_scissor_rect(ScissorRect::new(0, 0, 16, 64))
            .unwrap();
    });
    assert_eq!(
        refusal,
        Err(Error::NotImplemented(RuntimeGap::RateMapRegions(
            RegionRefusal::Scissor
        )))
    );
    layer_reported_nothing_at_all(&messages);
}
