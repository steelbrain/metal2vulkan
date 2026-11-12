//! Device tests of indexed draws whose indices run past their index buffer on the device the cfg
//! selects, under the validation layer.
//!
//! Metal's validation stops such a draw. Metal's runtime draws each position past the buffer's
//! length as index 0 plus the base vertex, bounded by the buffer even inside a heap (the probe on
//! an M3 Max with macOS 26.7: index buffers placed in a heap whose every word a read past them
//! would show; under a base vertex of 100 each such position drew vertex 100). A partial last index
//! reads the bytes past the length up to the length rounded up to 4 (the probe-ix on an M4 with
//! macOS 27.0.1), the heap's next bytes for a placed buffer.
//!
//! Each draw here is of points with rasterization disabled, by the owned fixture
//! `ob3_index_buffer`, whose vertex function marks `seen[min(vertex_id, 7)]`: four positions over
//! the 32-bit indices 1 and 2, under a base vertex of 4. Metal's answer marks 5 and 6 (the
//! indices) and 4 (index 0 plus the base vertex, the two positions past the buffer). A fetch that
//! read on past the buffer would mark 7 (the heap's padding and guard words, `0x5A5A5A5A` and
//! `0xE7E7E7E7`), and a draw that left the positions out would leave 4 unmarked.
//!
//! Each draw runs from a buffer placed in a heap between guards (`runtime_case::guarded_buffer`)
//! and from a buffer of its own, directly and indirectly (`MTLDrawIndexedPrimitivesIndirectArguments`
//! of the same counts), on the device as opened, with `maintenance5` withheld, and with the
//! robustness features withheld, so every venue runs each form of the bound. Vulkan states no
//! index for a fetch past the bound range, so the direct draw reads a copy of its indices padded
//! with zeros on every device (`isolated-index-bytes`, announced once). The indirect one stays one
//! draw on the device's range only where the device enables `robustBufferAccess2` and its driver
//! claims to read zero past the range (`DriverFact::RobustIndexFetchReadsZero`, measured by this
//! lane: RADV and ANV from Mesa 26.2.4, NVIDIA's from 615.71.09): it binds the range of the
//! buffer's whole indices (`vkCmdBindIndexBuffer2`), or reads the copy of them where the device
//! binds no such range (`isolated-index-bytes` again), and its Metal vertices are that claim's
//! check at every gate. Every other device (MoltenVK, lavapipe, a device that withholds the
//! robustness features) draws Metal's vertices through the index bounds pass
//! (`indirect-index-bounds-pass`, announced once).
//!
//! Each device also draws from offsets at and past the buffer's length, where every position lies
//! past the buffer and Metal reads the memory past it (probe-ix: whatever memory holds, so no
//! exact answer exists). The direct draw reads a padded copy of no index byte, so it draws index
//! 0 plus the base vertex at each position, and no copy of indices is announced
//! (`isolated-index-bytes` names the copy where it holds bytes). The indirect draw, whose
//! arguments the binder asks for on the CPU there, goes through the index bounds pass on every
//! device, which draws the same. Both announce `zeroed-index-fetch`, once, and a device whose
//! indirect draws stayed on its range announces the pass there first.
//!
//! And each draws a partial last index: 6 bytes, the index 1 and half of the index 2. Its
//! position 1 reads the half and the 2 bytes past the length: the zeroed padding of a buffer of
//! its own, which draws Metal's vertices as above, and the heap's padding words after a placed
//! buffer, which mark 7 as Metal's read of them does. The padded copy and the index bounds pass
//! read the same bytes.

use super::apart_writes::{layer_reported_nothing_but_apart_writes, NotedWrites};
use super::degrade_notices::{assert_made_once, assert_once};
use super::shared_kinds;
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::guards::GuardReach;
use crate::candidate::runtime_case::guarded_buffer;
use crate::coverage::ROBUSTNESS_FEATURES;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{IndexType, PrimitiveType};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandQueue, Degradation, Device, DeviceOptions, DriverFact, Error, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState,
};

/// The kinds the draws here take (`degrade_notices`' closure).
pub(super) const INDEX_FETCHES: [Degradation; 3] = [
    Degradation::IsolatedIndexBytes,
    Degradation::IndirectIndexBoundsPass,
    Degradation::ZeroedIndexFetch,
];

/// The index buffer's 32-bit indices.
const INDICES: [u32; 2] = [1, 2];

/// The positions each draw draws: two inside the buffer, two past it.
const POSITIONS: u32 = 4;

/// The base vertex, which Metal adds to index 0 at a position past the buffer.
const BASE_VERTEX: i32 = 4;

/// `seen` after Metal's draw: the indices plus the base vertex (5, 6), and index 0 plus the base
/// vertex (4).
const METAL: [u32; 8] = [0, 0, 0, 0, 1, 1, 1, 0];

/// `seen` after a draw from an offset at or past the length: index 0 plus the base vertex at each
/// position.
const ZEROS: [u32; 8] = [0, 0, 0, 0, 1, 0, 0, 0];

/// `seen` after Metal's draw of the partial index buffer placed in a heap: index 1 (5), the half
/// of index 2 with the heap's padding bytes (7), and index 0 (4).
const PARTIAL_PLACED: [u32; 8] = [0, 0, 0, 0, 1, 1, 0, 1];

/// How the device is opened.
#[derive(Clone, Copy, Debug)]
enum Opened {
    /// As the lane opens it.
    AsIs,
    /// Without `maintenance5`: no index range of a size.
    WithoutMaintenance5,
    /// Without the robustness features: no `robustBufferAccess2`.
    WithoutRobustness,
}

impl Opened {
    fn adjust(self, options: &mut DeviceOptions) {
        match self {
            Opened::AsIs => {}
            Opened::WithoutMaintenance5 => {
                options.withheld_features.insert(VkFeature::Maintenance5);
            }
            Opened::WithoutRobustness => {
                options.withheld_features = ROBUSTNESS_FEATURES.into_iter().collect();
            }
        }
    }
}

/// The device's queue, the fixture's vertex-only pipeline state, and the list the device reports
/// to.
struct Lane {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    messages: Messages,
}

impl Lane {
    fn new(device: Device, messages: &Messages) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("ob3_index_buffer.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = library.new_function_with_name("ob3_index_buffer").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.rasterization_enabled = false;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            device,
            pipeline,
            messages: messages.clone(),
        }
    }

    /// `seen` after the draw from `indices` at `offset`, direct or through arguments of the same
    /// counts; or the draw's refusal.
    fn drawn(&self, indices: &Buffer, offset: u64, indirect: bool) -> Result<[u32; 8], Error> {
        self.drawn_positions(indices, offset, indirect, POSITIONS)
    }

    /// [`drawn`](Self::drawn) of `positions` positions. An indirect draw, which the index bounds
    /// pass bounds on a device that does not keep it on its range, writes `seen` alone in its
    /// command buffer, so the layer's report of the pass's two draws passes
    /// ([`layer_reported_nothing_but_apart_writes`]).
    fn drawn_positions(
        &self,
        indices: &Buffer,
        offset: u64,
        indirect: bool,
        positions: u32,
    ) -> Result<[u32; 8], Error> {
        let seen = buffer_with(&self.device, &[0; 32]).unwrap();
        let mut writes = NotedWrites::new(&self.messages);
        if indirect {
            writes.drew_bounded(&seen, 0..32);
        }
        let arguments: Vec<u8> = [positions, 1, 0, BASE_VERTEX as u32, 0]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let arguments = buffer_with(&self.device, &arguments).unwrap();
        // The oracle's pass without attachments: one pixel.
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&seen), 0, 0).unwrap();
        let drawn = if indirect {
            encoder.draw_indexed_primitives_indirect(
                PrimitiveType::Point,
                IndexType::UInt32,
                indices,
                offset,
                &arguments,
                0,
            )
        } else {
            encoder.draw_indexed_primitives(
                PrimitiveType::Point,
                positions,
                IndexType::UInt32,
                indices,
                offset,
                1,
                BASE_VERTEX,
                0,
            )
        };
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let claim = format!("a draw from offset {offset}, indirect {indirect}");
        layer_reported_nothing_but_apart_writes(&self.messages, &claim, &writes);
        drawn?;
        let bytes = read_buffer(&seen, 0, 32).unwrap();
        let mut words = [0u32; 8];
        for (word, chunk) in words.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
            *word = u32::from_le_bytes(*chunk);
        }
        Ok(words)
    }
}

/// Each draw, from a buffer placed in a heap and from one of its own, direct and indirect, on the
/// device as opened, without `maintenance5`, and without robustness, draws Metal's vertices, as
/// the module docs say. On a driver that claims to read zero past the bound index range
/// (`DriverFact::RobustIndexFetchReadsZero`) under `robustBufferAccess2`, its indirect draws stay
/// one draw on the device's range, and their vertices assert that zero, so a release that reads
/// another value there fails here; every other device's take the index bounds pass. A partial
/// last index reads to the length rounded up to 4. From offsets at and past the length every
/// device's draws draw index 0 at each position, the direct one announcing no copy. Each device
/// announces the kinds of [`INDEX_FETCHES`] its draws take once, and not the other.
#[test]
fn index_positions_past_the_buffer_draw_index_zero_plus_the_base_vertex() {
    let bytes: Vec<u8> = INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();
    for opened in [
        Opened::AsIs,
        Opened::WithoutMaintenance5,
        Opened::WithoutRobustness,
    ] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                opened.adjust(options)
            })
            .unwrap();
            let enabled = &device.enabled().features;
            let robust = enabled.contains(&VkFeature::RobustBufferAccess2);
            let sized = enabled.contains(&VkFeature::Maintenance5);
            match opened {
                Opened::AsIs => {}
                Opened::WithoutMaintenance5 => {
                    assert!(!sized, "maintenance5 withheld, yet enabled")
                }
                Opened::WithoutRobustness => assert!(!robust, "robustness withheld, yet enabled"),
            }
            // The indirect draws stay one draw on the device's range.
            let on_range = robust
                && device
                    .properties()
                    .driver_facts
                    .contains(&DriverFact::RobustIndexFetchReadsZero);
            let name = device.name().to_owned();
            let shared = shared_kinds(&device);
            let lane = Lane::new(device, &messages);
            let placed = guarded_buffer(
                &lane.device,
                &bytes,
                GuardReach {
                    before: 0,
                    after: 64,
                },
            )
            .unwrap();
            let own = buffer_with(&lane.device, &bytes).unwrap();
            for (buffer, held) in [(&placed, "placed in a heap"), (&own, "of its own")] {
                for indirect in [false, true] {
                    let at = format!(
                        "{opened:?} on {name} (robust {robust}, sized {sized}, on its range \
                         {on_range}): a {} draw from a buffer {held}",
                        if indirect { "indirect" } else { "direct" }
                    );
                    assert_eq!(lane.drawn(buffer, 0, indirect), Ok(METAL), "{at}");
                }
            }
            // A partial last index: the index 1, then half of the index 2.
            let partial = &bytes[..6];
            let placed_partial = guarded_buffer(
                &lane.device,
                partial,
                GuardReach {
                    before: 0,
                    after: 64,
                },
            )
            .unwrap();
            let own_partial = buffer_with(&lane.device, partial).unwrap();
            for (buffer, held, metal) in [
                (&placed_partial, "placed in a heap", PARTIAL_PLACED),
                (&own_partial, "of its own", METAL),
            ] {
                for indirect in [false, true] {
                    let at = format!(
                        "{opened:?} on {name} (robust {robust}, sized {sized}, on its range \
                         {on_range}): a {} draw of a partial index from a buffer {held}",
                        if indirect { "indirect" } else { "direct" }
                    );
                    assert_eq!(lane.drawn(buffer, 0, indirect), Ok(metal), "{at}");
                }
            }
            // Every device's direct draws pad a copy of their indices; the indirect ones take
            // the index bounds pass unless they stay on the device's range.
            let mut kinds = vec![Degradation::IsolatedIndexBytes];
            if !on_range {
                kinds.push(Degradation::IndirectIndexBoundsPass);
            }
            // Beside them, what the lane's `Shared` buffers force.
            let announced = assert_made_once(&messages, &kinds, &shared);
            for kind in INDEX_FETCHES {
                assert_eq!(
                    announced.contains_key(&kind),
                    kinds.contains(&kind),
                    "{opened:?} on {name}: {kind:?} ({announced:?})"
                );
            }
            let length = bytes.len() as u64;
            for (buffer, held) in [(&placed, "placed in a heap"), (&own, "of its own")] {
                for offset in [length, length + 8] {
                    let at = format!(
                        "{opened:?} on {name}: a draw from offset {offset} of a buffer {held}"
                    );
                    assert_eq!(lane.drawn(buffer, offset, false), Ok(ZEROS), "{at}");
                    assert_eq!(
                        lane.drawn(buffer, offset, true),
                        Ok(ZEROS),
                        "{at}, indirect"
                    );
                }
            }
            // Past the length the direct draws copy no index, and the indirect ones take the
            // index bounds pass on every device.
            let mut past_kinds = vec![Degradation::ZeroedIndexFetch];
            if on_range {
                past_kinds.push(Degradation::IndirectIndexBoundsPass);
            }
            let past = assert_once(&messages.take_notices(), &past_kinds);
            for kind in INDEX_FETCHES {
                assert_eq!(
                    past.contains_key(&kind),
                    past_kinds.contains(&kind),
                    "{opened:?} on {name}: {kind:?} past the length ({past:?})"
                );
            }
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// One encoder's indexed indirect draws from one index buffer and one arguments buffer, on a
/// device opened without robustness, so that every venue bounds them: each draw's index offset,
/// positions, and `seen` after Metal's draw. Inside the buffer; two positions past it; from the
/// second index, then past it; of no position; from offsets at and past the length (index 0 plus
/// the base vertex at each position); and all four again.
const RUN: [(u64, u32, [u32; 8]); 7] = [
    (0, 2, [0, 0, 0, 0, 0, 1, 1, 0]),
    (0, POSITIONS, METAL),
    (4, 3, [0, 0, 0, 0, 1, 0, 1, 0]),
    (0, 0, [0; 8]),
    (8, POSITIONS, ZEROS),
    (16, 1, ZEROS),
    (0, POSITIONS, METAL),
];

/// The draws of [`RUN`] in one encoder each draw Metal's vertices into a `seen` of its own, and
/// each joins the one run of the index bounds pass their buffers share
/// (`WorkCounts::bounded_run_draws`): one plan and one fill bound them all, each draw's two
/// draws and reservation its own pass's.
#[test]
fn a_run_of_bounded_indirect_draws_draws_each_draws_metal_vertices() {
    let bytes: Vec<u8> = INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            Opened::WithoutRobustness.adjust(options)
        })
        .unwrap();
        let name = device.name().to_owned();
        let shared = shared_kinds(&device);
        let lane = Lane::new(device, &messages);
        let indices = buffer_with(&lane.device, &bytes).unwrap();
        let arguments: Vec<u8> = RUN
            .iter()
            .flat_map(|&(_, positions, _)| [positions, 1, 0, BASE_VERTEX as u32, 0])
            .flat_map(u32::to_le_bytes)
            .collect();
        let arguments = buffer_with(&lane.device, &arguments).unwrap();
        let seen: Vec<Buffer> = RUN
            .iter()
            .map(|_| buffer_with(&lane.device, &[0; 32]).unwrap())
            .collect();
        let before = lane.device.work_counts().bounded_run_draws;
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&lane.pipeline).unwrap();
        // Each draw writes a `seen` of its own, alone in the command buffer.
        let mut writes = NotedWrites::new(&messages);
        for (at, ((offset, _, _), seen)) in RUN.iter().zip(&seen).enumerate() {
            writes.drew_bounded(seen, 0..32);
            encoder.set_vertex_buffer(Some(seen), 0, 0).unwrap();
            encoder
                .draw_indexed_primitives_indirect(
                    PrimitiveType::Point,
                    IndexType::UInt32,
                    &indices,
                    *offset,
                    &arguments,
                    at as u64 * 20,
                )
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "on {name}");
        layer_reported_nothing_but_apart_writes(&messages, "the run's draws", &writes);
        assert_eq!(
            lane.device.work_counts().bounded_run_draws - before,
            RUN.len() as u64,
            "on {name}: the draws joined a run"
        );
        for (at, ((offset, positions, metal), seen)) in RUN.iter().zip(&seen).enumerate() {
            let bytes = read_buffer(seen, 0, 32).unwrap();
            let words: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| u32::from_le_bytes(*chunk))
                .collect();
            assert_eq!(
                words, metal,
                "on {name}: draw {at}, {positions} positions from offset {offset}"
            );
        }
        assert_made_once(
            &messages,
            &[
                Degradation::IndirectIndexBoundsPass,
                Degradation::ZeroedIndexFetch,
            ],
            &shared,
        );
        println!(
            "CBF-MARK {name}: {} indexed indirect draws joined one run of the index bounds pass \
             and drew Metal's vertices",
            RUN.len()
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// The positions of [`an_indirect_draw_past_the_old_index_pool_draws_metals_vertices`]'s draw:
/// 300,000 32-bit positions, 1.2 MB of indices in the command buffer's index pool, past the 1 MiB
/// every device's pool once held.
const PAST_OLD_POOL: u32 = 300_000;

/// The bytes of the index pool every device once had.
const OLD_POOL_BYTES: u64 = 1 << 20;

/// An indexed indirect draw whose pooled indices pass the index pool's old size draws Metal's
/// vertices (the two indices, then index 0 plus the base vertex), and its command buffer
/// completes, on a device opened without robustness, so that every venue takes the index bounds
/// pass: its announcement shows the path ran. Previously the command buffer failed
/// `IndexPoolOverflow`.
#[test]
fn an_indirect_draw_past_the_old_index_pool_draws_metals_vertices() {
    let bytes: Vec<u8> = INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();
    assert!(u64::from(PAST_OLD_POOL) * 4 > OLD_POOL_BYTES);
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            Opened::WithoutRobustness.adjust(options)
        })
        .unwrap();
        let name = device.name().to_owned();
        let unified = device.properties().has_unified_memory;
        let lane = Lane::new(device, &messages);
        let own = buffer_with(&lane.device, &bytes).unwrap();
        assert_eq!(
            lane.drawn_positions(&own, 0, true, PAST_OLD_POOL),
            Ok(METAL),
            "on {name}"
        );
        // On a device without unified memory, the lane's `Shared` buffers lie in system memory,
        // announced as the first is made (`blits::SYSTEM_MEMORY`).
        let expected: &[Degradation] = match unified {
            true => &[Degradation::IndirectIndexBoundsPass],
            false => &[
                Degradation::IndirectIndexBoundsPass,
                Degradation::SharedInSystemMemory,
            ],
        };
        assert_once(&messages.take_notices(), expected);
        println!(
            "CBF-MARK {name}: an indexed indirect draw of {PAST_OLD_POOL} positions ({} bytes of \
             pooled indices, past the old {OLD_POOL_BYTES}) drew Metal's vertices through the \
             index bounds pass",
            u64::from(PAST_OLD_POOL) * 4
        );
    }
    layer_reported_nothing_at_all(&messages);
}
