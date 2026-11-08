//! The uses of what a render encoder made resident that its draws' binding plans list, kept from
//! one draw to the next (`binding::command_uses_into`): a draw whose plan is the last draw's,
//! under a residency of the last draw's generation, works none of them out
//! (`WorkCounts::resident_uses_made`), nor walks them as it gathers them for its render pass
//! instance (`WorkCounts::resident_uses_gathered`, `render/shadows.rs`) or checks them against
//! what its pass's fragment stages wrote and read (`WorkCounts::resident_uses_against_fragments`,
//! `render/pass_start_reads.rs`) or its attachments
//! (`WorkCounts::resident_uses_against_attachments`, `EncoderCore::degrade_attachment_reads`), or
//! notes them for hazard tracking (`WorkCounts::resident_uses_noted`, `noted_apart.rs`), nor
//! works out again the keys it reaches through what the encoder made resident alone but those
//! its bound buffers name and the last draw's named (`WorkCounts::resident_keys_settled`), and
//! builds with debug assertions check each draw's kept uses against those it would make whole
//! (`binding::kept_resident_checks`), and its gathering, checks and noting against those of its
//! whole uses. The draws read through a root table and a handle table, so no draw of theirs is kept to
//! draw again or rebind (`render/draw_again.rs`): each takes every step.
//! A draw after the encoder made another resource resident, a draw of its pipeline state under
//! another plan, and the first draw of an encoder of the same residency's shape over other
//! resources (the same plan, a residency of another generation) work them out again. A draw after
//! draws of another pipeline state keeps those its own state's last draw worked out, which its
//! encoder set aside with that state's bind state (`live_bind.rs`), and notes them again: its
//! encoder keeps only its last draw's noting apart.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`) with debug assertions, on the kind of device the cfg selects. The functions are
//! the owned fixtures `validation/fixtures/public/engine/resident_draws.ll` and
//! `rt6_attachmentless.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{device, Heard};
use super::encoder_holds::{pass, render_state};
use crate::binding::kept_resident_checks;
use crate::render::{kept_heap_checks, plan_facts_made};
use crate::{
    Buffer, CommandBufferStatus, Degradation, Device, Region, RenderCommandEncoder,
    RenderPipelineState, ResourceOptions, ResourceRef, SamplerState, SamplerStateOptions, Texture,
    TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage};

/// The fixture.
const FIXTURE: &str = "engine/resident_draws.ll";

/// The fixture of the marker, whose fragments write word 0 of their buffer 0.
const MARKER: &str = "engine/rt6_attachmentless.ll";

/// The target's width and height.
const SIDE: u64 = 64;

/// The positions an encoder makes resident, and the draws it draws of each shape.
const POSITIONS: u64 = 6;

/// The bytes of an argument-table entry: a position's address, a texture's resource ID and a
/// sampler's.
const ENTRY: u64 = 24;

/// What the draws read: the functions' states, the texture and sampler the entries name, and the
/// target.
struct Scene {
    device: Device,
    entry: RenderPipelineState,
    slot: RenderPipelineState,
    sampled: Texture,
    sampler: SamplerState,
    target: Texture,
}

impl Scene {
    fn new(device: &Device) -> Scene {
        let mut sampled = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        sampled.usage = TextureUsage::SHADER_READ;
        sampled.resource_options = ResourceOptions::shared();
        let sampled = device.new_texture(&sampled).unwrap();
        let texels: Vec<u8> = [40u8, 160, 220, 255].repeat(16);
        (sampled.replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &texels, 16, 0)).unwrap();
        let options = SamplerStateOptions {
            support_argument_buffers: true,
            ..SamplerStateOptions::default()
        };
        let sampler = device
            .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
            .unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        Scene {
            entry: render_state(device, FIXTURE, "rd_entry_vertex", "rd_entry_fragment"),
            slot: render_state(device, FIXTURE, "rd_slot_vertex", "rd_color_fragment"),
            sampled,
            sampler,
            target: device.new_texture(&target).unwrap(),
            device: device.clone(),
        }
    }

    /// A buffer of `bytes`.
    fn buffer(&self, bytes: &[u8]) -> Buffer {
        (self.device)
            .new_buffer_with_bytes(bytes, ResourceOptions::shared())
            .unwrap()
    }

    /// The target's pixels that the draws covered.
    fn covered(&self) -> usize {
        let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        (self.target.get_bytes(&mut bytes, 4 * SIDE, 0, region, 0, 0)).unwrap();
        bytes.chunks(4).filter(|pixel| pixel[3] != 0).count()
    }
}

/// One encoder's resources: [`POSITIONS`] positions, a buffer each, two argument tables of an
/// entry for each after one unused (a binding at offset 0 takes a binding form of its own, so a
/// plan of its own, where the draws' later offsets share one), and a buffer made resident later.
struct Resources {
    positions: Vec<Buffer>,
    tables: [Buffer; 2],
    later: Buffer,
}

impl Resources {
    fn new(scene: &Scene, corner: f32) -> Resources {
        let positions: Vec<Buffer> = (0..POSITIONS)
            .map(|at| {
                let position = [corner + at as f32 / 8.0, corner, 0.0, 1.0];
                scene.buffer(&position.map(f32::to_le_bytes).concat())
            })
            .collect();
        let entries: Vec<u8> = (positions[..1].iter().chain(&positions))
            .flat_map(|position| {
                let ids = (
                    scene.sampled.gpu_resource_id().0,
                    scene.sampler.gpu_resource_id().0,
                );
                [position.gpu_address().0, ids.0, ids.1]
            })
            .flat_map(u64::to_le_bytes)
            .collect();
        Resources {
            tables: [scene.buffer(&entries), scene.buffer(&entries)],
            later: scene.buffer(&[0; 16]),
            positions,
        }
    }

    /// Make the positions, both tables and the texture resident.
    fn resident(&self, scene: &Scene, encoder: &mut RenderCommandEncoder) {
        let positions: Vec<ResourceRef> = self.positions.iter().map(ResourceRef::Buffer).collect();
        encoder
            .use_resources(&positions, ResourceUsage::READ, RenderStages::VERTEX)
            .unwrap();
        let both = RenderStages::VERTEX | RenderStages::FRAGMENT;
        for table in &self.tables {
            (encoder.use_resource(table.as_resource(), ResourceUsage::READ, both)).unwrap();
        }
        let sampled = scene.sampled.as_resource();
        (encoder.use_resource(sampled, ResourceUsage::READ, RenderStages::FRAGMENT)).unwrap();
    }
}

/// What a draw changed: the resident uses its draw worked out, the kept ones checked, and the
/// ones its gathering, its checks against its pass's fragment stages and attachments, and its
/// noting walked.
#[derive(Debug, Default)]
struct Drawn {
    made: u64,
    settled: u64,
    checked: u64,
    gathered: u64,
    against: u64,
    attachments: u64,
    noted: u64,
    unordered: u64,
    indirect: u64,
    geometry: u64,
    geometry_checked: u64,
    again: u64,
    heap_parts: u64,
    heap_checked: u64,
    unreadable_looks: u64,
    facts_made: u64,
}

/// Draw with `draw`, and what that changed.
fn drawn(device: &Device, draw: impl FnOnce()) -> Drawn {
    let before = (
        device.work_counts(),
        kept_resident_checks(),
        kept_heap_checks(),
        plan_facts_made(),
    );
    draw();
    let after = (
        device.work_counts(),
        kept_resident_checks(),
        kept_heap_checks(),
        plan_facts_made(),
    );
    Drawn {
        made: after.0.resident_uses_made - before.0.resident_uses_made,
        settled: after.0.resident_keys_settled - before.0.resident_keys_settled,
        checked: after.1 - before.1,
        gathered: after.0.resident_uses_gathered - before.0.resident_uses_gathered,
        against: after.0.resident_uses_against_fragments - before.0.resident_uses_against_fragments,
        attachments: (after.0.resident_uses_against_attachments)
            - before.0.resident_uses_against_attachments,
        noted: after.0.resident_uses_noted - before.0.resident_uses_noted,
        unordered: (after.0.resident_uses_unordered_writes)
            - before.0.resident_uses_unordered_writes,
        indirect: after.0.resident_uses_indirect_reads - before.0.resident_uses_indirect_reads,
        geometry: (after.0.resident_uses_geometry_writes) - before.0.resident_uses_geometry_writes,
        geometry_checked: (after.0.resident_uses_geometry_checked)
            - before.0.resident_uses_geometry_checked,
        again: after.0.resident_uses_again_checked - before.0.resident_uses_again_checked,
        heap_parts: after.0.draw_heap_parts_made - before.0.draw_heap_parts_made,
        heap_checked: after.2 - before.2,
        unreadable_looks: (after.0.draw_resident_textures_looked_at)
            - before.0.draw_resident_textures_looked_at,
        facts_made: after.3 - before.3,
    }
}

/// One encoder of `resources`, each draw's [`Drawn`]: [`POSITIONS`] draws of the entry shape,
/// draw `i` binding table `i % 2`'s entry of position `i` at its vertex and fragment buffer 0,
/// with the encoder's `later` buffer made resident before draw `later`; then one draw of the slot
/// shape, which reads no pointer, and one of the entry shape again.
fn encode(scene: &Scene, resources: &Resources, later: u64) -> Vec<Drawn> {
    let device = &scene.device;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&scene.target))
        .unwrap();
    resources.resident(scene, &mut encoder);
    encoder.set_render_pipeline_state(&scene.entry).unwrap();
    let mut each = Vec::new();
    let entry = |encoder: &mut RenderCommandEncoder, at: u64| {
        let table = &resources.tables[(at % 2) as usize];
        let offset = ENTRY * (1 + at % POSITIONS);
        (encoder.set_vertex_buffer(Some(table), offset, 0)).unwrap();
        (encoder.set_fragment_buffer(Some(table), offset, 0)).unwrap();
        drawn(device, || {
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
        })
    };
    for at in 0..POSITIONS {
        if at == later {
            let later = resources.later.as_resource();
            (encoder.use_resource(later, ResourceUsage::READ, RenderStages::VERTEX)).unwrap();
        }
        each.push(entry(&mut encoder, at));
    }
    encoder.set_render_pipeline_state(&scene.slot).unwrap();
    (encoder.set_vertex_buffer(Some(&resources.positions[0]), 0, 0)).unwrap();
    each.push(drawn(device, || {
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
    }));
    encoder.set_render_pipeline_state(&scene.entry).unwrap();
    each.push(entry(&mut encoder, 1));
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    each
}

/// Two encoders, the second over other resources of the same residency's shape: within each,
/// the draws after the first work out the uses of what it made resident only after it made
/// another resource resident (the first encoder's fourth), and only those walk them as they
/// gather them for their render pass instance, check them against its fragment stages and
/// attachments, and note them. The last, after a draw of another pipeline state, keeps those its
/// state's last draw worked out and noted, and notes them again, as its encoder keeps only its
/// last draw's noting apart; the keys it reaches through the resident set alone are those of its
/// state's last draw, which bound the same table, so it moves none. Every other draw keeps them,
/// and its kept uses are those it would make whole, and of the keys it reaches through the
/// resident set alone moves only the two tables, the one it binds and the one the last draw
/// bound, however many the encoder made resident. The second encoder's first draw, of the plan
/// of the first's last (the second makes every resource resident before it) under another
/// residency, works them out again.
#[test]
fn draws_keep_the_uses_of_what_their_encoder_made_resident_until_it_or_their_plan_changes() {
    let scratch = Scratch::new("resident-uses");
    let device = device(&scratch.0, &[], &Heard::default());
    let scene = Scene::new(&device);
    for (encoder, corner, later) in [(0, -0.9, 3), (1, 0.1, 0)] {
        let resources = Resources::new(&scene, corner);
        let each = encode(&scene, &resources, later);
        let entry_draws = POSITIONS as usize;
        let made_again = [0, later as usize];
        for (at, drawn) in each.iter().enumerate() {
            let what = format!("encoder {encoder} draw {at}: {drawn:?}");
            // No draw of the scene writes before its fragments: none checks against such writes.
            assert_eq!(drawn.geometry_checked, 0, "{what}");
            let walked = [
                drawn.made,
                drawn.gathered,
                drawn.against,
                drawn.attachments,
                drawn.noted,
                drawn.unordered,
                drawn.indirect,
                drawn.geometry,
                drawn.again,
            ];
            match at {
                // The slot shape's plan lists none.
                at if at == entry_draws => {
                    assert_eq!(walked, [0; 9], "{what}");
                    assert_eq!(drawn.settled, 0, "{what}");
                }
                at if made_again.contains(&at) => {
                    assert!(walked.iter().all(|walked| *walked > 0), "{what}");
                    // The positions and the tables, worked out whole.
                    assert!(drawn.settled > POSITIONS, "{what}");
                }
                // After the slot shape's draw, from the entry state's own bind state.
                at if at == entry_draws + 1 => {
                    let (noted, kept) = (walked[4], [&walked[..4], &walked[5..]].concat());
                    assert!(
                        noted > 0 && kept.iter().all(|walked| *walked == 0),
                        "{what}"
                    );
                    assert!(drawn.checked > 0, "{what}");
                    assert_eq!(drawn.settled, 0, "{what}");
                }
                _ => {
                    assert_eq!(walked, [0; 9], "{what}");
                    assert!(drawn.checked > 0, "{what}");
                    assert_eq!(drawn.settled, 2, "{what}");
                }
            }
        }
        assert!(scene.covered() > 0, "encoder {encoder} drew nothing");
    }
}

/// The two encoders of the test above: within each, a draw works out its heap part (its plan's
/// heap tables, where its heap writes go and its writes past the heap's entries:
/// `render/heap_part.rs`) at the encoder's first draw and after the encoder made another
/// resource resident (the first encoder's fourth), and keeps it at every other, though each
/// binds other buffers than the last; each kept part is checked against one worked out whole.
/// The slot shape's draw, whose plan reads no handle, works none out, and the entry shape's draw
/// after it keeps the entry plan's part, kept beside it.
#[test]
fn draws_keep_their_heap_part_until_what_their_encoder_made_resident_changes() {
    let scratch = Scratch::new("resident-heap-parts");
    let device = device(&scratch.0, &[], &Heard::default());
    let scene = Scene::new(&device);
    for (encoder, corner, later) in [(0, -0.9, 3), (1, 0.1, 0)] {
        let resources = Resources::new(&scene, corner);
        let each = encode(&scene, &resources, later);
        let slot_draw = POSITIONS as usize;
        for (at, drawn) in each.iter().enumerate() {
            let what = format!("encoder {encoder} draw {at}: {drawn:?}");
            let made = at == 0 || at == later as usize;
            assert_eq!(drawn.heap_parts, u64::from(made), "{what}");
            if !made && at != slot_draw {
                assert!(drawn.heap_checked > 0, "{what}");
            }
        }
        assert!(scene.covered() > 0, "encoder {encoder} drew nothing");
    }
}

/// The two encoders of the tests above: within each, a draw whose functions read texture handles
/// looks at the textures its encoder made resident for those whose bytes a carrier holds
/// (`pvrtc::ResidentUnreadable`) at the encoder's first draw and after the encoder made another
/// resource resident (the first encoder's fourth), and at no other draw, though each binds other
/// buffers than the last. The slot shape's draw, whose functions read no handle, looks at none.
#[test]
fn draws_look_for_textures_a_carrier_holds_until_what_their_encoder_made_resident_changes() {
    let scratch = Scratch::new("resident-unreadable");
    let device = device(&scratch.0, &[], &Heard::default());
    let scene = Scene::new(&device);
    for (encoder, corner, later) in [(0, -0.9, 3), (1, 0.1, 0)] {
        let resources = Resources::new(&scene, corner);
        let each = encode(&scene, &resources, later);
        for (at, drawn) in each.iter().enumerate() {
            let what = format!("encoder {encoder} draw {at}: {drawn:?}");
            let looked = at == 0 || at == later as usize;
            assert_eq!(drawn.unreadable_looks, u64::from(looked), "{what}");
        }
        assert!(scene.covered() > 0, "encoder {encoder} drew nothing");
    }
}

/// The two encoders of the tests above: what a draw asks of its plan's descriptor writes, which
/// number the device's textures (`render/plan_facts.rs`), is worked out as its pipeline state
/// settles the plan for its bind shape: at most at the entry shape's first draw, after the
/// encoder made another resource resident (the first encoder's fourth: the used resources are
/// the shape's), and at the slot shape's draw, and at no other draw, though each binds other
/// buffers than the last.
#[test]
fn draws_work_out_what_they_ask_of_their_plan_s_writes_once_a_settle() {
    let scratch = Scratch::new("resident-plan-facts");
    let device = device(&scratch.0, &[], &Heard::default());
    let scene = Scene::new(&device);
    for (encoder, corner, later) in [(0, -0.9, 3), (1, 0.1, 0)] {
        let resources = Resources::new(&scene, corner);
        let each = encode(&scene, &resources, later);
        let slot_draw = POSITIONS as usize;
        for (at, drawn) in each.iter().enumerate() {
            let what = format!("encoder {encoder} draw {at}: {drawn:?}");
            match at == 0 || at == later as usize || at == slot_draw {
                true => assert!(drawn.facts_made <= 1, "{what}"),
                false => assert_eq!(drawn.facts_made, 0, "{what}"),
            }
        }
        assert!(scene.covered() > 0, "encoder {encoder} drew nothing");
    }
}

/// Draws of the entry shape that alternate a point list and a triangle list, each binding other
/// buffers than the last: a draw of each type may take a plan of its own, and plans that list the
/// same uses share the uses of what the encoder made resident
/// (`binding::command_uses_into`), so only the first draw works them out; each plan's heap part is
/// worked out at its first draw alone.
#[test]
fn draws_of_two_primitive_types_keep_what_their_encoder_made_resident_alike() {
    let scratch = Scratch::new("resident-primitive-types");
    let device = device(&scratch.0, &[], &Heard::default());
    let scene = Scene::new(&device);
    let resources = Resources::new(&scene, -0.9);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&scene.target))
        .unwrap();
    resources.resident(&scene, &mut encoder);
    encoder.set_render_pipeline_state(&scene.entry).unwrap();
    let mut each = Vec::new();
    for at in 0..POSITIONS {
        let table = &resources.tables[(at % 2) as usize];
        let offset = ENTRY * (1 + at % POSITIONS);
        (encoder.set_vertex_buffer(Some(table), offset, 0)).unwrap();
        (encoder.set_fragment_buffer(Some(table), offset, 0)).unwrap();
        let primitive = [PrimitiveType::Point, PrimitiveType::Triangle][(at % 2) as usize];
        each.push(drawn(&device, || {
            (encoder.draw_primitives(primitive, 0, 3, 1, 0)).unwrap();
        }));
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    for (at, drawn) in each.iter().enumerate() {
        let what = format!("draw {at}: {drawn:?}");
        assert_eq!(drawn.made > 0, at == 0, "{what}");
        match at {
            0 => assert_eq!(drawn.heap_parts, 1, "{what}"),
            // The triangle list's plan, where it takes one of its own.
            1 => assert!(drawn.heap_parts <= 1, "{what}"),
            _ => assert_eq!(drawn.heap_parts, 0, "{what}"),
        }
    }
}

/// Draws of the entry shape after a draw whose fragments write a position the encoder made
/// resident to read in the vertex stage: each reads in place, in its vertex stage through what the
/// encoder made resident, bytes a fragment stage of the pass wrote (`render/pass_start_reads.rs`),
/// and announces it at each occurrence, where the device announces each, though those after the
/// first keep the uses of what the encoder made resident and so checks them as the last did.
#[test]
fn draws_announce_each_read_of_what_their_encoder_made_resident_their_pass_s_fragments_wrote() {
    let scratch = Scratch::new("resident-uses-fragments");
    let heard = Heard::default();
    let kinds = [Degradation::InPlaceVertexReads];
    let device = device(&scratch.0, &kinds, &heard);
    let scene = Scene::new(&device);
    let marker = render_state(&device, MARKER, "rt6_cover", "rt6_mark_sample");
    let resources = Resources::new(&scene, -0.9);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&scene.target))
        .unwrap();
    resources.resident(&scene, &mut encoder);
    let table = &resources.tables[0];
    let mut each = Vec::new();
    for at in 0..4 {
        if at == 1 {
            encoder.set_render_pipeline_state(&marker).unwrap();
            let position = &resources.positions[0];
            (encoder.set_fragment_buffer(Some(position), 0, 0)).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
        }
        encoder.set_render_pipeline_state(&scene.entry).unwrap();
        (encoder.set_vertex_buffer(Some(table), ENTRY, 0)).unwrap();
        (encoder.set_fragment_buffer(Some(table), ENTRY, 0)).unwrap();
        let before = heard.lock().unwrap().len();
        let drawn = drawn(&device, || {
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
        });
        let announced = (heard.lock().unwrap()[before..].iter())
            .filter(|notice| kinds.contains(&notice.degradation))
            .count();
        each.push((drawn, announced));
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    for (at, (drawn, announced)) in each.iter().enumerate() {
        let what = format!("draw {at}: {drawn:?}, announced {announced}");
        assert!(drawn.against > 0, "{what}");
        match at {
            // Before the marker; the rest, the first after it of its own state's bind state, which
            // the marker's draw went beside.
            0 => assert!(drawn.made > 0 && *announced == 0, "{what}"),
            _ => assert!(drawn.made == 0 && *announced == 1, "{what}"),
        }
    }
    assert!(scene.covered() > 0, "the draws drew nothing");
}

/// Draws of the entry shape whose encoder made their pass's attachment resident: each reads, in
/// its fragment stage through what the encoder made resident, memory the attachment renders
/// (`EncoderCore::degrade_attachment_reads`), and announces it at each occurrence, where the device
/// announces each, though those after the first keep the uses of what the encoder made resident:
/// those found to share memory with an attachment are checked at each draw.
#[test]
fn draws_announce_each_read_of_an_attachment_their_encoder_made_resident() {
    let scratch = Scratch::new("resident-uses-attachments");
    let heard = Heard::default();
    let kinds = [Degradation::AttachmentReadInPlace];
    let device = device(&scratch.0, &kinds, &heard);
    let scene = Scene::new(&device);
    let resources = Resources::new(&scene, -0.9);
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    target.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    resources.resident(&scene, &mut encoder);
    let read = (ResourceUsage::READ, RenderStages::FRAGMENT);
    (encoder.use_resource(target.as_resource(), read.0, read.1)).unwrap();
    encoder.set_render_pipeline_state(&scene.entry).unwrap();
    let table = &resources.tables[0];
    (encoder.set_vertex_buffer(Some(table), ENTRY, 0)).unwrap();
    (encoder.set_fragment_buffer(Some(table), ENTRY, 0)).unwrap();
    let mut each = Vec::new();
    for _ in 0..3 {
        let before = heard.lock().unwrap().len();
        let drawn = drawn(&device, || {
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
        });
        let announced = (heard.lock().unwrap()[before..].iter())
            .filter(|notice| kinds.contains(&notice.degradation))
            .count();
        each.push((drawn, announced));
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    for (at, (drawn, announced)) in each.iter().enumerate() {
        let what = format!("draw {at}: {drawn:?}, announced {announced}");
        assert!(drawn.attachments > 0 && *announced == 1, "{what}");
        assert_eq!(drawn.made > 0, at == 0, "{what}");
    }
}
