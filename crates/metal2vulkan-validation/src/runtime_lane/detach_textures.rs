//! Textures over detached bytes (`detach_backing`), on the device the cfg selects, under the
//! validation layer. What Metal answers, as an M4 under macOS 27.0.1 measured it:
//! a texture over a no-copy buffer's bytes, or in a no-copy heap, stays while they are detached and
//! is made over them meanwhile; its size queries answer its descriptor; a kernel that reaches it
//! through an argument buffer without naming it, where no command buffer made its pages resident,
//! reads zero in every texel and its writes land nowhere a read sees; a command that names it fails
//! its command buffer with code 8; and once the bytes lie in pages again the texture reads and
//! writes them.
//!
//! The kernel is the public case [`QUERY_READ`]: thread `g` writes `source(g) + (width,
//! is_null, 0, 0)` at texel `g` of `target`, two 4x1 `RGBA32Float` textures of its argument
//! buffer. A source over detached bytes reached unnamed gives `(4, 0, 0, 0)` in every texel.
//!
//! - **Both forms, both times**: a texture over a no-copy buffer and one in a no-copy heap, each
//!   made before the detach and made while detached, and a view made while detached.
//! - **Unnamed writes**: a write through one texture's stand-in, then reads of another of the
//!   same plan and of the first, read zero, as Metal's do.
//! - **Named uses** fail `DetachedBacking` (code 8); CPU access is refused
//!   (`NotImplemented(DetachedTexelAccess)`).
//! - **Resident before the detach**: a submission that may reach such a texture announces
//!   `StaleDetachedReach` once for it, with its ID, and counts it once in `WorkCounts`.
//! - **Through storage** (a measured open gap): a kernel that writes such a texture through a
//!   storage-class element, then reads it so, reads its write back from the stand-in's sink,
//!   where Metal reads zero; `DetachedStorageKeepsWrites` is announced once for it, with its ID,
//!   and `StaleDetachedReach` is not ([`DETACHED_REACHES`]). A device without
//!   `shaderStorageImageReadWithoutFormat` refuses that kernel, typed.
//! - **Only its own slots**: a detach, and a rebind, of one form's bytes write again the heap slots
//!   of the textures over them alone (`WorkCounts::heap_slots_rewritten`), and every texture,
//!   those over the other form's bytes and one of the device's own among them, reads its own
//!   texels through its slot after each.
//! - **Attached again**: a rebind gives every texture its pages back, those made while detached
//!   among them; a kernel's write of one made while detached lands in the new pages, merged back,
//!   as the CPU may write them.
//! - **A two-plane extended-range heap texture** made in a detached heap answers its descriptor.

use super::degrade_notices::on_device as on_device_noticed;
use super::degrade_notices::on_device_with as on_device_noticed_with;
use super::depth_compares::made_pipeline_of;
use super::host_memory::on_device;
use super::rebind_reach::{heap_member, member_over, rebind_heap_to, rebind_to, A, B};
use super::{
    at_plan, bound_buffers, encode_case, pipeline_state, public_case, CaseResources, PublicCase,
};
use super::{buffer_with, texture_descriptor, WAIT};
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Capability, Feature, Needs};
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    ComputePipelineState, Degradation, Device, Error, Heap, Origin, Region, ResourceOptions,
    RuntimeGap, Size, Texture, TextureDescriptor,
};
use std::collections::BTreeMap;

/// The kinds the reach rows take: stale pages, and a stand-in's sink.
pub(super) const DETACHED_REACHES: [Degradation; 2] = [
    Degradation::StaleDetachedReach,
    Degradation::DetachedStorageKeepsWrites,
];

/// The case whose kernel reads, queries and null-tests its argument buffer's source texture and
/// writes its target.
const QUERY_READ: &str = "read-query-null-test-and-write-argument-buffer-textures";

/// The source's place in the argument buffer: binding 0, field 0.
const SOURCE: (u32, u32) = (0, 0);

/// The target's place: binding 0, field 8.
const TARGET: (u32, u32) = (0, 8);

/// What every texel of the target holds where the source is over detached bytes: zero read, the
/// width 4, not null.
const DETACHED: [f32; 4] = [4.0, 0.0, 0.0, 0.0];

/// The bytes a no-copy buffer's textures lie apart at, and their rows' pitch.
const APART: u64 = 4096;
const ROW: u64 = 256;

/// The bytes the texture is over.
#[derive(Clone, Copy, Debug)]
enum Over {
    /// A no-copy buffer: the texture is over its bytes.
    Buffer,
    /// A no-copy heap: the texture is placed in it.
    Heap,
}

/// The texture is made before the bytes are detached, or while they are.
#[derive(Clone, Copy, Debug)]
enum Made {
    Before,
    While,
}

/// What a form's textures are made over: the no-copy buffer or heap.
enum Bytes {
    Buffer(Buffer),
    /// The heap ([`heap_member`]'s).
    Heap(Heap),
}

impl Bytes {
    /// The form's bytes on `device`; `None`, printed, where the device imports no host memory.
    fn new(device: &Device, over: Over) -> Option<Bytes> {
        match over {
            Over::Buffer => member_over(device, A).map(|(buffer, _)| Bytes::Buffer(buffer)),
            Over::Heap => heap_member(device).map(|(heap, _member)| Bytes::Heap(heap)),
        }
    }

    /// The `index`th texture of `descriptor` over them: `None`, printed, where the device makes
    /// no such texture in a heap over host memory.
    fn texture(&self, descriptor: &TextureDescriptor, index: u64) -> Option<Texture> {
        let made = match self {
            Bytes::Buffer(buffer) => buffer.new_texture(descriptor, index * APART, ROW),
            Bytes::Heap(heap) => {
                let size = heap
                    .device()
                    .heap_texture_size_and_align(descriptor)
                    .unwrap();
                let apart = size.size.div_ceil(size.align) * size.align;
                heap.new_texture_at_offset(descriptor, index * apart)
            }
        };
        match made {
            Ok(texture) => Some(texture),
            Err(error @ Error::NotImplemented(RuntimeGap::HeapTextureMemory)) => {
                eprintln!("DETACH textures: no texture in a heap over host memory: {error:?}");
                None
            }
            Err(error) => panic!("the texture over the bytes: {error:?}"),
        }
    }

    /// Detach them, at once.
    fn detach(&self) {
        let detached = match self {
            Bytes::Buffer(buffer) => buffer.detach_backing(),
            Bytes::Heap(heap) => heap.detach_backing(),
        };
        assert_eq!(detached, Ok(true));
    }

    /// Give them fresh pages.
    fn rebind(&self) {
        match self {
            Bytes::Buffer(buffer) => drop(rebind_to(buffer, B)),
            Bytes::Heap(heap) => rebind_heap_to(heap, B),
        }
    }

    /// The resource options a texture over them takes: the heap's, or the buffer's.
    fn options(&self) -> ResourceOptions {
        match self {
            Bytes::Buffer(_) => ResourceOptions::default(),
            Bytes::Heap(heap) => heap.resource_options(),
        }
    }
}

/// [`QUERY_READ`] on a device: its pipeline state, a queue, and its source texels.
struct QueryRead {
    case: PublicCase,
    pipeline: ComputePipelineState,
    queue: CommandQueue,
}

impl QueryRead {
    fn new(device: &Device) -> QueryRead {
        let case = public_case(QUERY_READ).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        QueryRead {
            case,
            pipeline,
            queue: device.new_command_queue().unwrap(),
        }
    }

    /// The descriptor of the case's textures, readable and writable, of `options`.
    fn descriptor(&self, options: ResourceOptions) -> TextureDescriptor {
        let literal = &self.case.compute.argument_buffer_textures[&SOURCE];
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let mut descriptor = texture_descriptor(literal, usage).unwrap();
        descriptor.resource_options = options;
        descriptor
    }

    /// A texture of the device's own, as the case's, holding `texels`.
    fn own(&self, device: &Device, texels: &[f32; 16]) -> Texture {
        let texture = device
            .new_texture(&self.descriptor(ResourceOptions::default()))
            .unwrap();
        fill(&texture, texels);
        texture
    }

    /// The source texels the case gives.
    fn source_texels(&self) -> [f32; 16] {
        let bytes = &self.case.compute.argument_buffer_textures[&SOURCE].bytes;
        floats(bytes)
    }

    /// A command buffer dispatching the case with `source` and `target` in its argument buffer,
    /// its encoder making `resident` resident (and nothing else), committed and complete.
    fn run(
        &self,
        device: &Device,
        source: &Texture,
        target: &Texture,
        resident: &[&Texture],
    ) -> CommandBuffer {
        let naming = CaseResources {
            argument_textures: BTreeMap::from([(SOURCE, source.clone()), (TARGET, target.clone())]),
            ..CaseResources::default()
        };
        let buffers = bound_buffers(device, &self.case.compute, &naming).unwrap();
        // `encode_case` makes the argument textures it is given resident: these, keyed apart.
        let reaching = CaseResources {
            argument_textures: resident
                .iter()
                .enumerate()
                .map(|(field, texture)| ((u32::MAX, field as u32), (*texture).clone()))
                .collect(),
            ..CaseResources::default()
        };
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(
            &mut encoder,
            &self.pipeline,
            &self.case.compute,
            &buffers,
            &reaching,
        )
        .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        command_buffer
    }

    /// [`run`](Self::run) with `source` unnamed into a new target, which alone is resident:
    /// the target's texels, the command buffer having completed.
    fn read_unnamed(&self, device: &Device, source: &Texture) -> [f32; 16] {
        let target = self.own(device, &[0.0; 16]);
        let run = self.run(device, source, &target, &[&target]);
        completed(&run);
        texels(&target)
    }
}

/// `command_buffer` completed with no error.
fn completed(command_buffer: &CommandBuffer) {
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
}

/// The little-endian floats of `bytes`.
fn floats(bytes: &[u8]) -> [f32; 16] {
    let mut floats = [0.0; 16];
    for (float, word) in floats.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
        *float = f32::from_le_bytes(*word);
    }
    floats
}

/// The region of the case's textures: 4x1.
fn whole() -> Region {
    Region::new_2d(0, 0, 4, 1)
}

/// Write `texels` into `texture`.
fn fill(texture: &Texture, texels: &[f32; 16]) {
    let bytes: Vec<u8> = texels
        .iter()
        .flat_map(|float| float.to_le_bytes())
        .collect();
    texture
        .replace_region(whole(), 0, 0, &bytes, 64, 64)
        .unwrap();
}

/// The texels of `texture`.
fn texels(texture: &Texture) -> [f32; 16] {
    let mut bytes = [0; 64];
    texture
        .get_bytes(&mut bytes, 64, 64, whole(), 0, 0)
        .unwrap();
    floats(&bytes)
}

/// [`DETACHED`] in every texel.
fn detached_texels() -> [f32; 16] {
    let mut texels = [0.0; 16];
    for texel in texels.as_chunks_mut::<4>().0.iter_mut() {
        texel.copy_from_slice(&DETACHED);
    }
    texels
}

/// The case's target where `source` holds the case's texels: `(g + 4, 0.5 g, -1, 2)`.
fn read_texels() -> [f32; 16] {
    let mut texels = [0.0; 16];
    for (g, texel) in texels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        texel.copy_from_slice(&[g as f32 + 4.0, 0.5 * g as f32, -1.0, 2.0]);
    }
    texels
}

/// Every form: over a no-copy buffer and in a no-copy heap, made before the detach and while
/// detached.
fn forms() -> [(Over, Made); 4] {
    [
        (Over::Buffer, Made::Before),
        (Over::Buffer, Made::While),
        (Over::Heap, Made::Before),
        (Over::Heap, Made::While),
    ]
}

/// The bytes of form `over`, detached, and the `index`th texture over them made at `made`
/// (holding the case's source texels where made before); `None`, printed, where the device
/// makes none.
fn detached_texture(
    device: &Device,
    query: &QueryRead,
    (over, made): (Over, Made),
) -> Option<(Bytes, Texture)> {
    let bytes = Bytes::new(device, over)?;
    let descriptor = query.descriptor(bytes.options());
    let texture = match made {
        Made::Before => {
            let texture = bytes.texture(&descriptor, 0)?;
            fill(&texture, &query.source_texels());
            bytes.detach();
            texture
        }
        Made::While => {
            bytes.detach();
            bytes.texture(&descriptor, 0)?
        }
    };
    Some((bytes, texture))
}

/// A texture over detached bytes, which no command buffer made resident, answers its size, and
/// a kernel reaching it unnamed reads zero texels (and its width, and not null), every form; a
/// view made while detached does too. The device's stand-in memory is printed.
#[test]
fn an_unnamed_read_of_a_texture_over_detached_bytes_answers_its_size_and_reads_zero() {
    on_device(|device| {
        let query = QueryRead::new(device);
        for form in forms() {
            let Some((_bytes, texture)) = detached_texture(device, &query, form) else {
                return;
            };
            assert_eq!(
                (texture.width(), texture.height(), texture.pixel_format()),
                (4, 1, PixelFormat::RGBA32Float),
                "{form:?}"
            );
            assert_eq!(
                query.read_unnamed(device, &texture),
                detached_texels(),
                "{form:?}"
            );
            let view = texture.new_texture_view(PixelFormat::RGBA32Float).unwrap();
            assert_eq!(view.width(), 4, "{form:?}: the view");
            assert_eq!(
                query.read_unnamed(device, &view),
                detached_texels(),
                "{form:?}: the view"
            );
            let counts = device.work_counts();
            eprintln!(
                "DETACH textures {form:?}: read zero unnamed; {} stand-ins made, {} bytes live",
                counts.stand_ins_made, counts.stand_in_bytes
            );
        }
    });
}

/// A kernel's unnamed write to a texture over detached bytes, then unnamed reads of another of
/// the same plan and of the first, read zero, as Metal's do (its writes land nowhere): the
/// stand-in the two share is never written.
#[test]
fn an_unnamed_write_through_a_stand_in_lands_where_no_read_looks() {
    on_device(|device| {
        let query = QueryRead::new(device);
        for over in [Over::Buffer, Over::Heap] {
            let Some(bytes) = Bytes::new(device, over) else {
                return;
            };
            let descriptor = query.descriptor(bytes.options());
            let (Some(a), Some(b)) = (bytes.texture(&descriptor, 0), bytes.texture(&descriptor, 1))
            else {
                return;
            };
            bytes.detach();
            let source = query.own(device, &query.source_texels());
            let write = query.run(device, &source, &a, &[&source]);
            completed(&write);
            assert_eq!(
                query.read_unnamed(device, &b),
                detached_texels(),
                "{over:?}: b"
            );
            assert_eq!(
                query.read_unnamed(device, &a),
                detached_texels(),
                "{over:?}: a"
            );
            eprintln!("DETACH textures {over:?}: the write landed nowhere read");
        }
    });
}

/// A command that names a texture over detached bytes fails its command buffer with
/// `DetachedBacking` (Metal's code 8), every form; CPU access through it is refused
/// (`NotImplemented(DetachedTexelAccess)`).
#[test]
fn a_named_use_fails_with_code_8_and_cpu_access_is_refused() {
    on_device(|device| {
        let query = QueryRead::new(device);
        let gap = Some(Error::NotImplemented(RuntimeGap::DetachedTexelAccess));
        for form in forms() {
            let Some((_bytes, texture)) = detached_texture(device, &query, form) else {
                return;
            };
            let target = query.own(device, &[0.0; 16]);
            let named = query.run(device, &texture, &target, &[&texture, &target]);
            assert_eq!(named.status(), CommandBufferStatus::Error, "{form:?}");
            assert_eq!(
                named.error(),
                Some(CommandBufferError::DetachedBacking),
                "{form:?}"
            );
            assert_eq!(texels(&target), [0.0; 16], "{form:?}: nothing ran");
            let mut out = [0; 64];
            assert_eq!(
                texture.get_bytes(&mut out, 64, 64, whole(), 0, 0).err(),
                gap,
                "{form:?}"
            );
            assert_eq!(
                texture.replace_region(whole(), 0, 0, &out, 64, 64).err(),
                gap,
                "{form:?}"
            );
        }
        eprintln!("DETACH textures: named uses failed with code 8, CPU access refused");
    });
}

/// A texture a submitted command buffer made resident, whose bytes are detached after, is
/// announced once (`StaleDetachedReach`, with its ID) at the first submission that may reach it
/// unnamed, and counted once: two such submissions, one notice, a count of 1. Metal's read
/// there reads the pages the program gave back; the runtime's reads zero.
#[test]
fn a_texture_detached_after_it_was_resident_is_announced_once_where_reached() {
    on_device_noticed(|device, messages| {
        let query = QueryRead::new(device);
        let Some(bytes) = Bytes::new(device, Over::Heap) else {
            return;
        };
        let descriptor = query.descriptor(bytes.options());
        let Some(texture) = bytes.texture(&descriptor, 0) else {
            return;
        };
        fill(&texture, &query.source_texels());
        let target = query.own(device, &[0.0; 16]);
        let resident = query.run(device, &texture, &target, &[&texture, &target]);
        completed(&resident);
        assert_eq!(texels(&target), read_texels(), "resident before the detach");
        bytes.detach();
        let before = device.work_counts().stale_detached_reaches;
        assert_eq!(query.read_unnamed(device, &texture), detached_texels());
        assert_eq!(query.read_unnamed(device, &texture), detached_texels());
        let reached = device.work_counts().stale_detached_reaches - before;
        assert_eq!(reached, 1, "counted once");
        let notices: Vec<_> = messages
            .take_notices()
            .into_iter()
            .filter_map(|message| message.degradation)
            .filter(|notice| notice.degradation == Degradation::StaleDetachedReach)
            .collect();
        assert_eq!(notices.len(), 1, "announced once: {notices:?}");
        assert_eq!(notices[0].object_id, Some(texture.object_id()));
        eprintln!("DETACH textures: stale reach announced once, counted once");
    });
}

/// Once the bytes lie in pages again, every texture over them reads and writes them, one made
/// while they were detached too: each form, filled through the CPU and read by a kernel that
/// names it.
#[test]
fn a_rebind_gives_every_texture_its_pages_again() {
    on_device(|device| {
        let query = QueryRead::new(device);
        for form in forms() {
            let Some((bytes, texture)) = detached_texture(device, &query, form) else {
                return;
            };
            bytes.rebind();
            fill(&texture, &query.source_texels());
            let target = query.own(device, &[0.0; 16]);
            let named = query.run(device, &texture, &target, &[&texture, &target]);
            completed(&named);
            assert_eq!(texels(&target), read_texels(), "{form:?}");
        }
        eprintln!("DETACH textures: attached again, read and written");
    });
}

/// Texels of the `n`th texture of [`a_rebind_or_a_detach_rewrites_its_own_textures_slots_alone`]
/// as of its `round`th fill, apart from every other texture's and round's.
fn marked(n: usize, round: usize) -> [f32; 16] {
    std::array::from_fn(|i| (n * 100 + round * 20 + i) as f32)
}

/// What [`QUERY_READ`] writes of a source holding `source`: each texel plus `(4, 0, 0, 0)`, its
/// width and not null.
fn read_of(source: &[f32; 16]) -> [f32; 16] {
    std::array::from_fn(|i| source[i] + if i % 4 == 0 { 4.0 } else { 0.0 })
}

/// A detach and a rebind of one form's bytes write again the heap slots of the textures over
/// them alone (`WorkCounts::heap_slots_rewritten`): two for the two textures over them, after a
/// detach and after the rebind that gives them pages again, and two or none after a rebind of
/// attached bytes (none where nothing over them was remade: shadows of the bytes), none for the
/// two textures over the other form's bytes or for the device's own. After each, every texture
/// reads its own texels through its slot: a kernel reads each through its argument buffer, made
/// resident, those over the bytes filled again after the rebind.
#[test]
fn a_rebind_or_a_detach_rewrites_its_own_textures_slots_alone() {
    on_device(|device| {
        let query = QueryRead::new(device);
        let (Some(buffer), Some(heap)) = (
            Bytes::new(device, Over::Buffer),
            Bytes::new(device, Over::Heap),
        ) else {
            return;
        };
        // Two textures over each form's bytes, and one of the device's own, by the form's index.
        let mut textures: Vec<(Option<usize>, Texture)> = Vec::new();
        for (form, bytes) in [&buffer, &heap].into_iter().enumerate() {
            let descriptor = query.descriptor(bytes.options());
            for index in 0..2 {
                let Some(texture) = bytes.texture(&descriptor, index) else {
                    return;
                };
                textures.push((Some(form), texture));
            }
        }
        textures.push((None, query.own(device, &[0.0; 16])));
        let mut held: Vec<[f32; 16]> = (0..textures.len()).map(|n| marked(n, 0)).collect();
        for ((_, texture), texels) in textures.iter().zip(&held) {
            fill(texture, texels);
        }
        let read_all = |held: &[[f32; 16]], step: &str| {
            for (n, ((form, texture), want)) in textures.iter().zip(held).enumerate() {
                let target = query.own(device, &[0.0; 16]);
                let run = query.run(device, texture, &target, &[texture, &target]);
                completed(&run);
                assert_eq!(
                    texels(&target),
                    read_of(want),
                    "{step}: texture {n} over {form:?}"
                );
            }
        };
        read_all(&held, "made");
        let rewritten = || device.work_counts().heap_slots_rewritten;
        let mut round = 0;
        let forms = [(&buffer, "a no-copy buffer"), (&heap, "a no-copy heap")];
        for (form, (bytes, over)) in forms.into_iter().enumerate() {
            let refill = |held: &mut [[f32; 16]], round: usize| {
                for (n, (of, texture)) in textures.iter().enumerate() {
                    if *of == Some(form) {
                        held[n] = marked(n, round);
                        fill(texture, &held[n]);
                    }
                }
            };
            let before = rewritten();
            bytes.rebind();
            let attached = rewritten() - before;
            assert!(
                attached == 0 || attached == 2,
                "{over}: a rebind rewrote {attached} slots"
            );
            round += 1;
            refill(&mut held, round);
            read_all(&held, &format!("{over}, rebound"));

            let before = rewritten();
            bytes.detach();
            assert_eq!(rewritten() - before, 2, "{over}: the detach");
            let before = rewritten();
            bytes.rebind();
            assert_eq!(
                rewritten() - before,
                2,
                "{over}: the rebind after the detach"
            );
            round += 1;
            refill(&mut held, round);
            read_all(&held, &format!("{over}, detached and rebound"));
            eprintln!(
                "DETACH textures: {over}: a rebind rewrote {attached} heap slots, a detach and \
                 its rebind 2 each, of 5 textures"
            );
        }
    });
}

/// A texture made while its bytes are detached is their shadow, presenting a stand-in; once the
/// bytes lie in pages again, a kernel's write of it lands in them, both forms. The bytes are host
/// pages the CPU may write, so the write is merged into them in place of a copy back
/// (`shadow_merge.rs`): no copy back is dispatched, even where the device copies a shadow back by a
/// dispatch (`DriverFact::ShadowCopyBackDispatchFaster`).
#[test]
fn a_texture_made_while_detached_writes_its_pages_once_they_are_given_again() {
    on_device(|device| {
        let query = QueryRead::new(device);
        let before = device.work_counts().shadow_copies_back_dispatched;
        for over in [Over::Buffer, Over::Heap] {
            let Some((bytes, texture)) = detached_texture(device, &query, (over, Made::While))
            else {
                return;
            };
            bytes.rebind();
            let source = query.own(device, &query.source_texels());
            let write = query.run(device, &source, &texture, &[&source, &texture]);
            completed(&write);
            assert_eq!(texels(&texture), read_texels(), "{over:?}");
        }
        let dispatched = device.work_counts().shadow_copies_back_dispatched - before;
        assert_eq!(dispatched, 0, "copies back dispatched");
        eprintln!("DETACH textures: made while detached, written once attached again, merged");
    });
}

/// A two-plane extended-range texture of 11 x 16 made in a detached heap answers its
/// descriptor; a named use (a copy out of it) fails with code 8, and its CPU access is refused
/// until the heap lies in pages again.
#[test]
fn an_extended_range_texture_made_in_a_detached_heap_answers_its_descriptor() {
    on_device(|device| {
        let Some(bytes) = Bytes::new(device, Over::Heap) else {
            return;
        };
        let Bytes::Heap(heap) = &bytes else {
            unreachable!("a heap's bytes");
        };
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGB10A82PXR10, 11, 16, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = bytes.options();
        bytes.detach();
        let texture = match heap.new_texture_at_offset(&descriptor, 0) {
            Ok(texture) => texture,
            Err(error @ (Error::NotImplemented(_) | Error::Unsupported(_))) => {
                eprintln!("DETACH textures: no extended-range texture in this heap: {error:?}");
                return;
            }
            Err(error) => panic!("new_texture_at_offset: {error:?}"),
        };
        assert_eq!(
            (texture.width(), texture.height(), texture.pixel_format()),
            (11, 16, PixelFormat::RGB10A82PXR10)
        );
        let region = Region::new_2d(0, 0, 11, 16);
        let mut out = vec![0; 11 * 16 * 4];
        assert_eq!(
            texture.get_bytes(&mut out, 44, 0, region, 0, 0).err(),
            Some(Error::NotImplemented(RuntimeGap::DetachedTexelAccess))
        );
        let into = device
            .new_buffer(11 * 16 * 4, ResourceOptions::default())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(11, 16, 1),
            &into,
            0,
            44,
            44 * 16,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.error(),
            Some(CommandBufferError::DetachedBacking)
        );
        bytes.rebind();
        texture.get_bytes(&mut out, 44, 0, region, 0, 0).unwrap();
        eprintln!("DETACH textures: an extended-range texture made in a detached heap");
    });
}

/// A kernel `k` whose constant argument buffer 0 is `{texture2d<float, access::read_write> t;
/// texture2d<float, access::write> out;}`: thread `g` writes `t(g) + (t's width, 0, 0, 0)` at
/// texel `g` of `out`, then `(9, 9, 9, 9)` at texel `g` of `t`: `t` read and written through
/// storage-class elements (the AIR `a4_heap_textures`'s, with `t`'s access `read_write`).
const READ_WRITE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Resources = type { %"struct.metal::texture2d", %"struct.metal::texture2d" }
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define void @k(ptr addrspace(2) noundef readonly align 8 dereferenceable(16) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <2 x i32> <i32 poison, i32 0>, i32 %1, i64 0
  %4 = getelementptr inbounds %struct.Resources, ptr addrspace(2) %0, i64 0, i32 0, i32 0
  %5 = load ptr addrspace(1), ptr addrspace(2) %4, align 8
  %6 = tail call ptr addrspace(2) @air.get_read_sampler() #5
  %7 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly %5, ptr addrspace(2) %6, <2 x i32> %3, <2 x i32> zeroinitializer, i32 0, i32 3) #6
  %8 = extractvalue { <4 x float>, i8 } %7, 0
  %9 = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) readonly %5, i32 0) #6
  %10 = tail call fast float @air.convert.f.f32.u.i32(i32 %9) #7
  %11 = insertelement <4 x float> zeroinitializer, float %10, i64 0
  %12 = fadd fast <4 x float> %11, %8
  %13 = getelementptr inbounds %struct.Resources, ptr addrspace(2) %0, i64 0, i32 1, i32 0
  %14 = load ptr addrspace(1), ptr addrspace(2) %13, align 8
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %14, <2 x i32> %3, <4 x float> %12, i32 0, i32 2) #8
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %5, <2 x i32> %3, <4 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>, i32 0, i32 3) #8
  ret void
}

declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #2
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly, ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #3
declare i32 @air.get_width_texture_2d(ptr addrspace(1) readonly, i32) local_unnamed_addr #3
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr #4

attributes #0 = { mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #3 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #6 = { nounwind willreturn memory(argmem: read) }
attributes #7 = { nounwind willreturn memory(none) }
attributes #8 = { nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!air.version = !{!21}
!air.language_version = !{!22}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @k, !10, !11}
!10 = !{}
!11 = !{!12, !16}
!12 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !13, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Resources", !"air.arg_name", !"res"}
!13 = !{i32 0, i32 8, i32 0, !"texture2d<float, read_write>", !"t", !"air.indirect_argument", !14, i32 8, i32 8, i32 0, !"texture2d<float, write>", !"out", !"air.indirect_argument", !15}
!14 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"t"}
!15 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!16 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The measured open gap: a kernel that reaches a texture over detached bytes (made before the
/// detach, never resident) through an argument buffer, reads it and then writes it through
/// storage-class elements, reads its earlier write back from the stand-in's sink at its second run,
/// where Metal's reads zero (its writes dropped). A sampled read of the same texture reads zero,
/// and its size is its own. `DetachedStorageKeepsWrites` is announced once for the texture, with
/// its ID, in the first commit; `StaleDetachedReach` is not.
///
/// A device without `shaderStorageImageReadWithoutFormat` (Intel's ANV) refuses the kernel:
/// a storage texture reached by ID has no format its module could declare, so its read needs
/// that feature (`Unsupported(ArgumentBufferTextures)` naming it).
#[test]
fn a_storage_read_after_an_unnamed_storage_write_reads_the_sink_and_is_announced_once() {
    on_device_noticed_with(at_plan, |device, messages| {
        let query = QueryRead::new(device);
        let Some(bytes) = Bytes::new(device, Over::Heap) else {
            return;
        };
        let descriptor = query.descriptor(bytes.options());
        let Some(texture) = bytes.texture(&descriptor, 0) else {
            return;
        };
        bytes.detach();
        let made = made_pipeline_of(device, &llvm_assemble_text(READ_WRITE).unwrap(), "k");
        let feature = VkFeature::ShaderStorageImageReadWithoutFormat;
        let state = if device
            .compile_device()
            .profile()
            .enabled()
            .features
            .contains(&feature)
        {
            made.unwrap()
        } else {
            let Err(Error::Unsupported(refused)) = made else {
                panic!("made without {feature:?}: {made:?}");
            };
            assert_eq!(refused.feature, Feature::ArgumentBufferTextures);
            assert_eq!(
                refused.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(feature)]])
            );
            eprintln!("DETACH textures: refused: the device lacks {feature:?}");
            return;
        };
        let queue = device.new_command_queue().unwrap();
        let before = device.work_counts();
        let run = || {
            let out = query.own(device, &[0.0; 16]);
            let mut ids = Vec::new();
            for resource in [&texture, &out] {
                ids.extend_from_slice(&resource.gpu_resource_id().0.to_le_bytes());
            }
            let arguments = buffer_with(device, &ids).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_buffer(Some(&arguments), 0, 0).unwrap();
            encoder
                .use_resource(out.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            encoder
                .dispatch_threads(Size::new(4, 1, 1), Size::new(4, 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            completed(&command_buffer);
            texels(&out)
        };
        assert_eq!(run(), detached_texels(), "the first run reads zero");
        let mut kept = [0.0; 16];
        for texel in kept.as_chunks_mut::<4>().0.iter_mut() {
            texel.copy_from_slice(&[13.0, 9.0, 9.0, 9.0]);
        }
        assert_eq!(
            run(),
            kept,
            "the second run reads the sink's write (Metal: zero)"
        );
        assert_eq!(
            query.read_unnamed(device, &texture),
            detached_texels(),
            "sampled"
        );
        let after = device.work_counts();
        assert_eq!(
            after.detached_storage_reaches - before.detached_storage_reaches,
            1
        );
        assert_eq!(after.stale_detached_reaches, before.stale_detached_reaches);
        let notices = messages.take_notices();
        let of = |kind| {
            notices
                .iter()
                .filter_map(|message| message.degradation.as_ref())
                .filter(|notice| notice.degradation == kind)
                .map(|notice| notice.object_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            of(Degradation::DetachedStorageKeepsWrites),
            [Some(texture.object_id())],
            "announced once, with its ID"
        );
        assert_eq!(of(Degradation::StaleDetachedReach), []);
        eprintln!("DETACH textures: the storage residue measured and announced once");
    });
}
