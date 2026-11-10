//! Device tests of no-copy buffers whose pointer misses the buffer's memory alignment on the device
//! the cfg selects, under the validation layer. Such a buffer's bytes start past its `VkBuffer`'s
//! start, by the pointer's offset modulo the alignment (or 16 bytes, where that is larger), and
//! each use reaches them there: a kernel through its descriptors and device addresses; a draw's
//! vertex-stage buffers, and its index buffer and arguments in place or, where Vulkan reads neither
//! at that offset, through a copy made before the pass; visibility results through an upload region
//! copied on into the buffer; a 2D texture over the buffer through a staging buffer of its own; a
//! texture buffer in place, or through a staging buffer where Vulkan views no texels at its offset;
//! blits, staged where Vulkan cannot copy texels in place. A pointer 4 bytes past a page starts the
//! bytes past the `VkBuffer`'s on every device; 1, 2 and 3 bytes past it also miss an index's, an
//! argument's or a word's alignment.
//!
//! Each answer is the one the same use gives over a buffer at a page: the probes' and cases'
//! measured Metal answers, or the runtime's own run over its own buffers. No byte of the pages
//! outside a buffer changes.

use super::degrade_notices::{assert_made_once, forced, shared_in_system_memory, HOST_BYTES};
use super::host_memory::on_device_with;
use super::host_memory::{no_copy, on_device, run, Mapping, GUEST_PAGE};
use super::render_state::{quiet, rows, validated, Probe, EMPTY, FULL, PASS, WHITE};
use super::{
    at_plan, buffer_with, encode_case, pipeline_state, public_case, read_buffer, run_case,
    texture_descriptor, CaseResources, PublicCase,
};
use crate::candidate::engine::ComputeOutput;
use crate::case::TextureType;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, Degradation, Device, Origin, RenderCommandEncoder,
    ResourceOptions, Size, TextureDescriptor, VisibilityResultMode,
};
use std::collections::BTreeMap;
use std::ptr::NonNull;

/// The bytes of the pages around each buffer.
const AROUND: u8 = 0x3c;

/// A kernel of 32-bit loads and stores over an input buffer (0) and an output buffer (1).
const KERNEL: &str = "saturating-halving-and-wide-product-integers-at-their-edges";

/// Samples three 2x2 `rgba32_float` 2D textures (texture 0 among them) into its output buffer.
pub(super) const SAMPLER: &str = "sample-each-of-three-textures-through-one-dynamic-array-index";

/// Writes one of two 4x1 `rgba32_float` 2D storage textures per thread (texture 0, its output).
pub(super) const TEXTURE_WRITER: &str = "write-through-a-storage-texture-selected-per-thread";

/// Writes eight `rgba8_uint` texels through a texture buffer (texture 1, its output).
const TEXEL_WRITER: &str = "write-four-bytes-into-a-storage-texel-buffer";

/// Reads a four-texel `r32_uint` texture buffer (texture 0) into its 32-byte output buffer (1)
/// and writes another (texture 1), both past their ends too.
const TEXEL_READER: &str = "reads-and-writes-past-a-texture-buffer";

/// A no-copy buffer over bytes `at` bytes past the start of two fresh guest pages, whose other
/// bytes are [`AROUND`]. The buffer is released before the pages (field order).
struct HostBytes {
    buffer: Buffer,
    pages: Mapping,
    at: usize,
    len: usize,
}

impl HostBytes {
    /// A no-copy buffer over `bytes`, `at` bytes into its pages: made on every device.
    fn new(device: &Device, bytes: &[u8], at: usize) -> HostBytes {
        let pages = Mapping::anonymous(2 * GUEST_PAGE);
        pages.fill(AROUND);
        pages.write(at, bytes);
        let pointer = NonNull::new(pages.pointer().as_ptr().wrapping_add(at)).unwrap();
        // SAFETY: the pages outlive the buffer, which is released first.
        let made = unsafe {
            no_copy(
                device,
                pointer,
                bytes.len(),
                ResourceOptions::default(),
                None,
            )
        };
        let buffer = made.unwrap_or_else(|error| panic!("{at} bytes past a page: {error:?}"));
        assert_eq!(buffer.contents(), Some(pointer));
        HostBytes {
            buffer,
            pages,
            at,
            len: bytes.len(),
        }
    }

    /// The buffer's bytes, read through its pages.
    fn bytes(&self) -> Vec<u8> {
        self.pages.read(self.at..self.at + self.len)
    }

    /// Fail if a byte of the pages outside the buffer changed.
    fn check_around(&self, name: &str) {
        let all = self.pages.read(0..2 * GUEST_PAGE);
        let outside = all[..self.at].iter().chain(&all[self.at + self.len..]);
        assert!(
            outside.into_iter().all(|&byte| byte == AROUND),
            "{name}: a byte outside the buffer changed"
        );
    }
}

/// Little-endian `u32`s.
fn u32s(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Little-endian `u16`s.
fn u16s(indices: &[u16]) -> Vec<u8> {
    indices.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// A public kernel case runs over no-copy buffers 4 bytes past a page, its input read and its
/// output written through their descriptors and device addresses: its output is Metal's.
#[test]
fn kernels_reach_bytes_that_start_past_their_vkbuffer() {
    on_device(|device| {
        let case = public_case(KERNEL).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let held: BTreeMap<u32, HostBytes> = case
            .compute
            .buffers
            .iter()
            .map(|(&index, bytes)| (index, HostBytes::new(device, bytes, 4)))
            .collect();
        let buffers: BTreeMap<u32, Buffer> = held
            .iter()
            .map(|(&index, host)| (index, host.buffer.clone()))
            .collect();
        let resources = CaseResources::new(device, &case).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &pipeline, &case.compute, &buffers, &resources).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = case.compute.output
        else {
            panic!("{KERNEL}: a buffer output");
        };
        let bytes = held[&binding].bytes();
        assert_eq!(
            bytes[offset as usize..(offset + length) as usize],
            case.metal_output
        );
        for (index, host) in &held {
            host.check_around(&format!("buffer {index}"));
        }
    });
}

/// Blits over buffers 1 to 4 bytes past a page: copies and fills of any range move exactly their
/// bytes, and an `rgba32_float` region copied into a texture and back (Vulkan copies its texels
/// only at multiples of 16 bytes, so both copies are staged) is the bytes it was.
#[test]
fn blits_move_bytes_that_start_past_their_vkbuffer() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let counting = |n: usize| -> Vec<u8> { (0..n).map(|i| (i * 7 % 251) as u8).collect() };
        for at in [1, 2, 3, 4] {
            let source = HostBytes::new(device, &counting(300), at);
            let destination = HostBytes::new(device, &[0; 300], at + 1);
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&source.buffer, 5, &destination.buffer, 3, 290)
                .unwrap();
            blit.fill_buffer(&source.buffer, 1..299, 0xe1).unwrap();
            blit.end_encoding().unwrap();
            run(&command_buffer);
            let mut copied = vec![0; 300];
            copied[3..293].copy_from_slice(&counting(300)[5..295]);
            assert_eq!(destination.bytes(), copied, "a copy {at} bytes past a page");
            let mut filled = counting(300);
            filled[1..299].fill(0xe1);
            assert_eq!(source.bytes(), filled, "a fill {at} bytes past a page");
            source.check_around("the copy's source");
            destination.check_around("the copy's destination");
        }
        texels_round_trip(device, &queue);
    });
}

/// A 4x4 `rgba32_float` region from a buffer 4 bytes past a page into a texture, and back into
/// one 12 bytes past a page.
fn texels_round_trip(device: &Device, queue: &CommandQueue) {
    let texels: Vec<u8> = (0..64u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let source = HostBytes::new(device, &texels, 4);
    let back = HostBytes::new(device, &[0; 256], 12);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, 4, 4, false);
    descriptor.resource_options = ResourceOptions::private();
    let image = device.new_texture(&descriptor).unwrap();
    let (size, origin, none) = (Size::new(4, 4, 1), Origin::new(0, 0, 0), BlitOption::None);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(&source.buffer, 0, 64, 256, size, &image, 0, 0, origin, none)
        .unwrap();
    blit.copy_from_texture_to_buffer(&image, 0, 0, origin, size, &back.buffer, 0, 64, 256, none)
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    assert_eq!(back.bytes(), texels);
    back.check_around("the texels copied back");
}

/// `case`'s run with each texture `under` names over its buffer, from the buffer's first byte
/// (rows packed; a texture buffer's row the literal's length): its buffers and resources.
pub(super) fn over(
    device: &Device,
    case: &PublicCase,
    under: &[(u32, &Buffer)],
) -> (BTreeMap<u32, Buffer>, CaseResources) {
    let buffers = case
        .compute
        .buffers
        .iter()
        .map(|(&index, bytes)| (index, buffer_with(device, bytes).unwrap()))
        .collect();
    let mut resources = CaseResources::new(device, case).unwrap();
    for &(index, buffer) in under {
        let literal = &case.compute.textures[&index];
        let descriptor = texture_descriptor(literal, case.texture_usage[&index]).unwrap();
        let row = match literal.texture_type {
            TextureType::Buffer => literal.bytes.len() as u64,
            _ => descriptor.width * literal.format.bytes_per_pixel() as u64,
        };
        let texture = buffer.new_texture(&descriptor, 0, row).unwrap();
        resources.textures.insert(index, texture);
    }
    (buffers, resources)
}

/// `case`'s output, a range of one of `buffers`.
pub(super) fn buffer_output(case: &PublicCase, buffers: &BTreeMap<u32, Buffer>) -> Vec<u8> {
    let ComputeOutput::Buffer {
        binding,
        offset,
        length,
    } = case.compute.output
    else {
        panic!("{}: a buffer output", case.case.name);
    };
    read_buffer(&buffers[&binding], offset, length).unwrap()
}

/// Run `case` over `buffers` and `resources` on `queue`.
pub(super) fn dispatch(
    queue: &CommandQueue,
    case: &PublicCase,
    buffers: &BTreeMap<u32, Buffer>,
    resources: &CaseResources,
) {
    let pipeline = pipeline_state(queue.device(), case).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode_case(&mut encoder, &pipeline, &case.compute, buffers, resources).unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
}

/// 2D `rgba32_float` textures over buffers 4 bytes past a page, whose texels Vulkan copies only
/// at multiples of 16 bytes: a kernel samples one, and another writes one, through a shadow image
/// copied in and back through a staging buffer. Each output is the runtime's run of the case
/// over textures of its own.
#[test]
fn textures_over_bytes_that_start_past_their_vkbuffer_copy_through_staging() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let case = public_case(SAMPLER).unwrap();
        let expected = run_case(device, &case).unwrap().output;
        let under = HostBytes::new(device, &case.compute.textures[&0].bytes, 4);
        let (buffers, resources) = over(device, &case, &[(0, &under.buffer)]);
        dispatch(&queue, &case, &buffers, &resources);
        assert_eq!(buffer_output(&case, &buffers), expected, "{SAMPLER}");
        drop((buffers, resources));
        under.check_around(SAMPLER);

        let case = public_case(TEXTURE_WRITER).unwrap();
        let expected = run_case(device, &case).unwrap().output;
        let under = HostBytes::new(device, &case.compute.textures[&0].bytes, 4);
        let (buffers, resources) = over(device, &case, &[(0, &under.buffer)]);
        dispatch(&queue, &case, &buffers, &resources);
        drop((buffers, resources));
        assert_eq!(under.bytes(), expected, "{TEXTURE_WRITER}");
        under.check_around(TEXTURE_WRITER);
    });
}

/// Texture buffers over buffers 2 and 4 bytes past a page. Vulkan views texels only at
/// multiples of `minTexelBufferOffsetAlignment` of a `VkBuffer` (4 on RADV, 16 on the other
/// venues), so 2 bytes past a page every venue views them through a staging buffer, copied in
/// before the dispatch and back after it, and 4 bytes past RADV views them in place. A kernel
/// writes eight texels through one; another reads four through one and writes four through
/// another, past their ends too. Each output is the runtime's run of the case, and each
/// texture's bytes its run over buffers of its own.
#[test]
fn texture_buffers_over_bytes_that_start_past_their_vkbuffer() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        for at in [2, 4] {
            let case = public_case(TEXEL_WRITER).unwrap();
            let expected = run_case(device, &case).unwrap().output;
            let under = HostBytes::new(device, &case.compute.textures[&1].bytes, at);
            let (buffers, resources) = over(device, &case, &[(1, &under.buffer)]);
            dispatch(&queue, &case, &buffers, &resources);
            drop((buffers, resources));
            assert_eq!(
                under.bytes(),
                expected,
                "{TEXEL_WRITER}, {at} bytes past a page"
            );
            under.check_around(TEXEL_WRITER);

            let case = public_case(TEXEL_READER).unwrap();
            let expected = run_case(device, &case).unwrap().output;
            let literal = |index: u32| &case.compute.textures[&index].bytes;
            let own = buffer_with(device, literal(1)).unwrap();
            let (buffers, resources) = over(device, &case, &[(1, &own)]);
            dispatch(&queue, &case, &buffers, &resources);
            drop((buffers, resources));
            let written = read_buffer(&own, 0, own.length()).unwrap();
            let read = HostBytes::new(device, literal(0), at);
            let write = HostBytes::new(device, literal(1), at);
            let textures = [(0, &read.buffer), (1, &write.buffer)];
            let (buffers, resources) = over(device, &case, &textures);
            dispatch(&queue, &case, &buffers, &resources);
            let output = buffer_output(&case, &buffers);
            drop((buffers, resources));
            assert_eq!(output, expected, "{TEXEL_READER}, {at} bytes past a page");
            assert_eq!(
                write.bytes(),
                written,
                "{TEXEL_READER}, {at} bytes past a page"
            );
            assert_eq!(
                read.bytes(),
                *literal(0),
                "{TEXEL_READER}, {at} bytes past a page"
            );
            read.check_around(TEXEL_READER);
            write.check_around(TEXEL_READER);
        }
    });
}

/// One kernel writes eight `rgba8_uint` texels through a texture buffer 2 bytes past a page,
/// staged on every venue, and another reads the first four as `r32_uint` through a second
/// texture buffer over the same bytes, in one encoder and in two of one command buffer: the
/// write is copied back before the read is copied in. The read's output is the runtime's run of
/// the reader over the written bytes, which its run over the bytes before the write differs
/// from.
#[test]
fn staged_texture_buffers_over_the_same_bytes_see_each_others_writes() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let writer = public_case(TEXEL_WRITER).unwrap();
        let written = run_case(device, &writer).unwrap().output;
        let initial = &writer.compute.textures[&1].bytes;
        let reading = |bytes: &[u8]| {
            let mut case = public_case(TEXEL_READER).unwrap();
            let texels = &mut case.compute.textures.get_mut(&0).unwrap().bytes;
            let n = texels.len();
            texels.copy_from_slice(&bytes[..n]);
            case
        };
        let expected = run_case(device, &reading(&written)).unwrap().output;
        let unwritten = run_case(device, &reading(initial)).unwrap().output;
        assert_ne!(unwritten, expected, "the read tells the write apart");
        let reader = reading(initial);
        let pipelines = [&writer, &reader].map(|case| pipeline_state(device, case).unwrap());
        for one_encoder in [true, false] {
            let under = HostBytes::new(device, initial, 2);
            let (write_buffers, write_resources) = over(device, &writer, &[(1, &under.buffer)]);
            let (read_buffers, read_resources) = over(device, &reader, &[(0, &under.buffer)]);
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            let writes = &writer.compute;
            encode_case(
                &mut encoder,
                &pipelines[0],
                writes,
                &write_buffers,
                &write_resources,
            )
            .unwrap();
            if !one_encoder {
                encoder.end_encoding().unwrap();
                encoder = command_buffer.compute_command_encoder().unwrap();
            }
            let reads = &reader.compute;
            encode_case(
                &mut encoder,
                &pipelines[1],
                reads,
                &read_buffers,
                &read_resources,
            )
            .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            let how = if one_encoder {
                "one encoder"
            } else {
                "two encoders"
            };
            assert_eq!(buffer_output(&reader, &read_buffers), expected, "{how}");
            drop((write_buffers, write_resources, read_buffers, read_resources));
            assert_eq!(under.bytes(), written, "{how}");
            under.check_around(how);
        }
    });
}

/// The probe's pipeline and a depth buffer set on `encoder`, as before each draw.
fn prepare(probe: &Probe, encoder: &mut RenderCommandEncoder, depth: &Buffer) {
    encoder
        .set_render_pipeline_state(&probe.pipeline(WHITE))
        .unwrap();
    encoder.set_vertex_buffer(Some(depth), 0, 0).unwrap();
}

/// Draws over buffers 1 to 4 bytes past a page, each one a Metal probe on an M4 measured or its
/// direct form, with the fixture's fullscreen triangle (vertices 0 to 2; 3 and up land outside the
/// target): the vertex function's depth buffer; 32-bit indices 2 bytes past a page (read through a
/// copy) and 4 past it (in place); arguments 2 bytes past a page (a copy), whose second set draws;
/// and an indexed indirect draw of 16-bit indices 1 byte past a page (a copy, which the index
/// bounds pass reads where the device lacks `robustBufferAccess2`). Each stores the texels Metal
/// stored.
#[test]
fn draws_read_indices_and_arguments_where_their_bytes_lie() {
    let (messages, _cache, probe) = validated();
    let device = &probe.device;
    let (full, empty) = (rows(&FULL), rows(&EMPTY));
    let depth = HostBytes::new(device, &[0.5f32, 0.0].map(f32::to_le_bytes).concat(), 4);
    let indices = u32s(&[5, 6, 7, 0, 1, 2]);
    for at in [2, 4] {
        let index_buffer = HostBytes::new(device, &indices, at);
        let stored = probe.run(PASS, |p, e, _| {
            prepare(p, e, &depth.buffer);
            e.draw_indexed_primitives(
                PrimitiveType::Triangle,
                3,
                IndexType::UInt32,
                &index_buffer.buffer,
                12,
                1,
                0,
                0,
            )
            .unwrap();
        });
        assert_eq!(stored.color, full, "32-bit indices {at} bytes past a page");
        index_buffer.check_around("the index buffer");
    }
    // MTLDrawPrimitivesIndirectArguments: from vertex 3 at offset 0, from vertex 0 at 16.
    let arguments = HostBytes::new(device, &u32s(&[3, 1, 3, 0, 3, 1, 0, 0]), 2);
    for (offset, metal) in [(0, &empty), (16, &full)] {
        let stored = probe.run(PASS, |p, e, _| {
            prepare(p, e, &depth.buffer);
            e.draw_primitives_indirect(PrimitiveType::Triangle, &arguments.buffer, offset)
                .unwrap();
        });
        assert_eq!(
            &stored.color, metal,
            "arguments at {offset}, 2 bytes past a page"
        );
    }
    // MTLDrawIndexedPrimitivesIndirectArguments from index 3, over 16-bit indices.
    let index_buffer = HostBytes::new(device, &u16s(&[5, 6, 7, 0, 1, 2]), 1);
    let indexed = HostBytes::new(device, &u32s(&[3, 1, 3, 0, 0]), 2);
    let mut answer = None;
    let stored = probe.run(PASS, |p, e, _| {
        prepare(p, e, &depth.buffer);
        answer = Some(e.draw_indexed_primitives_indirect(
            PrimitiveType::Triangle,
            IndexType::UInt16,
            &index_buffer.buffer,
            0,
            &indexed.buffer,
            0,
        ));
    });
    assert_eq!(answer, Some(Ok(())), "indexed indirect");
    assert_eq!(stored.color, full, "indexed indirect");
    for host in [&depth, &arguments, &index_buffer, &indexed] {
        host.check_around("a draw's buffer");
    }
    drop((depth, arguments, index_buffer, indexed));
    quiet(probe, &messages);
}

/// Visibility results in buffers 4 bytes past a page (Vulkan copies results only to multiples of
/// 8, so they land in an upload region and are copied on) and 8 past it (in place): probe 1's
/// `Boolean` words, 1 for a covering draw (the device's nonzero value, read by its contract), 0 for
/// one covering nothing, and an offset never set keeps its bytes.
#[test]
fn visibility_results_land_where_their_bytes_lie() {
    use VisibilityResultMode::Boolean;
    let (messages, _cache, probe) = validated();
    for at in [4, 8] {
        let vis = HostBytes::new(&probe.device, &[0xaa; 32], at);
        probe
            .try_run_with(PASS, Some(&vis.buffer), |p, e, h| {
                let depth = p.depth(0.5, 0.0);
                e.set_render_pipeline_state(&p.pipeline(WHITE)).unwrap();
                e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
                e.set_visibility_result_mode(Boolean, 8).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
                e.set_visibility_result_mode(Boolean, 24).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0)
                    .unwrap();
                h.push(depth);
            })
            .unwrap();
        let words: Vec<u64> = vis
            .bytes()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect();
        let u = 0xaaaa_aaaa_aaaa_aaaa;
        let words = super::visibility::by_contract(words, &[1, 3], u);
        assert_eq!(words, [u, 1, u, 0], "results {at} bytes past a page");
        vis.check_around("the visibility buffer");
    }
    quiet(probe, &messages);
}

/// `Accumulate` results in buffers 4 bytes past a page (landed in an upload region, into which each
/// word is copied before the combine takes the results into it) and 8 past it (in place): a
/// `Boolean` 0 under a covering draw becomes nonzero, a 2 under a draw covering nothing stays 2, a
/// `Counting` 100 under a covering draw reads 116 where the device counts, and the words around
/// them keep their bytes.
#[test]
fn accumulated_visibility_results_land_where_their_bytes_lie() {
    use super::visibility::{complete, encode_pass, precise};
    use metal2vulkan_runtime::VisibilityResultType::Accumulate;
    use VisibilityResultMode::{Boolean, Counting};
    let (messages, _cache, probe) = validated();
    let counting = precise(&probe);
    let color = super::render_state::texture(&probe.device, PixelFormat::RGBA8Unorm, 4);
    let u = 0xaaaa_aaaa_aaaa_aaaa;
    for at in [4, 8] {
        let start: Vec<u8> = [u, 0, 2, 100]
            .iter()
            .flat_map(|w: &u64| w.to_le_bytes())
            .collect();
        let vis = HostBytes::new(&probe.device, &start, at);
        let command_buffer = probe.queue.command_buffer().unwrap();
        let mut held = Vec::new();
        encode_pass(
            &probe,
            &command_buffer,
            &color,
            &vis.buffer,
            Accumulate,
            &mut held,
            |p, e, h| {
                let depth = p.depth(0.5, 0.0);
                e.set_render_pipeline_state(&p.pipeline(WHITE)).unwrap();
                e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
                e.set_visibility_result_mode(Boolean, 8).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
                e.set_visibility_result_mode(Boolean, 16).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0)
                    .unwrap();
                if counting {
                    e.set_visibility_result_mode(Counting, 24).unwrap();
                    e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                        .unwrap();
                }
                h.push(depth);
            },
        );
        complete(command_buffer, held);
        let words: Vec<u64> = vis
            .bytes()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect();
        let counted = if counting { 116 } else { 100 };
        assert!(words[1] != 0, "results {at} bytes past a page: {words:?}");
        assert_eq!(
            [words[0], words[2], words[3]],
            [u, 2, counted],
            "results {at} bytes past a page"
        );
        vis.check_around("the visibility buffer");
    }
    quiet(probe, &messages);
}

/// Each degradation these buffers take is announced once on a device, through its message sink,
/// however often the device takes it: two 2D textures over bytes 4 past a page (shadowed), two
/// texture buffers 2 past one (staged on every venue), two indexed draws over indices 2 past one
/// (copied before the pass), and two passes' visibility results 4 past one (landed in an upload
/// region). The notices fail nothing; `degrade_notices` checks each one's form. Beside them the
/// device announces what the steps force: the white pipeline's making's kinds, the cases'
/// `Shared` buffers' system memory, and the passes' `Boolean` results
/// (`visibility::BOOLEANS`).
#[test]
fn host_byte_degradations_are_announced_once_per_device_and_kind() {
    use VisibilityResultMode::Boolean;
    let (messages, _cache, probe) = validated();
    let device = &probe.device;
    let mut held = Vec::new();
    for (name, index, at) in [(SAMPLER, 0, 4), (TEXEL_WRITER, 1, 2)] {
        let case = public_case(name).unwrap();
        for _ in 0..2 {
            let under = HostBytes::new(device, &case.compute.textures[&index].bytes, at);
            // The textures drop before the bytes under them.
            held.push((over(device, &case, &[(index, &under.buffer)]), under));
        }
    }
    let depth = HostBytes::new(device, &[0.5f32, 0.0].map(f32::to_le_bytes).concat(), 4);
    for _ in 0..2 {
        let indices = HostBytes::new(device, &u32s(&[5, 6, 7, 0, 1, 2]), 2);
        probe.run(PASS, |p, e, _| {
            prepare(p, e, &depth.buffer);
            e.draw_indexed_primitives(
                PrimitiveType::Triangle,
                3,
                IndexType::UInt32,
                &indices.buffer,
                12,
                1,
                0,
                0,
            )
            .unwrap();
        });
        let vis = HostBytes::new(device, &[0xaa; 16], 4);
        probe
            .try_run_with(PASS, Some(&vis.buffer), |p, e, _| {
                prepare(p, e, &depth.buffer);
                e.set_visibility_result_mode(Boolean, 8).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            })
            .unwrap();
    }
    drop((held, depth));
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(device),
    )]);
    first.extend(probe.made(WHITE));
    first.extend(super::visibility::BOOLEANS);
    assert_made_once(&messages, &HOST_BYTES, &first);
    quiet(probe, &messages);
}
