//! Device tests of two textures over one IOSurface's memory through two no-copy imports of it, on
//! the device the cfg selects, under the validation layer. This is the consumer's display: the
//! guest's WindowServer writes a 1920x1080 `RGBA16Float` texture of an IOSurface plane, which the
//! runtime makes over a no-copy buffer of the surface's pages
//! (`Buffer::new_texture_with_iosurface_plane`), and the host's display reads the same surface
//! through a second texture over a SEPARATE no-copy buffer of the same pages. Each texture over
//! host pages is shadowed (its imported memory backs no image), so the writer's texels reach the
//! pages by its shadow's copy back, and the reader's leave them by its own copies or its blits.
//!
//! Each case writes a new color through the first texture (a render pass's clear, or a blit
//! from a buffer) and reads the second (a blit into a buffer, after a render pass that loads and
//! stores it, through its shadow, where the case says), placed as a program would place them: in
//! one blit encoder, in one command buffer, in two command buffers of one queue, on two queues
//! with no ordering, on two queues ordered by an event, and on two queues ordered by the host's
//! wait. Every texel the reader copied out is counted fresh (the round's color), stale (the round
//! before's) or other.
//!
//! Metal orders two textures over one IOSurface wherever it orders one resource's accesses: every
//! placement but the unordered queues read fresh texels in each of 16 rounds, plain and under API
//! validation (an M4, macOS 27.0.1: a render pass's, a kernel's and a blit's
//! writes, read by a blit, a kernel and a render pass). The runtime joins two no-copy buffers over
//! the same memory, and lays the second in the first's import (`host_aliases.rs`), so its reads are
//! fresh there too.
//!
//! The second import may come over the pages otherwise (the consumer's boot whose keys 113 and 115
//! lay over one surface's pages unordered): made over other pages and rebound to the surface's
//! ([`Arrival::Rebound`]), or made after the program reported the surface's range remapped while
//! the first lived ([`Arrival::AfterRemap`]). Neither joins the first, as joining keeps to the
//! memory a buffer was made over; the runtime orders their uses by the bytes each lies over now
//! (`host_aliases.rs`'s placements), so their reads are fresh wherever one buffer's would be, two
//! queues in submission order included.
//!
//! The consumer's conversion of a two-plane YCbCr surface reads the second import as a plain
//! buffer: the guest writes the surface through a texture over the first import, and the consumer's
//! kernel reads the planes' bytes through a no-copy buffer of its own over the same pages, with no
//! texture over it ([`Reader::Kernel`]). The same rule orders that pair, as one resource's
//! accesses.

use super::at_plan;
use super::host_memory::on_device_with;
use super::host_memory::{no_copy, Mapping, GUEST_PAGE};
use super::WAIT;
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, ComputePipelineState, Device,
    EventRef, HostMemory, IoSurfacePlane, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};

/// The consumer's display surface: 1920x1080 `RGBA16Float`, 8 bytes a texel, rows packed.
const W: u64 = 1920;
const H: u64 = 1080;
const TEXEL: u64 = 8;
const BPR: u64 = W * TEXEL;
/// The rounds of each case: each writes a new color.
const ROUNDS: u64 = 4;

/// How the first texture is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Writer {
    /// A render pass that clears it and stores: its shadow's copy back.
    Clear,
    /// A blit from a buffer: a write of its bytes.
    Blit,
}

/// How the second texture is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reader {
    /// A blit into a buffer: a read of its bytes.
    Blit,
    /// A render pass that loads it and stores it, then a blit into a buffer: its shadow's copy
    /// in, whose stale bytes the copy back would put in the pages.
    Pass,
    /// A kernel that copies the second import's words, read as a plain buffer with no texture
    /// over it, into a buffer: the consumer's conversion kernel's read.
    Kernel,
}

/// The kernel of [`Reader::Kernel`]: thread `gid` copies word `gid` of buffer 0 into buffer 1.
const COPY_WORDS: &str = r#"; Owned synthetic kernel: one thread a word.
source_filename = "surface_aliases.metal"

define void @copy_words(ptr addrspace(1) noalias %in, ptr addrspace(1) noalias %out, i32 %gid) {
entry:
  %g = zext i32 %gid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %g
  %v = load i32, ptr addrspace(1) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @copy_words, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// The pipeline state of [`COPY_WORDS`] on `device`.
fn copy_words(device: &Device) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(COPY_WORDS).unwrap())
        .unwrap();
    let function = library.new_function_with_name("copy_words").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// Where the write and the read lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// One blit encoder (a blit writer only).
    Encoder,
    /// Two encoders of one command buffer.
    CommandBuffer,
    /// Two command buffers of one queue, committed one after the other.
    Queue,
    /// Two queues, with nothing ordering them.
    Queues,
    /// Two queues, the read waiting for an event the write signals.
    Event,
    /// Two queues, the read committed once the write completed.
    Host,
}

/// The color of round `k`: halves, exact in `RGBA16Float`.
fn color(k: u64) -> [f32; 4] {
    [k as f32 / 64.0, 1.0 - k as f32 / 64.0, 0.5, 1.0]
}

/// The bits of `x`, a half (an exact one: a multiple of 2^-10 in [2^-14, 2), or 0).
pub(super) fn half_bits(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3ff;
    assert!(
        (1..31).contains(&exponent) && bits & 0x1fff == 0,
        "{x} is no exact half"
    );
    ((exponent as u16) << 10) | mantissa as u16
}

/// Round `k`'s texel, as the bytes `RGBA16Float` holds.
fn texel(k: u64) -> [u8; 8] {
    let mut out = [0; 8];
    for (i, channel) in color(k).into_iter().enumerate() {
        out[2 * i..2 * i + 2].copy_from_slice(&half_bits(channel).to_le_bytes());
    }
    out
}

/// How the second import comes over the surface's pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arrival {
    /// Made over them while the first lives: it joins the first.
    Made,
    /// Made over other pages, then rebound to the surface's: it joins nothing.
    Rebound,
    /// Made over them after the program reported their range remapped while the first lived
    /// (with no remap made): it joins nothing, and imports them anew.
    AfterRemap,
}

/// One surface's memory, two imports of it, and a texture of its plane over the first, and over
/// the second where the reader reads one.
struct Surface {
    writer: Texture,
    reader: Option<Texture>,
    imports: [Buffer; 2],
    /// Dropped last: the imports are over it.
    memory: Mapping,
    /// The pages a rebound second import was made over: dropped after it.
    _left: Option<Mapping>,
}

impl Surface {
    /// The surface: its second buffer lies in the first's import where it was made over the pages,
    /// else comes over them as `arrival` says; a texture over the second where `reader_texture`
    /// says.
    fn new(device: &Device, reader_texture: bool, arrival: Arrival) -> Surface {
        let len = (BPR * H).next_multiple_of(GUEST_PAGE as u64) as usize;
        let memory = Mapping::anonymous(len);
        let import = |pointer| {
            // SAFETY: the mappings outlive both buffers (`Surface`'s field order).
            unsafe { no_copy(device, pointer, len, ResourceOptions::default(), None) }
        };
        let first = import(memory.pointer()).unwrap();
        let mut left = None;
        let second = match arrival {
            Arrival::Made => import(memory.pointer()).unwrap(),
            Arrival::Rebound => {
                let other = Mapping::anonymous(len);
                let second = import(other.pointer()).unwrap();
                // SAFETY: the surface's mapping outlives the buffer, and the pages it leaves stay
                // mapped until the surface goes; no command uses the buffer yet.
                unsafe {
                    second.rebind_bytes_no_copy(HostMemory::new(memory.pointer()), None, None)
                }
                .unwrap();
                left = Some(other);
                second
            }
            Arrival::AfterRemap => {
                device.host_range_remapped(memory.pointer(), len as u64);
                import(memory.pointer()).unwrap()
            }
        };
        let plane = IoSurfacePlane::new(0, BPR, false);
        let texture = |buffer: &Buffer| {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, W, H, false);
            d.usage = TextureUsage::RENDER_TARGET
                | TextureUsage::SHADER_READ
                | TextureUsage::SHADER_WRITE;
            d.resource_options = buffer.resource_options();
            buffer.new_texture_with_iosurface_plane(&d, plane).unwrap()
        };
        Surface {
            writer: texture(&first),
            reader: reader_texture.then(|| texture(&second)),
            imports: [first, second],
            memory,
            _left: left,
        }
    }

    /// The texture over the second import.
    fn reader(&self) -> &Texture {
        self.reader
            .as_ref()
            .expect("the case's reader reads a texture")
    }
}

/// The buffers a case copies through: a writer blit's source, and the reader's destination; and
/// the kernel of a reader that reads by one.
struct Copies {
    source: Buffer,
    out: Buffer,
    kernel: Option<ComputePipelineState>,
}

impl Copies {
    fn new(device: &Device, reader: Reader) -> Copies {
        let new = || {
            device
                .new_buffer(BPR * H, ResourceOptions::shared())
                .unwrap()
        };
        Copies {
            source: new(),
            out: new(),
            kernel: (reader == Reader::Kernel).then(|| copy_words(device)),
        }
    }

    /// Fill the source with round `k`'s texels, and the destination with bytes no round writes.
    fn prepare(&self, k: u64) {
        let texel = texel(k);
        let source = self.source.contents().unwrap();
        let out = self.out.contents().unwrap();
        let n = (BPR * H) as usize;
        // SAFETY: both are Shared buffers of `n` bytes no command uses now (each round waits for
        // its command buffers).
        unsafe {
            let source = std::slice::from_raw_parts_mut(source.as_ptr(), n);
            for chunk in source.as_chunks_mut::<{ TEXEL as usize }>().0.iter_mut() {
                chunk.copy_from_slice(&texel);
            }
            std::ptr::write_bytes(out.as_ptr(), 0xff, n);
        }
    }

    /// The texels the reader copied out: fresh (round `k`'s), stale (round `k - 1`'s), other.
    fn counted(&self, k: u64) -> [u64; 3] {
        let (fresh, stale) = (texel(k), texel(k - 1));
        let out = self.out.contents().unwrap();
        // SAFETY: as in `prepare`.
        let bytes = unsafe { std::slice::from_raw_parts(out.as_ptr(), (BPR * H) as usize) };
        let mut counts = [0; 3];
        for chunk in bytes.as_chunks::<{ TEXEL as usize }>().0.iter() {
            let at = if *chunk == fresh {
                0
            } else if *chunk == stale {
                1
            } else {
                2
            };
            counts[at] += 1;
        }
        counts
    }
}

fn whole() -> (Origin, Size) {
    (Origin::new(0, 0, 0), Size::new(W, H, 1))
}

/// Encode the write of round `k` through `surface.writer` into `command_buffer`.
fn write(
    command_buffer: &CommandBuffer,
    writer: Writer,
    surface: &Surface,
    copies: &Copies,
    k: u64,
) {
    match writer {
        Writer::Clear => {
            let [r, g, b, a] = color(k).map(f64::from);
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(surface.writer.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.store_action = StoreAction::Store;
            attachment.clear_color = ClearColor::new(r, g, b, a);
            command_buffer
                .render_command_encoder(&pass)
                .unwrap()
                .end_encoding()
                .unwrap();
        }
        Writer::Blit => {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit_in(&mut blit, surface, copies);
            blit.end_encoding().unwrap();
        }
    }
}

fn blit_in(
    blit: &mut metal2vulkan_runtime::BlitCommandEncoder,
    surface: &Surface,
    copies: &Copies,
) {
    let (origin, size) = whole();
    blit.copy_from_buffer_to_texture(
        &copies.source,
        0,
        BPR,
        BPR * H,
        size,
        &surface.writer,
        0,
        0,
        origin,
        BlitOption::None,
    )
    .unwrap();
}

fn blit_out(
    blit: &mut metal2vulkan_runtime::BlitCommandEncoder,
    surface: &Surface,
    copies: &Copies,
) {
    let (origin, size) = whole();
    blit.copy_from_texture_to_buffer(
        surface.reader(),
        0,
        0,
        origin,
        size,
        &copies.out,
        0,
        BPR,
        BPR * H,
        BlitOption::None,
    )
    .unwrap();
}

/// Encode the read through the second import into `command_buffer`: through its texture, or by
/// a kernel reading it as a buffer.
fn read(command_buffer: &CommandBuffer, reader: Reader, surface: &Surface, copies: &Copies) {
    if let Some(kernel) = &copies.kernel {
        let words = BPR * H / 4;
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(kernel).unwrap();
        encoder.set_buffer(Some(&surface.imports[1]), 0, 0).unwrap();
        encoder.set_buffer(Some(&copies.out), 0, 1).unwrap();
        encoder
            .dispatch_threads(Size::new(words, 1, 1), Size::new(64, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        return;
    }
    if reader == Reader::Pass {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(surface.reader().clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        command_buffer
            .render_command_encoder(&pass)
            .unwrap()
            .end_encoding()
            .unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit_out(&mut blit, surface, copies);
    blit.end_encoding().unwrap();
}

fn finish(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Run round `k` of `writer` placed as `place`, read by `reader`, on `queues`.
fn round(
    device: &Device,
    queues: &[CommandQueue; 2],
    (writer, place, reader): Case,
    surface: &Surface,
    copies: &Copies,
    k: u64,
) {
    let first = queues[0].command_buffer().unwrap();
    match place {
        Place::Encoder => {
            assert_eq!(
                (writer, reader),
                (Writer::Blit, Reader::Blit),
                "one encoder writes and reads by blits"
            );
            let mut blit = first.blit_command_encoder().unwrap();
            blit_in(&mut blit, surface, copies);
            blit_out(&mut blit, surface, copies);
            blit.end_encoding().unwrap();
            first.commit().unwrap();
            finish(&first);
        }
        Place::CommandBuffer => {
            write(&first, writer, surface, copies, k);
            read(&first, reader, surface, copies);
            first.commit().unwrap();
            finish(&first);
        }
        Place::Queue | Place::Queues => {
            let on = if place == Place::Queue { 0 } else { 1 };
            let second = queues[on].command_buffer().unwrap();
            write(&first, writer, surface, copies, k);
            read(&second, reader, surface, copies);
            first.commit().unwrap();
            second.commit().unwrap();
            finish(&second);
            finish(&first);
        }
        Place::Event => {
            let event = device.new_event().unwrap();
            let second = queues[1].command_buffer().unwrap();
            write(&first, writer, surface, copies, k);
            first
                .encode_signal_event(EventRef::from(&event), 1)
                .unwrap();
            second
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            read(&second, reader, surface, copies);
            first.commit().unwrap();
            second.commit().unwrap();
            finish(&second);
            finish(&first);
        }
        Place::Host => {
            write(&first, writer, surface, copies, k);
            first.commit().unwrap();
            finish(&first);
            let second = queues[1].command_buffer().unwrap();
            read(&second, reader, surface, copies);
            second.commit().unwrap();
            finish(&second);
        }
    }
}

/// A case: the writer, where the write and the read lie, and the reader.
type Case = (Writer, Place, Reader);

/// The rounds of `case` whose read saw only fresh texels, only stale ones, or a mix; each
/// round's write is in the surface's pages once the round completed.
fn rounds(device: &Device, case: Case) -> [u64; 3] {
    rounds_after(device, case, Arrival::Made)
}

/// [`rounds`] with the second import come over the pages as `arrival` says.
fn rounds_after(device: &Device, case: Case, arrival: Arrival) -> [u64; 3] {
    let (writer, place, reader) = case;
    let surface = Surface::new(device, reader != Reader::Kernel, arrival);
    let copies = Copies::new(device, reader);
    let queues = [
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    ];
    // Round 0: the surface's pages hold color 0.
    let zero = texel(0);
    for at in (0..BPR * H).step_by(TEXEL as usize) {
        surface.memory.write(at as usize, &zero);
    }
    let mut seen = [0; 3];
    for k in 1..=ROUNDS {
        copies.prepare(k);
        round(device, &queues, case, &surface, &copies, k);
        let [fresh, stale, other] = copies.counted(k);
        let at = if fresh == W * H {
            0
        } else if stale == W * H {
            1
        } else {
            2
        };
        seen[at] += 1;
        eprintln!(
            "surface aliases {arrival:?} {writer:?} {place:?} {reader:?} round {k}: fresh \
             {fresh}, stale {stale}, other {other}"
        );
        let last = (BPR * H - TEXEL) as usize;
        for at in [0, last / 2 / TEXEL as usize * TEXEL as usize, last] {
            assert_eq!(
                surface.memory.read(at..at + TEXEL as usize),
                texel(k),
                "{case:?} round {k}: the write is not in the pages at byte {at}"
            );
        }
    }
    eprintln!(
        "surface aliases {} {arrival:?} {case:?}: fresh {}, stale {}, mixed {} of {ROUNDS}",
        device.name(),
        seen[0],
        seen[1],
        seen[2]
    );
    seen
}

/// A write through one texture over a surface's pages, then a read through a second texture over
/// a separate import of the same pages, every texel the reader copies out fresh wherever Metal
/// orders the two (the module docs): in one blit encoder, in one command buffer, in two command
/// buffers of one queue, on two queues ordered by an event or by the host's wait. Two queues with
/// nothing ordering them are counted and reported.
#[test]
fn a_second_import_of_a_surface_reads_what_the_first_wrote() {
    on_device_with(at_plan, |device| {
        let cases = [
            (Writer::Blit, Place::Encoder, Reader::Blit),
            (Writer::Clear, Place::CommandBuffer, Reader::Blit),
            (Writer::Blit, Place::CommandBuffer, Reader::Blit),
            (Writer::Clear, Place::CommandBuffer, Reader::Pass),
            (Writer::Blit, Place::CommandBuffer, Reader::Pass),
            (Writer::Clear, Place::Queue, Reader::Blit),
            (Writer::Blit, Place::Queue, Reader::Blit),
            (Writer::Clear, Place::Queue, Reader::Pass),
            (Writer::Blit, Place::Queue, Reader::Pass),
            (Writer::Clear, Place::Event, Reader::Blit),
            (Writer::Clear, Place::Event, Reader::Pass),
            (Writer::Clear, Place::Host, Reader::Blit),
            (Writer::Clear, Place::Queues, Reader::Blit),
        ];
        let mut ordered_but_not_fresh = Vec::new();
        for case in cases {
            let seen = rounds(device, case);
            if case.1 != Place::Queues && seen[0] != ROUNDS {
                ordered_but_not_fresh.push((case, seen));
            }
        }
        assert!(
            ordered_but_not_fresh.is_empty(),
            "a read through the second import missed the first's write: {ordered_but_not_fresh:?}"
        );
    });
}

/// A write through a texture over a surface's pages, then a kernel's read of a second import of
/// the same pages as a plain buffer, with no texture over it (the consumer's conversion of a
/// two-plane surface): every word the kernel copies out fresh wherever Metal orders one
/// resource's accesses, in one command buffer, in two command buffers of one queue, and on two
/// queues ordered by an event or by the host's wait. Two queues with nothing ordering them are
/// counted and reported.
#[test]
fn a_kernel_reads_what_a_texture_over_another_import_wrote() {
    on_device_with(at_plan, |device| {
        let cases = [
            (Writer::Clear, Place::CommandBuffer, Reader::Kernel),
            (Writer::Blit, Place::CommandBuffer, Reader::Kernel),
            (Writer::Clear, Place::Queue, Reader::Kernel),
            (Writer::Blit, Place::Queue, Reader::Kernel),
            (Writer::Clear, Place::Event, Reader::Kernel),
            (Writer::Blit, Place::Event, Reader::Kernel),
            (Writer::Clear, Place::Host, Reader::Kernel),
            (Writer::Clear, Place::Queues, Reader::Kernel),
        ];
        let mut ordered_but_not_fresh = Vec::new();
        for case in cases {
            let seen = rounds(device, case);
            if case.1 != Place::Queues && seen[0] != ROUNDS {
                ordered_but_not_fresh.push((case, seen));
            }
        }
        assert!(
            ordered_but_not_fresh.is_empty(),
            "a kernel's read of the second import missed the texture's write: \
             {ordered_but_not_fresh:?}"
        );
    });
}

/// A render through a texture over one import of a surface's pages, then a read through a
/// texture over a second buffer over them that joined nothing (rebound there, or made after the
/// range was reported remapped): every texel the reader copies out fresh in every round, in one
/// command buffer, in two of one queue, on two queues ordered by an event or by the host's wait,
/// and on two queues in submission order with nothing else ordering them, as the runtime orders
/// one resource's accesses there (the consumer's render of one key and sample of the other in
/// the next command buffer on another queue). Two imports made over the pages, which join, are
/// the control.
#[test]
fn a_second_buffer_that_joined_nothing_reads_what_the_first_wrote() {
    on_device_with(at_plan, |device| {
        let places = [
            Place::CommandBuffer,
            Place::Queue,
            Place::Event,
            Place::Host,
            Place::Queues,
        ];
        let mut not_fresh = Vec::new();
        for arrival in [Arrival::Made, Arrival::Rebound, Arrival::AfterRemap] {
            for place in places {
                for reader in [Reader::Blit, Reader::Pass, Reader::Kernel] {
                    let case = (Writer::Clear, place, reader);
                    let seen = rounds_after(device, case, arrival);
                    if seen[0] != ROUNDS {
                        not_fresh.push((arrival, case, seen));
                    }
                }
            }
        }
        assert!(
            not_fresh.is_empty(),
            "a read through the second buffer missed the first's write: {not_fresh:?}"
        );
    });
}
