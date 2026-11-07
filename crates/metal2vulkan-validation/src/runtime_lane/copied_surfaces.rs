//! Device tests of no-copy buffers over one IOSurface's pages whose imports the caller's gate
//! refuses (`HostMemory::with_import_gate`), so the pages are a copied host buffer's
//! (`copied-host-pages`, the runtime's `copied.rs`), on the device the cfg selects, under the
//! validation layer. This is the consumer's display past its gate: it makes a new no-copy buffer
//! for every texture of a surface's plane at every present (`Buffer::new_texture_with_iosurface_plane`),
//! each over the surface's pages, while the buffer of the surface's first texture lives.
//!
//! Metal orders two textures over one IOSurface as it orders one resource's accesses, in one
//! command buffer and across queues an event orders: on the M3 Max under macOS 27.0.1 the
//! consumer's probe of this shape (two textures over one 64x64 `BGRA8Unorm` surface, 256-byte
//! rows, each over its own no-copy buffer of the pages) read the written texels in both
//! [`Place::CommandBuffer`] and [`Place::EventReadFirst`], as `surface_aliases`'s M4 probe reads
//! them in every ordered placement. That measurement was made on Metal by the consumer; the
//! validation crate's Metal side runs authored kernel cases, not this host-memory shape, and Metal
//! has no import gate. So each case here holds the runtime to it: every texel fresh.
//!
//! The runtime lays a buffer made over exactly the bytes a live copied host buffer was made over in
//! that copy, as it lays one in a live import (`host_aliases.rs`): one `VkBuffer`, one snapshot
//! and one hazard key for the memory, the gate not asked, counted
//! (`WorkCounts::copied_host_buffers_shared`). Two copies instead agree only through the pages, a
//! merge back at the writer's completion and a copy in at the reader's commit, so a reader
//! committed before the writer completed read the pages' bytes from before the write: every texel
//! stale in [`Place::CommandBuffer`], [`Place::EventReadFirst`] and [`Place::QueueInFlight`]
//! before the runtime joined copies (M3 Max, MoltenVK).
//!
//! Each "apart" case keeps a copy or an import of its own, for the reason its test gives.

use super::host_memory::{host_page, no_copy, on_device, Mapping};
use super::WAIT;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBuffer, CommandQueue,
    CpuCacheMode, Device, EventRef, HazardTrackingMode, HostMemory, ImportGate, IoSurfacePlane,
    Origin, ResourceOptions, Size, StorageMode, Texture, TextureDescriptor, WorkCounts,
};
use std::ptr::NonNull;
use std::sync::Arc;

/// The consumer's probe surface: 64x64 `BGRA8Unorm`, 256-byte rows.
const W: u64 = 64;
const H: u64 = 64;
const TEXEL: u64 = 4;
const BPR: u64 = W * TEXEL;
const LEN: u64 = BPR * H;
/// The rounds of each case: each writes a new color.
const ROUNDS: u8 = 4;

/// The gate of the consumer past its budget: it refuses every import.
fn refusing() -> ImportGate {
    Arc::new(|_| false)
}

/// A no-copy buffer of `options` over the surface's [`LEN`] bytes at `pointer`, made as the
/// consumer makes one (`new_buffer_with_descriptor`), its import refused by the gate.
///
/// # Safety
///
/// [`HostMemory::new`]'s contract for [`LEN`] bytes.
unsafe fn refused(device: &Device, pointer: NonNull<u8>, options: ResourceOptions) -> Buffer {
    // SAFETY: the caller's contract.
    let memory = unsafe { HostMemory::new(pointer) }.with_import_gate(refusing());
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: None,
    };
    device
        .new_buffer_with_descriptor(BufferDescriptor::new(LEN, options, contents))
        .unwrap()
}

/// The texture of the surface's plane over `buffer`.
fn plane_texture(buffer: &Buffer) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, W, H, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = buffer.resource_options();
    buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, BPR, false))
        .unwrap()
}

/// Round `k`'s texel.
fn texel(k: u8) -> [u8; 4] {
    [k.wrapping_mul(37) | 1, 0xff - k, 0x40 + k, 0xff]
}

/// How many of `bytes`' texels are `texel`.
fn count(bytes: &[u8], texel: [u8; 4]) -> u64 {
    let texels = bytes.as_chunks::<{ TEXEL as usize }>().0;
    texels.iter().filter(|chunk| **chunk == texel).count() as u64
}

/// The bytes of a Shared buffer of the runtime's own.
fn bytes_of(buffer: &Buffer) -> &[u8] {
    let contents = buffer.contents().unwrap();
    // SAFETY: a Shared buffer of `length` bytes, which no command uses while it is read (each
    // round waits for its command buffers).
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }
}

/// Write `bytes` over `buffer`, a Shared buffer of the runtime's own of their length.
fn fill(buffer: &Buffer, bytes: &[u8]) {
    let contents = buffer.contents().unwrap();
    assert_eq!(bytes.len() as u64, buffer.length());
    // SAFETY: as in `bytes_of`.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

/// A round's source (round `k`'s texels) and destination (bytes no round writes).
struct Copies {
    source: Buffer,
    out: Buffer,
}

impl Copies {
    fn new(device: &Device) -> Copies {
        let new = || device.new_buffer(LEN, ResourceOptions::shared()).unwrap();
        Copies {
            source: new(),
            out: new(),
        }
    }

    fn prepare(&self, k: u8) {
        fill(&self.source, &texel(k).repeat((W * H) as usize));
        fill(&self.out, &vec![0xee; LEN as usize]);
    }
}

fn whole() -> (Origin, Size) {
    (Origin::new(0, 0, 0), Size::new(W, H, 1))
}

/// Encode, in an encoder of its own, the blit of `copies.source` into `texture`.
fn write(command_buffer: &CommandBuffer, texture: &Texture, copies: &Copies) {
    let (origin, size) = whole();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_buffer_to_texture(
        &copies.source,
        0,
        BPR,
        LEN,
        size,
        texture,
        0,
        0,
        origin,
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
}

/// Encode, in an encoder of its own, the blit of `texture` into `copies.out`.
fn read(command_buffer: &CommandBuffer, texture: &Texture, copies: &Copies) {
    let (origin, size) = whole();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        origin,
        size,
        &copies.out,
        0,
        BPR,
        LEN,
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
}

/// Encode the blit of `buffer`'s bytes into `copies.out`.
fn read_bytes(command_buffer: &CommandBuffer, buffer: &Buffer, copies: &Copies) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_buffer_to_buffer(buffer, 0, &copies.out, 0, LEN)).unwrap();
    blit.end_encoding().unwrap();
}

fn finish(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Where the write through the first texture and the read through the second lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// The consumer's first case: two blit encoders of one command buffer.
    CommandBuffer,
    /// The consumer's second case: the read's command buffer on a second queue waits for an event
    /// and is committed first; the write's, on the first queue, signals it and is committed
    /// second.
    EventReadFirst,
    /// Two command buffers of one queue, both committed before either completes.
    QueueInFlight,
    /// Two command buffers of one queue, the read committed once the write completed: ordered
    /// through the pages, the control.
    Host,
}

/// Run round `k`, placed as `place`, writing through `writer` and reading through `reader`.
fn round(
    device: &Device,
    queues: &[CommandQueue; 2],
    place: Place,
    (writer, reader): (&Texture, &Texture),
    copies: &Copies,
) {
    let first = queues[0].command_buffer().unwrap();
    match place {
        Place::CommandBuffer => {
            write(&first, writer, copies);
            read(&first, reader, copies);
            first.commit().unwrap();
            finish(&first);
        }
        Place::EventReadFirst => {
            let event = device.new_event().unwrap();
            let reading = queues[1].command_buffer().unwrap();
            reading
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            read(&reading, reader, copies);
            reading.commit().unwrap();
            write(&first, writer, copies);
            first
                .encode_signal_event(EventRef::from(&event), 1)
                .unwrap();
            first.commit().unwrap();
            finish(&reading);
            finish(&first);
        }
        Place::QueueInFlight => {
            let second = queues[0].command_buffer().unwrap();
            write(&first, writer, copies);
            read(&second, reader, copies);
            first.commit().unwrap();
            second.commit().unwrap();
            finish(&second);
            finish(&first);
        }
        Place::Host => {
            write(&first, writer, copies);
            first.commit().unwrap();
            finish(&first);
            let second = queues[0].command_buffer().unwrap();
            read(&second, reader, copies);
            second.commit().unwrap();
            finish(&second);
        }
    }
}

/// One surface's pages, and a texture over each of two no-copy buffers made over them (each
/// texture holds its buffer).
struct Surface {
    textures: [Texture; 2],
    /// Dropped last: the buffers are over it.
    memory: Mapping,
}

/// The counts of a surface's making: copies made (and of those, refused by the gate), buffers
/// that lay in a live copy, and imports the gate refused.
fn copy_counts(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.copied_host_buffers_made - before.copied_host_buffers_made,
        after.copied_host_buffers_refused - before.copied_host_buffers_refused,
        after.copied_host_buffers_shared - before.copied_host_buffers_shared,
        after.host_imports_refused_by_host - before.host_imports_refused_by_host,
    ]
}

impl Surface {
    /// The surface, its two buffers made over its pages behind the refusing gate, with the
    /// counts of their making ([`copy_counts`]).
    fn new(device: &Device) -> (Surface, [u64; 4]) {
        let memory = Mapping::anonymous(LEN as usize);
        let before = device.work_counts();
        // SAFETY: the mapping outlives both buffers (`Surface`'s field order).
        let buffers = [(); 2]
            .map(|_| unsafe { refused(device, memory.pointer(), ResourceOptions::default()) });
        let counts = copy_counts(&before, &device.work_counts());
        let textures = [plane_texture(&buffers[0]), plane_texture(&buffers[1])];
        let surface = Surface { textures, memory };
        (surface, counts)
    }
}

/// The rounds of `place` whose read saw only fresh texels, and the fewest fresh texels a round
/// read; each round's write is in the surface's pages once it completed.
fn rounds(device: &Device, place: Place, surface: &Surface) -> (u8, u64) {
    let copies = Copies::new(device);
    let queues = [
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    ];
    surface.memory.write(0, &texel(0).repeat((W * H) as usize));
    let (mut fresh_rounds, mut fewest) = (0, W * H);
    for k in 1..=ROUNDS {
        copies.prepare(k);
        let [writer, reader] = &surface.textures;
        round(device, &queues, place, (writer, reader), &copies);
        let out = bytes_of(&copies.out);
        let (fresh, stale) = (count(out, texel(k)), count(out, texel(k - 1)));
        eprintln!("copied surfaces {place:?} round {k}: fresh {fresh}, stale {stale}");
        fresh_rounds += u8::from(fresh == W * H);
        fewest = fewest.min(fresh);
        assert_eq!(
            count(&surface.memory.read(0..LEN as usize), texel(k)),
            W * H,
            "{place:?} round {k}: the write is not in the pages"
        );
    }
    (fresh_rounds, fewest)
}

/// Two textures over one surface's pages, each over its own no-copy buffer whose import the gate
/// refuses (the consumer's probe, the module docs): the second buffer lies in the first's copy
/// (one copy made, one buffer shared, the gate asked once), and a read through the second texture
/// sees a write through the first wherever Metal orders them: in one command buffer, on two queues
/// an event orders with the read committed first, in two command buffers of one queue in flight
/// together, and after the host's wait.
#[test]
fn two_buffers_over_one_refused_surface_are_one_copy() {
    on_device(|device| {
        let (mut missed, mut miscounted) = (Vec::new(), Vec::new());
        for place in [
            Place::CommandBuffer,
            Place::EventReadFirst,
            Place::QueueInFlight,
            Place::Host,
        ] {
            let (surface, counts) = Surface::new(device);
            let (fresh, fewest) = rounds(device, place, &surface);
            eprintln!(
                "copied surfaces {}: {place:?} fresh {fresh} of {ROUNDS}, fewest fresh texels \
                 {fewest} of {}; copies made, refused, shared, gate refusals {counts:?}",
                device.name(),
                W * H
            );
            if fresh != ROUNDS {
                missed.push((place, fresh, fewest));
            }
            if counts != [1, 1, 1, 1] {
                miscounted.push((place, counts));
            }
        }
        assert!(
            missed.is_empty() && miscounted.is_empty(),
            "a read through the second buffer missed the first's write (place, fresh rounds, \
             fewest fresh texels): {missed:?}; surfaces whose two buffers were not one copy \
             (place, [copies made, refused, shared, gate refusals]): {miscounted:?}"
        );
    });
}

/// A `Shared` and a `Managed` buffer over one refused surface's pages share one copy: the copy's
/// memory is the runtime's own of one class whatever the buffer's storage mode, so the managed
/// buffer takes the copy a copy of its own would be. Its managed contract holds over it: the CPU's
/// writes through its contents, told by `did_modify_range`, reach a read through the shared
/// buffer, and a GPU write through the shared buffer reaches its contents after a
/// `synchronize_resource` of it; in one command buffer a write through either reaches a read
/// through the other.
#[test]
fn a_shared_and_a_managed_buffer_over_one_refused_surface_share_its_copy() {
    on_device(|device| {
        let memory = Mapping::anonymous(LEN as usize);
        let before = device.work_counts();
        // SAFETY: the mapping outlives both buffers, dropped first.
        let shared = unsafe { refused(device, memory.pointer(), ResourceOptions::shared()) };
        let options = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        // SAFETY: as above.
        let managed = unsafe { refused(device, memory.pointer(), options) };
        let counts = copy_counts(&before, &device.work_counts());
        assert_eq!(
            counts,
            [1, 1, 1, 1],
            "copies made, refused, shared, gate refusals"
        );
        let queue = device.new_command_queue().unwrap();
        let copies = Copies::new(device);
        // The CPU writes through the managed buffer's contents, and says so.
        let cpu = texel(1).repeat((W * H) as usize);
        assert_eq!(managed.contents(), Some(memory.pointer()));
        memory.write(0, &cpu);
        managed.did_modify_range(0..LEN).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        read_bytes(&command_buffer, &shared, &copies);
        command_buffer.commit().unwrap();
        finish(&command_buffer);
        assert_eq!(
            bytes_of(&copies.out),
            &cpu[..],
            "the GPU reads the CPU's write"
        );
        // A GPU write through the shared buffer, then a synchronize of the managed one.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&shared, 0..LEN, 0x5a).unwrap();
        blit.synchronize_resource((&managed).into()).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        finish(&command_buffer);
        assert_eq!(
            memory.read(0..LEN as usize),
            vec![0x5a; LEN as usize],
            "the managed buffer's contents hold the GPU's write"
        );
        // In one command buffer, a write through either reaches a read through the other.
        for (written, value) in [(&managed, 0x33), (&shared, 0x44)] {
            let reader = if std::ptr::eq(written, &managed) {
                &shared
            } else {
                &managed
            };
            copies.prepare(0);
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(written, 0..LEN, value).unwrap();
            blit.end_encoding().unwrap();
            read_bytes(&command_buffer, reader, &copies);
            command_buffer.commit().unwrap();
            finish(&command_buffer);
            assert_eq!(
                bytes_of(&copies.out),
                &vec![value; LEN as usize][..],
                "a read through one buffer sees the other's write in one command buffer"
            );
        }
        drop((shared, managed));
    });
}

/// Whether the runtime imports the pages of no-copy memory made with Metal's private option bit 24
/// through a read-only view of them here (`degrade_notices`'s `VIEWS_DROP_WRITES`): on macOS. Else
/// such memory is a copy of its own whose GPU writes are dropped.
const VIEWS_DROP_WRITES: bool = cfg!(target_os = "macos");

/// A buffer made with Metal's private option bit 24 over a refused surface's pages stays apart from
/// the surface's copy: Metal drops the GPU's writes to that memory, and a copy shared with the
/// surface's buffer would take its writes and merge them back. On macOS the runtime imports a
/// read-only view of the pages for it, making no copy; elsewhere it makes a copy of its own that
/// drops its writes, sharing none. The gate is not asked either way. Its fill is dropped, while it
/// reads the surface buffer's write once that completed.
#[test]
fn a_buffer_whose_writes_are_dropped_stays_apart_from_a_copy() {
    on_device(|device| {
        let memory = Mapping::anonymous(LEN as usize);
        // SAFETY: the mapping outlives both buffers, dropped first.
        let surface = unsafe { refused(device, memory.pointer(), ResourceOptions::default()) };
        let bit = ResourceOptions::from_raw(
            ResourceOptions::default().raw() | ResourceOptions::PRIVATE_BIT,
        )
        .unwrap();
        let before = device.work_counts();
        // SAFETY: as above.
        let dropping = unsafe { refused(device, memory.pointer(), bit) };
        let counts = copy_counts(&before, &device.work_counts());
        let expected = match VIEWS_DROP_WRITES {
            true => [0, 0, 0, 0],
            false => [1, 0, 0, 0],
        };
        assert_eq!(
            counts, expected,
            "copies made, refused, shared, gate refusals"
        );
        let queue = device.new_command_queue().unwrap();
        let copies = Copies::new(device);
        // Its fill is dropped where its writes are: the pages keep the CPU's bytes.
        memory.fill(0x11);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&dropping, 0..LEN, 0x5a).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        finish(&command_buffer);
        assert_eq!(memory.read(0..LEN as usize), vec![0x11; LEN as usize]);
        // The surface buffer's write, once completed, is what it reads.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&surface, 0..LEN, 0x77).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        finish(&command_buffer);
        let command_buffer = queue.command_buffer().unwrap();
        read_bytes(&command_buffer, &dropping, &copies);
        command_buffer.commit().unwrap();
        finish(&command_buffer);
        assert_eq!(bytes_of(&copies.out), &vec![0x77; LEN as usize][..]);
        drop((dropping, surface));
    });
}

/// Rebinds onto a refused surface's pages. A buffer with no texture over it, rebound there, lies in
/// the surface's copy (one copy shared, none made, the gate not asked), and a read through it sees
/// a write through the surface's texture in one command buffer. A buffer whose texture was planned
/// at another offset in its page than the copy lays the bytes at (the copy lays them from its first
/// byte) stays apart: its texture's texel views stay where they were planned, in a copy of its own
/// laid out at the old offset (`WorkCounts::copied_rebind_pages_moved`), which agrees with the
/// surface's through the pages, so it reads the surface's write once that completed.
#[test]
fn a_rebind_onto_a_refused_surface_lies_in_its_copy_unless_its_texture_moved() {
    on_device(|device| {
        let memory = Mapping::anonymous(LEN as usize);
        // SAFETY: the mapping outlives every buffer, dropped first.
        let surface = unsafe { refused(device, memory.pointer(), ResourceOptions::default()) };
        let writer = plane_texture(&surface);
        let queue = device.new_command_queue().unwrap();
        let copies = Copies::new(device);
        let others = [
            Mapping::anonymous(LEN as usize),
            Mapping::anonymous(2 * LEN as usize),
        ];
        let rebind = |buffer: &Buffer| {
            // SAFETY: the surface's mapping outlives the buffer, and the pages it leaves stay
            // mapped until `others` goes; no command uses the buffer now.
            unsafe {
                let memory = HostMemory::new(memory.pointer()).with_import_gate(refusing());
                buffer.rebind_bytes_no_copy(memory, None, None)
            }
            .unwrap();
        };
        // A plain buffer, imported over other pages, then rebound onto the surface's.
        // SAFETY: the other mapping outlives the buffer.
        let plain = unsafe {
            no_copy(
                device,
                others[0].pointer(),
                LEN as usize,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let before = device.work_counts();
        rebind(&plain);
        let counts = copy_counts(&before, &device.work_counts());
        assert_eq!(
            counts,
            [0, 0, 1, 0],
            "the plain rebind: made, refused, shared, gate"
        );
        copies.prepare(1);
        let command_buffer = queue.command_buffer().unwrap();
        write(&command_buffer, &writer, &copies);
        read_bytes(&command_buffer, &plain, &copies);
        command_buffer.commit().unwrap();
        finish(&command_buffer);
        assert_eq!(
            count(bytes_of(&copies.out), texel(1)),
            W * H,
            "a read through the rebound buffer misses the surface's write in one command buffer"
        );
        // A buffer at half a page into other pages, a texture over it planned there, rebound onto
        // the surface's: the bytes moved in their page under the texture.
        let off = host_page() / 2;
        // SAFETY: the other mapping outlives the buffer, and holds `off + LEN` bytes.
        let pointer = unsafe { others[1].pointer().add(off) };
        let made = device.work_counts().copied_host_buffers_made;
        // SAFETY: as above.
        let moved = unsafe {
            no_copy(
                device,
                pointer,
                LEN as usize,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let reader = plane_texture(&moved);
        let before = device.work_counts();
        assert_eq!(
            before.copied_host_buffers_made, made,
            "the buffer over other pages imports them"
        );
        rebind(&moved);
        let after = device.work_counts();
        assert_eq!(
            copy_counts(&before, &after),
            [1, 0, 0, 0],
            "the moved rebind: made, refused, shared, gate"
        );
        assert_eq!(
            after.copied_rebind_pages_moved - before.copied_rebind_pages_moved,
            1
        );
        copies.prepare(2);
        let command_buffer = queue.command_buffer().unwrap();
        write(&command_buffer, &writer, &copies);
        command_buffer.commit().unwrap();
        finish(&command_buffer);
        let command_buffer = queue.command_buffer().unwrap();
        read(&command_buffer, &reader, &copies);
        command_buffer.commit().unwrap();
        finish(&command_buffer);
        assert_eq!(count(bytes_of(&copies.out), texel(2)), W * H);
        drop((reader, moved, plain, writer, surface));
    });
}
