//! A render pass's store-out into memory of Metal's private option bit 24 on a device whose
//! system maps no read-only view of it (`store_out.rs`): the command buffer fails with
//! `PageFault` from the pass on, as Metal's GPU faults it.
//!
//! - **Each store-out faults.** A colour attachment over a texture over a no-copy buffer of the
//!   bit at `Store` (the buffer with a read twin and without one), a multisample attachment
//!   resolving into such a texture (the descriptor's `MultisampleResolve` and
//!   `StoreAndMultisampleResolve`, and one the encoder set), a depth attachment in a no-copy heap
//!   of the bit at `CustomSampleDepthStore` (the descriptor's and the encoder's) and at `Store`,
//!   and a multisample depth attachment resolving into a texture in such a heap at
//!   `MultisampleResolve`: the encoder before the pass runs, the pass and the encoder after it do
//!   not, the command buffer's event signal fires, the pages keep their bytes, and the next
//!   command buffer of the queue runs and reads them. Each fault runs on a queue of its own: a
//!   queue's second fault revokes it (`command_buffer/revocation.rs`).
//! - **A stencil stored or resolved into such a placement heap faults**, as the M3 Max faults a
//!   stencil's store and resolve into a heap of the bit: a `Stencil8` texture at the start of a
//!   placement no-copy heap of the bit, as a guest makes a read-only heap, stored at `Store` (the
//!   descriptor's and the encoder's), and a 4-sample `Stencil8` attachment of the device's own
//!   resolving into it at `MultisampleResolve` (the descriptor's and the encoder's) and
//!   `StoreAndMultisampleResolve`; at `DontCare`, at a late `CustomSampleDepthStore` (a stencil's
//!   `DontCare`) and attached unresolved, it completes. On a device without `S8_UINT` (Intel's
//!   UHD 630 under its Windows driver) its image is a combined depth/stencil one
//!   (`StencilInCombined`), printed.
//! - **A depth stored and resolved into such a heap completes**, its resolve dropped, as on the
//!   M3 Max: a multisample depth attachment of the device's own at `StoreAndMultisampleResolve`
//!   into a texture in a no-copy heap of the bit (the descriptor's action and one the encoder set,
//!   on one queue): each command buffer completes, the pages and the resolve texture keep their
//!   bytes, and a later pass on that queue loads the depth's stored samples and resolves them
//!   into a writable texture.
//! - **What does not store out completes.** The colour attachment at `DontCare`, and a resolve
//!   texture over the buffer that the pass does not resolve into; and a no-copy buffer without
//!   the bit takes the store.
//! - **A blit into a texture over it is dropped**, as Metal's is: a copy into it out of a buffer
//!   and out of a texture complete, and the pages keep their bytes (a kernel's write to such a
//!   texture: `tests/read_twin_device.rs`).
//! - **The CPU's write lands.** `replace_region` of a texture over the buffer writes the pages, as
//!   Metal's does (the bit leaves them writable), and the next command buffer reads it.
//! - **The negative control.** With the fault off on the test's device, the colour store over the
//!   buffer of the bit completes.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). A device
//! that imports no host memory, and a system that maps read-only views (macOS, where Metal's own
//! GPU faults the store), say so and pass.

use crate::{
    BlitOption, Buffer, CommandBufferError, CommandBufferStatus, CommandQueue, Device, DeviceKind,
    DeviceOptions, EventRef, Heap, HeapDescriptor, HeapType, HostMemory, LoadAction, Origin,
    Region, RenderCommandEncoder, RenderPassDescriptor, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor,
};
use metal2vulkan::device::{ContainmentPolicy, DevicePolicy, GuestIsolation};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(120);
/// The pages' bytes, and their alignment: a heap's pages hold at least these.
const LENGTH: usize = 64 * 1024;
/// The texture's side, in texels of 4 bytes, and its rows' bytes.
const SIDE: u64 = 64;
const ROW: u64 = SIDE * 4;
/// The bytes of the fills before and after the pass.
const FILLED: usize = 256;
/// The bits of the depth the depth passes clear to.
const QUARTER: u32 = 0x3e80_0000;

/// Host memory of whole pages in a pattern, freed when dropped (after the device).
struct Pages {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    fn new() -> Pages {
        Pages::of(LENGTH, LENGTH)
    }

    /// `length` bytes at an alignment of `align`, both multiples of [`LENGTH`].
    fn of(length: usize, align: usize) -> Pages {
        let layout = Layout::from_size_align(length, align).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
        let pattern: Vec<u8> = (0..length)
            .map(|at| (at as u8).wrapping_mul(29) | 1)
            .collect();
        // SAFETY: the allocation's bytes, which nothing else uses yet.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), pointer.as_ptr(), length) };
        Pages { pointer, layout }
    }

    /// Pages a heap over which holds a texture of `descriptor`: as many as the device's answer
    /// for it in a heap (`heap_texture_size_and_align`, which a program sizes its heap by) at its
    /// alignment, and [`LENGTH`] at least.
    fn holding(device: &Device, descriptor: &TextureDescriptor) -> Pages {
        let answer = device.heap_texture_size_and_align(descriptor).unwrap();
        let align = (answer.align as usize).max(LENGTH);
        Pages::of((answer.size as usize).next_multiple_of(align), align)
    }

    fn length(&self) -> usize {
        self.layout.size()
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the allocation's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length()) }.to_vec()
    }

    fn memory(&self) -> HostMemory {
        // SAFETY: the pages outlive every buffer and heap over them and the device (dropped
        // after them).
        unsafe { HostMemory::new(self.pointer) }
    }

    /// A no-copy buffer over the pages, of bit 24 where `private_bit`.
    fn buffer(&self, device: &Device, private_bit: bool) -> Buffer {
        // SAFETY: as in `memory`.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                self.memory(),
                self.length() as u64,
                options(private_bit),
                None,
            )
        }
        .expect("a no-copy buffer over the pages")
    }

    /// An automatic no-copy heap of bit 24 over the pages.
    fn heap(&self, device: &Device) -> Heap {
        self.heap_of(device, HeapType::Automatic)
    }

    /// A no-copy heap of `heap_type` and bit 24 over the pages.
    fn heap_of(&self, device: &Device, heap_type: HeapType) -> Heap {
        let descriptor = HeapDescriptor::new(heap_type, self.length() as u64, options(true));
        // SAFETY: as in `memory`.
        unsafe { device.new_heap_with_bytes_no_copy(self.memory(), &descriptor, None, None) }
            .expect("a no-copy heap of bit 24 over the pages")
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// `Shared`, with Metal's private option bit 24 where `private_bit`.
fn options(private_bit: bool) -> ResourceOptions {
    let bit = match private_bit {
        true => ResourceOptions::PRIVATE_BIT,
        false => 0,
    };
    ResourceOptions::from_raw(ResourceOptions::shared().raw() | bit).unwrap()
}

/// The test's device, under `policy` where given; `None` where the store-out cannot fault here
/// (said).
fn device(policy: Option<DevicePolicy>) -> Option<Device> {
    device_of(policy, DeviceOptions::default())
}

/// As [`device`] under the default policy, with the queue plan (`queue_limits: None`) in place of
/// the runtime tests' default queue limits (`device/create.rs`): for a test that keeps more than
/// two of its command queues live at once.
fn plan_device() -> Option<Device> {
    let base = DeviceOptions {
        queue_limits: None,
        ..DeviceOptions::default()
    };
    device_of(None, base)
}

/// [`device`], its other options `base`'s.
fn device_of(policy: Option<DevicePolicy>, base: DeviceOptions) -> Option<Device> {
    if crate::read_only_view::VIEWS_DROP_WRITES {
        eprintln!("STORE-OUT: bit 24 maps a read-only view here; nothing to test");
        return None;
    }
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..base
    };
    if let Some(policy) = policy {
        options.policy = policy;
    }
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("STORE-OUT: the device imports no host memory here; nothing to test");
        return None;
    }
    Some(device)
}

/// A `SIDE` x `SIDE` `RGBA8Unorm` render target over `buffer`'s bytes from its first.
fn over(buffer: &Buffer) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    buffer.new_texture(&descriptor, 0, ROW).unwrap()
}

/// A `Private` texture of the device's own, `SIDE` x `SIDE`, of `format` and `samples`.
fn own_texture(device: &Device, format: PixelFormat, samples: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    if samples > 1 {
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = samples;
    }
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// A buffer of the device's own of `length` bytes.
fn own(device: &Device, length: usize) -> Buffer {
    device
        .new_buffer(length as u64, ResourceOptions::shared())
        .unwrap()
}

/// `buffer`'s first `length` bytes as the CPU reads them.
fn contents(buffer: &Buffer, length: usize) -> Vec<u8> {
    let at = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), length) }.to_vec()
}

/// What a command buffer around one pass answered.
#[derive(Debug, PartialEq)]
struct Answer {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    /// The fill of the encoder before the pass landed, and the one after it.
    before: bool,
    after: bool,
    /// The value its event signal set.
    signaled: u64,
}

impl Answer {
    /// Metal's answer to a store-out into memory of bit 24: a page fault from the pass on, its
    /// event signal fired.
    const FAULTED: Answer = Answer {
        status: CommandBufferStatus::Error,
        error: Some(CommandBufferError::PageFault),
        before: true,
        after: false,
        signaled: 1,
    };
    const COMPLETED: Answer = Answer {
        status: CommandBufferStatus::Completed,
        error: None,
        before: true,
        after: true,
        signaled: 1,
    };
}

/// Commit, on `queue`, a fill of a buffer of the device's own, the pass of `pass` (its encoder
/// given to `late` before it ends), a fill of another, and a signal of a new shared event to 1;
/// wait for it and answer what it did.
fn around_pass(
    device: &Device,
    queue: &CommandQueue,
    pass: &RenderPassDescriptor,
    late: impl FnOnce(&mut RenderCommandEncoder),
) -> Answer {
    let event = device.new_shared_event().unwrap();
    let [first, second] = [own(device, FILLED), own(device, FILLED)];
    let command_buffer = queue.command_buffer().unwrap();
    let fill = |buffer: &Buffer| {
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(buffer, 0..FILLED as u64, 0x5a).unwrap();
        encoder.end_encoding().unwrap();
    };
    fill(&first);
    let mut encoder = command_buffer.render_command_encoder(pass).unwrap();
    late(&mut encoder);
    encoder.end_encoding().unwrap();
    fill(&second);
    (command_buffer.encode_signal_event(EventRef::Shared(&event), 1)).unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let filled = |buffer: &Buffer| contents(buffer, FILLED) == [0x5a; FILLED];
    Answer {
        status: command_buffer.status(),
        error: command_buffer.error(),
        before: filled(&first),
        after: filled(&second),
        signaled: event.signaled_value(),
    }
}

/// Whether the next command buffer of `queue`, a copy out of `buffer`, runs and reads `pages`'s
/// bytes.
fn next_reads_pages(device: &Device, queue: &CommandQueue, buffer: &Buffer, pages: &Pages) {
    let out = own(device, LENGTH);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_buffer_to_buffer(buffer, 0, &out, 0, LENGTH as u64)).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(
        contents(&out, LENGTH) == pages.bytes(),
        "the pages' bytes read"
    );
}

/// A pass storing `target` (`Clear`, then `store`).
fn color_pass(target: &Texture, store: StoreAction) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = store;
    pass
}

/// A pass of the 4-sample `Depth32Float` texture `samples` (`load`ed, 0.25 where cleared)
/// resolving into `into` at `store`.
fn depth_resolve_pass(
    samples: &Texture,
    load: LoadAction,
    into: &Texture,
    store: StoreAction,
) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let depth = &mut pass.depth_attachment;
    depth.texture = Some(samples.clone());
    depth.load_action = load;
    depth.clear_depth = 0.25;
    depth.store_action = store;
    depth.resolve_texture = Some(into.clone());
    pass
}

/// The bits of the `Depth32Float` texels of `texture`, `SIDE` x `SIDE`, copied out by a command
/// buffer of `queue`, which completes.
fn depth_texels(device: &Device, queue: &CommandQueue, texture: &Texture) -> Vec<u32> {
    let out = own(device, (ROW * SIDE) as usize);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &out,
        0,
        ROW,
        ROW * SIDE,
        BlitOption::None,
    ))
    .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let bytes = contents(&out, (ROW * SIDE) as usize);
    let words = bytes.as_chunks::<4>().0.iter();
    words.map(|&word| u32::from_ne_bytes(word)).collect()
}

/// A 4-sample pass of the device's own target resolving into `into` at `store`.
fn resolve_pass(device: &Device, into: &Texture, store: StoreAction) -> RenderPassDescriptor {
    let samples = own_texture(device, PixelFormat::RGBA8Unorm, 4);
    let mut pass = color_pass(&samples, store);
    pass.color_attachments[0].resolve_texture = Some(into.clone());
    pass
}

/// The colour store over a buffer of bit 24 that has a read twin where `twinned` faults.
fn store_faults(twinned: bool) {
    let policy = (!twinned).then(|| {
        let mut policy = DevicePolicy::UNTRUSTED;
        policy.containment = ContainmentPolicy::UnsafeOff;
        policy.isolation = GuestIsolation::Off;
        policy
    });
    let pages = Pages::new();
    let Some(device) = device(policy) else {
        return;
    };
    let kept = pages.bytes();
    let buffer = pages.buffer(&device, true);
    assert_eq!(buffer.read_twin().is_some(), twinned);
    let target = over(&buffer);
    let queue = device.new_command_queue().unwrap();
    let answer = around_pass(
        &device,
        &queue,
        &color_pass(&target, StoreAction::Store),
        |_| {},
    );
    assert_eq!(answer, Answer::FAULTED);
    assert!(pages.bytes() == kept, "the pages keep their bytes");
    next_reads_pages(&device, &queue, &buffer, &pages);
    eprintln!("STORE-OUT: a colour store over a buffer of bit 24 (twin: {twinned}): {answer:?}");
}

#[test]
fn a_color_store_into_a_private_bit_buffer_faults_its_command_buffer() {
    store_faults(true);
}

#[test]
fn a_color_store_into_a_private_bit_buffer_without_a_twin_faults_its_command_buffer() {
    store_faults(false);
}

#[test]
fn a_resolve_into_a_private_bit_buffer_faults_its_command_buffer() {
    let pages = Pages::new();
    let Some(device) = plan_device() else {
        return;
    };
    let kept = pages.bytes();
    let buffer = pages.buffer(&device, true);
    let into = over(&buffer);
    // The descriptor's resolve, then one the encoder sets, each on a queue of its own (a queue's
    // second fault revokes it).
    let pass = resolve_pass(&device, &into, StoreAction::MultisampleResolve);
    let queue = device.new_command_queue().unwrap();
    assert_eq!(around_pass(&device, &queue, &pass, |_| {}), Answer::FAULTED);
    let queue = device.new_command_queue().unwrap();
    let pass = resolve_pass(&device, &into, StoreAction::Unknown);
    let late = |encoder: &mut RenderCommandEncoder| {
        (encoder.set_color_store_action(StoreAction::MultisampleResolve, 0)).unwrap();
    };
    assert_eq!(around_pass(&device, &queue, &pass, late), Answer::FAULTED);
    // Stored and resolved, as the M3 Max faults it (unlike a depth's, below).
    let stored = device.new_command_queue().unwrap();
    let pass = resolve_pass(&device, &into, StoreAction::StoreAndMultisampleResolve);
    let answer = around_pass(&device, &stored, &pass, |_| {});
    assert_eq!(answer, Answer::FAULTED, "StoreAndMultisampleResolve");
    assert!(pages.bytes() == kept, "the pages keep their bytes");
    next_reads_pages(&device, &queue, &buffer, &pages);
    // Attached and not resolved into, it is not written: the pass completes.
    let pass = resolve_pass(&device, &into, StoreAction::Store);
    assert_eq!(
        around_pass(&device, &queue, &pass, |_| {}),
        Answer::COMPLETED
    );
    assert!(pages.bytes() == kept);
}

#[test]
fn a_depth_store_into_a_private_bit_heap_faults_its_command_buffer() {
    // Declared before the device, so dropped after it.
    let pages;
    let Some(device) = device(None) else {
        return;
    };
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options(false);
    pages = Pages::holding(&device, &descriptor);
    let kept = pages.bytes();
    let heap = pages.heap(&device);
    let depth = heap.new_texture(&descriptor).unwrap();
    let depth_pass = |store: StoreAction| {
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 0.25;
        pass.depth_attachment.store_action = store;
        pass
    };
    // Each fault on a queue of its own (a queue's second fault revokes it).
    for store in [StoreAction::CustomSampleDepthStore, StoreAction::Store] {
        let queue = device.new_command_queue().unwrap();
        let answer = around_pass(&device, &queue, &depth_pass(store), |_| {});
        assert_eq!(answer, Answer::FAULTED, "{store:?}");
    }
    let late = |encoder: &mut RenderCommandEncoder| {
        (encoder.set_depth_store_action(StoreAction::CustomSampleDepthStore)).unwrap();
    };
    let queue = device.new_command_queue().unwrap();
    let answer = around_pass(&device, &queue, &depth_pass(StoreAction::Unknown), late);
    assert_eq!(answer, Answer::FAULTED, "a late CustomSampleDepthStore");
    let answer = around_pass(&device, &queue, &depth_pass(StoreAction::DontCare), |_| {});
    assert_eq!(answer, Answer::COMPLETED, "DontCare");
    assert!(pages.bytes() == kept, "the pages keep their bytes");
}

/// A `SIDE` x `SIDE` `Stencil8` render target's descriptor, `Shared` (a heap of bit 24 holds it).
fn stencil_descriptor() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::Stencil8, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options(false);
    descriptor
}

/// A pass of the stencil texture `texture`, cleared to 0x5a, at `store`, resolving into `into`
/// where given.
fn stencil_pass(
    texture: &Texture,
    store: StoreAction,
    into: Option<&Texture>,
) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let stencil = &mut pass.stencil_attachment;
    stencil.texture = Some(texture.clone());
    stencil.load_action = LoadAction::Clear;
    stencil.clear_stencil = 0x5a;
    stencil.store_action = store;
    stencil.resolve_texture = into.cloned();
    pass
}

/// An encoder's late stencil store action `store`.
fn late_stencil(store: StoreAction) -> impl FnOnce(&mut RenderCommandEncoder) {
    move |encoder: &mut RenderCommandEncoder| encoder.set_stencil_store_action(store).unwrap()
}

#[test]
fn a_stencil_store_into_a_private_bit_placement_heap_faults_its_command_buffer() {
    // Declared before the device, so dropped after it.
    let pages;
    let Some(device) = device(None) else {
        return;
    };
    let descriptor = stencil_descriptor();
    pages = Pages::holding(&device, &descriptor);
    let kept = pages.bytes();
    let heap = pages.heap_of(&device, HeapType::Placement);
    let stencil = heap.new_texture_at_offset(&descriptor, 0).unwrap();
    let format = stencil.image_plan().unwrap().format;
    eprintln!("STORE-OUT: a Stencil8 texture in a placement heap of bit 24, held in {format:?}");
    // Each fault on a queue of its own (a queue's second fault revokes it).
    let queue = device.new_command_queue().unwrap();
    let pass = stencil_pass(&stencil, StoreAction::Store, None);
    assert_eq!(
        around_pass(&device, &queue, &pass, |_| {}),
        Answer::FAULTED,
        "Store"
    );
    let queue = device.new_command_queue().unwrap();
    let pass = stencil_pass(&stencil, StoreAction::Unknown, None);
    let late = late_stencil(StoreAction::Store);
    assert_eq!(
        around_pass(&device, &queue, &pass, late),
        Answer::FAULTED,
        "a late Store"
    );
    let queue = device.new_command_queue().unwrap();
    let pass = stencil_pass(&stencil, StoreAction::DontCare, None);
    assert_eq!(
        around_pass(&device, &queue, &pass, |_| {}),
        Answer::COMPLETED,
        "DontCare"
    );
    let pass = stencil_pass(&stencil, StoreAction::Unknown, None);
    let late = late_stencil(StoreAction::CustomSampleDepthStore);
    let answer = around_pass(&device, &queue, &pass, late);
    assert_eq!(answer, Answer::COMPLETED, "a late CustomSampleDepthStore");
    assert!(pages.bytes() == kept, "the pages keep their bytes");
}

#[test]
fn a_stencil_resolve_into_a_private_bit_placement_heap_faults_its_command_buffer() {
    // Declared before the device, so dropped after it.
    let pages;
    let Some(device) = plan_device() else {
        return;
    };
    let descriptor = stencil_descriptor();
    pages = Pages::holding(&device, &descriptor);
    let kept = pages.bytes();
    let heap = pages.heap_of(&device, HeapType::Placement);
    let into = heap.new_texture_at_offset(&descriptor, 0).unwrap();
    let samples = own_texture(&device, PixelFormat::Stencil8, 4);
    let format = samples.image_plan().unwrap().format;
    eprintln!("STORE-OUT: a 4-sample Stencil8 texture held in {format:?}");
    // Each fault on a queue of its own (a queue's second fault revokes it).
    let resolve = StoreAction::MultisampleResolve;
    let smsr = StoreAction::StoreAndMultisampleResolve;
    for (store, late) in [
        (resolve, None),
        (StoreAction::Unknown, Some(resolve)),
        (smsr, None),
    ] {
        let queue = device.new_command_queue().unwrap();
        let pass = stencil_pass(&samples, store, Some(&into));
        let answer = match late {
            Some(late) => around_pass(&device, &queue, &pass, late_stencil(late)),
            None => around_pass(&device, &queue, &pass, |_| {}),
        };
        assert_eq!(answer, Answer::FAULTED, "{store:?}, late {late:?}");
    }
    // Attached and not resolved into, it is not written: the pass completes.
    let queue = device.new_command_queue().unwrap();
    let pass = stencil_pass(&samples, StoreAction::Store, Some(&into));
    assert_eq!(
        around_pass(&device, &queue, &pass, |_| {}),
        Answer::COMPLETED,
        "Store"
    );
    assert!(pages.bytes() == kept, "the pages keep their bytes");
}

#[test]
fn a_depth_stored_and_resolved_into_a_private_bit_heap_completes_without_its_resolve() {
    // Declared before the device, so dropped after it.
    let pages;
    let Some(device) = device(None) else {
        return;
    };
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options(false);
    pages = Pages::holding(&device, &descriptor);
    let kept = pages.bytes();
    let heap = pages.heap(&device);
    let into = heap.new_texture(&descriptor).unwrap();
    let samples = own_texture(&device, PixelFormat::Depth32Float, 4);
    let queue = device.new_command_queue().unwrap();
    let texels = depth_texels(&device, &queue, &into);
    assert!(
        texels.iter().any(|&depth| depth != QUARTER),
        "the resolve texture holds other texels than the resolve's"
    );
    // The descriptor's action, then one the encoder sets, on one queue: a fault would revoke it
    // at the second.
    let clear = LoadAction::Clear;
    let smsr = StoreAction::StoreAndMultisampleResolve;
    let pass = depth_resolve_pass(&samples, clear, &into, smsr);
    let answer = around_pass(&device, &queue, &pass, |_| {});
    assert_eq!(answer, Answer::COMPLETED, "the descriptor's action");
    let pass = depth_resolve_pass(&samples, clear, &into, StoreAction::Unknown);
    let late = |encoder: &mut RenderCommandEncoder| {
        encoder.set_depth_store_action(smsr).unwrap();
    };
    let answer = around_pass(&device, &queue, &pass, late);
    assert_eq!(answer, Answer::COMPLETED, "the encoder's action");
    assert!(pages.bytes() == kept, "the pages keep their bytes");
    assert!(
        depth_texels(&device, &queue, &into) == texels,
        "the resolve texture keeps its texels"
    );
    // The depth's samples were stored: a later pass loads them and resolves 0.25.
    let writable = own_texture(&device, PixelFormat::Depth32Float, 1);
    let resolve = StoreAction::MultisampleResolve;
    let pass = depth_resolve_pass(&samples, LoadAction::Load, &writable, resolve);
    let answer = around_pass(&device, &queue, &pass, |_| {});
    assert_eq!(answer, Answer::COMPLETED, "the later pass");
    let resolved = depth_texels(&device, &queue, &writable);
    assert!(
        resolved.iter().all(|&depth| depth == QUARTER),
        "the stored samples resolve to the clear"
    );
    // The control: resolved alone into the heap, the depth faults its command buffer.
    let queue = device.new_command_queue().unwrap();
    let pass = depth_resolve_pass(&samples, clear, &into, resolve);
    let answer = around_pass(&device, &queue, &pass, |_| {});
    assert_eq!(answer, Answer::FAULTED, "MultisampleResolve");
    assert!(pages.bytes() == kept, "the pages keep their bytes");
    eprintln!("STORE-OUT: a depth stored and resolved into a heap of bit 24 completes");
}

#[test]
fn what_stores_nothing_into_a_private_bit_buffer_completes() {
    let pages = Pages::new();
    let writable = Pages::new();
    let Some(device) = device(None) else {
        return;
    };
    let kept = pages.bytes();
    let buffer = pages.buffer(&device, true);
    let target = over(&buffer);
    let queue = device.new_command_queue().unwrap();
    let pass = color_pass(&target, StoreAction::DontCare);
    assert_eq!(
        around_pass(&device, &queue, &pass, |_| {}),
        Answer::COMPLETED
    );
    assert!(pages.bytes() == kept);
    // The writable control: a buffer without the bit takes the store.
    let buffer = writable.buffer(&device, false);
    let target = over(&buffer);
    let pass = color_pass(&target, StoreAction::Store);
    assert_eq!(
        around_pass(&device, &queue, &pass, |_| {}),
        Answer::COMPLETED
    );
    assert!(writable.bytes() != kept, "the store reaches the pages");
}

#[test]
fn with_the_fault_off_a_color_store_into_a_private_bit_buffer_completes() {
    let pages = Pages::new();
    let Some(device) = device(None) else {
        return;
    };
    (device.shared().store_out_unfaulted).store(true, Ordering::Relaxed);
    let buffer = pages.buffer(&device, true);
    let target = over(&buffer);
    let queue = device.new_command_queue().unwrap();
    let pass = color_pass(&target, StoreAction::Store);
    assert_eq!(
        around_pass(&device, &queue, &pass, |_| {}),
        Answer::COMPLETED
    );
}

#[test]
fn replace_region_of_a_texture_over_a_private_bit_buffer_writes_its_pages() {
    let pages = Pages::new();
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.buffer(&device, true);
    let texture = over(&buffer);
    let texels = vec![0x3c; (ROW * SIDE) as usize];
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    (texture.replace_region(region, 0, 0, &texels, ROW, 0)).unwrap();
    assert!(
        pages.bytes()[..texels.len()] == texels[..],
        "the CPU's write reaches the pages"
    );
    let queue = device.new_command_queue().unwrap();
    next_reads_pages(&device, &queue, &buffer, &pages);
}

#[test]
fn a_blit_into_a_texture_over_a_private_bit_buffer_is_dropped() {
    let pages = Pages::new();
    let Some(device) = device(None) else {
        return;
    };
    let kept = pages.bytes();
    let buffer = pages.buffer(&device, true);
    let into = over(&buffer);
    let source = own(&device, (ROW * SIDE) as usize);
    let texels = own_texture(&device, PixelFormat::RGBA8Unorm, 1);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder.fill_buffer(&source, 0..ROW * SIDE, 0x3c).unwrap();
    let size = Size::new(SIDE, SIDE, 1);
    let origin = Origin::new(0, 0, 0);
    (encoder.copy_from_buffer_to_texture(
        &source,
        0,
        ROW,
        0,
        size,
        &into,
        0,
        0,
        origin,
        BlitOption::None,
    ))
    .unwrap();
    encoder.copy_texture(&texels, &into).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(pages.bytes() == kept, "the pages keep their bytes");
    next_reads_pages(&device, &queue, &buffer, &pages);
}
