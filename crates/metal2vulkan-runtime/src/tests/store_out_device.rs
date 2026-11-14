//! A render pass's store-out into memory of Metal's private option bit 24 on a device whose
//! system maps no read-only view of it (`store_out.rs`): the command buffer fails with
//! `PageFault` from the pass on, as Metal's GPU faults it.
//!
//! - **Each store-out faults.** A colour attachment over a texture over a no-copy buffer of the
//!   bit at `Store` (the buffer with a read twin and without one), a multisample attachment
//!   resolving into such a texture (the descriptor's action and one the encoder set), and a depth
//!   attachment in a no-copy heap of the bit at `CustomSampleDepthStore` (the descriptor's and the
//!   encoder's) and at `Store`: the encoder before the pass runs, the pass and the encoder after
//!   it do not, the command buffer's event signal fires, the pages keep their bytes, and the
//!   next command buffer of the queue runs and reads them. Each fault runs on a queue of its own:
//!   a queue's second fault revokes it (`command_buffer/revocation.rs`).
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
/// The pages' bytes.
const LENGTH: usize = 64 * 1024;
/// The texture's side, in texels of 4 bytes, and its rows' bytes.
const SIDE: u64 = 64;
const ROW: u64 = SIDE * 4;
/// The bytes of the fills before and after the pass.
const FILLED: usize = 256;

/// Host memory of whole pages in a pattern, freed when dropped (after the device).
struct Pages {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    fn new() -> Pages {
        let layout = Layout::from_size_align(LENGTH, 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
        let pattern: Vec<u8> = (0..LENGTH)
            .map(|at| (at as u8).wrapping_mul(29) | 1)
            .collect();
        // SAFETY: the allocation's bytes, which nothing else uses yet.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), pointer.as_ptr(), LENGTH) };
        Pages { pointer, layout }
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the allocation's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), LENGTH) }.to_vec()
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
                LENGTH as u64,
                options(private_bit),
                None,
            )
        }
        .expect("a no-copy buffer over the pages")
    }

    /// An automatic no-copy heap of bit 24 over the pages.
    fn heap(&self, device: &Device) -> Heap {
        let descriptor = HeapDescriptor::new(HeapType::Automatic, LENGTH as u64, options(true));
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
        ..DeviceOptions::default()
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
    let Some(device) = device(None) else {
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
    let pages = Pages::new();
    let Some(device) = device(None) else {
        return;
    };
    let kept = pages.bytes();
    let heap = pages.heap(&device);
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options(false);
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
