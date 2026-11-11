//! A placement heap's `new_texture_at_offset` that makes no texture at an offset inside the heap
//! names it in an informational message (`metal2vulkan-heap-texture-refused`), the device's first
//! such refusal alone: one off the size query's alignment, or one at an aligned offset whose bytes
//! run past the heap's end. Each later one is counted
//! (`WorkCounts::heap_texture_refused_withheld`), not named. One at the heap's end, outside it, is
//! neither named nor counted, nor is one the heap makes.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU.

use crate::device::lock;
use crate::{
    DebugMessage, Device, DeviceKind, DeviceOptions, Error, Heap, HeapDescriptor, HeapType,
    MessageSeverity, MessageSink, ResourceOptions, SizeAndAlign, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::{Arc, Mutex};

/// What the device's sink heard of the message: each one's severity and text.
type Heard = Arc<Mutex<Vec<(MessageSeverity, String)>>>;

/// The device the cfg selects, with a sink that keeps every `metal2vulkan-heap-texture-refused`
/// message.
fn listening() -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let kept = Arc::clone(&heard);
    let sink = MessageSink::new(move |message: &DebugMessage| {
        if message.id_name == "metal2vulkan-heap-texture-refused" {
            lock(&kept).push((message.severity, message.text.clone()));
        }
    });
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(sink),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

/// A heap of room for `d` and its alignment again on `device`, with the size query's answer for
/// `d` and the last offset at the alignment inside the heap, whose bytes run past its end.
fn placement_heap(device: &Device, d: &TextureDescriptor) -> (Heap, SizeAndAlign, u64) {
    let answer = device.heap_texture_size_and_align(d).unwrap();
    let hd = HeapDescriptor::new(
        HeapType::Placement,
        answer.size + answer.align,
        ResourceOptions::shared(),
    );
    let heap = device.new_heap(&hd).unwrap();
    let size = heap.size();
    let last = (size - 1) / answer.align * answer.align;
    assert!(last + answer.size > size, "{answer:?} in {size:#x}");
    (heap, answer, last)
}

/// The descriptor each test places.
fn descriptor() -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R16Float, 383, 152, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    d
}

/// Ask `heap` for a texture of `d` at `offset`, which it refuses, `Nil`.
fn refused(heap: &Heap, d: &TextureDescriptor, offset: u64) {
    let Err(e) = heap.new_texture_at_offset(d, offset) else {
        panic!("a texture at {offset:#x}");
    };
    assert!(matches!(e, Error::Nil(_)), "{offset:#x}: {e}");
}

/// The texts of the refusals each device names first, and that a later one on the same device is
/// counted and not named: on one device, the refusal off the alignment is named, the one at the
/// last aligned offset counted; on another, the one at the last aligned offset is named.
#[test]
fn a_placement_heap_names_a_texture_it_does_not_make_inside_it() {
    let d = descriptor();
    let withheld = |device: &Device| device.work_counts().heap_texture_refused_withheld;
    let (device, heard) = listening();
    let (heap, answer, last) = placement_heap(&device, &d);
    let size = heap.size();
    // Made at the heap's start: nothing said.
    assert!(heap.new_texture_at_offset(&d, 0).is_ok());
    refused(&heap, &d, 1);
    assert_eq!(lock(&heard).len(), 1, "{:?}", lock(&heard));
    assert_eq!(withheld(&device), 0);
    // A second refusal: counted, not named.
    refused(&heap, &d, last);
    assert_eq!(lock(&heard).len(), 1, "{:?}", lock(&heard));
    assert_eq!(withheld(&device), 1);
    // Outside the heap: neither.
    refused(&heap, &d, size);
    assert_eq!(withheld(&device), 1);
    let mut heard = lock(&heard).clone();
    // Another device names the one at the last aligned offset, its first.
    let (other, other_heard) = listening();
    let (other_heap, _, other_last) = placement_heap(&other, &d);
    assert_eq!((other_heap.size(), other_last), (size, last));
    refused(&other_heap, &d, last);
    heard.extend(lock(&other_heard).iter().cloned());
    assert_eq!(heard.len(), 2, "{heard:?}");
    assert_eq!(withheld(&other), 0);
    for ((severity, text), (offset, against)) in heard.iter().zip([
        (
            1,
            "the offset not a multiple of it, the bytes inside the heap",
        ),
        (
            last,
            "the offset a multiple of it, the bytes not inside the heap",
        ),
    ]) {
        assert_eq!(*severity, MessageSeverity::Info, "{text}");
        let texture = "R16Float texture of 383x152x1, 1 levels, 1 slices, 1 samples";
        assert!(text.contains(texture), "{text}");
        let at = format!("at offset {offset:#x} of a placement heap of {size:#x} bytes");
        assert!(text.contains(&at), "{text}");
        let answered = format!(
            "the size query answers it {:#x} bytes at an alignment of {:#x} ({against})",
            answer.size, answer.align
        );
        assert!(text.contains(&answered), "{text}");
        let planned = format!(
            "the making planned {:#x} bytes at an alignment of {:#x} ({against}), its image of ",
            answer.size, answer.align
        );
        assert!(text.contains(&planned), "{text}");
    }
}
