//! A dma-buf source's standing refusal (`DmaBufStandingRefusal`, `texture/in_place_image.rs`), on a
//! device that imports no host pointer (`tests/no_pointer_import.rs`'s device): an `RGBA8Unorm`
//! `ShaderRead` texture over a no-copy buffer of a sealed `memfd`'s pages, imported as dma-bufs,
//! whose making tries its image in place with each of its plans and asks the buffer's source for
//! each plan's pages. Each case makes its own buffer and texture, its source counting the texture's
//! requests (those that take no shared file; the buffer's own takes them):
//! - **plain refusals** (a plain `io::Error`): asked once for each plan the device tries, and the
//!   texture is a shadow (`host_texture_shadows_refused_source`); the count of plans;
//! - **standing refusals**: asked once whatever the plans, and the texture is a shadow;
//! - **a plain refusal, then the pages**: where the device tries more than one plan, asked twice,
//!   and the texture is in place over the second plan's image (`host_textures_in_place`): only a
//!   standing refusal stops the asking, and a wider plan may still be taken after a plain one.
//!
//! The standing case's one call against the plain case's count is the skip: the two sources
//! differ in their error's type alone. A device that tries one plan says the skip is not shown
//! there; one with no dma-buf route, or whose check takes no texture over such pages in place
//! (no request asked), says so and passes. Where `/dev/udmabuf` cannot be opened the test says so
//! and passes without running.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.

use super::host_residency::{own_dma_buf_images, udmabuf, udmabuf_openable, Ram, Scratch};
use super::no_pointer_import::{device, lists_dma_buf_route};
use crate::{
    Buffer, Device, DmaBufChunk, DmaBufRequest, DmaBufSource, DmaBufStandingRefusal, HostMemory,
    ResourceOptions, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const TEST: &str = "a_standing_refusal_is_asked_once_and_a_plain_one_for_each_plan";
/// The texture's side, in texels: rows of 256 bytes, which RADV lays such a plane at.
const SIDE: u64 = 64;
/// Each case's buffer, and its place in the file.
const LENGTH: usize = 64 << 10;
/// The longest the test waits for the device's in-place checks to answer.
const CHECKS: Duration = Duration::from_secs(120);

/// How a case's source answers the texture's requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answer {
    Plain,
    Standing,
    PlainThenPages,
}

/// A source of `ram`'s pages: one dma-buf of the pages asked for, but where a texture's request
/// (one that takes no shared file) is refused as `answer` says; those requests counted in `calls`.
fn source(ram: &Ram, answer: Answer, calls: &Arc<AtomicUsize>) -> DmaBufSource {
    let fd = Arc::new(ram.file().try_clone().unwrap());
    let base = ram.pointer(0).as_ptr() as usize;
    let calls = Arc::clone(calls);
    Arc::new(move |request: DmaBufRequest| {
        if !request.shared_files {
            let call = calls.fetch_add(1, Ordering::AcqRel);
            let refused = match answer {
                Answer::Plain => true,
                Answer::Standing => {
                    let refusal = DmaBufStandingRefusal::new("the test's standing refusal");
                    return Err(std::io::Error::other(refusal));
                }
                Answer::PlainThenPages => call == 0,
            };
            if refused {
                return Err(std::io::Error::other("the test's plain refusal"));
            }
        }
        let offset = (request.start.as_ptr() as usize)
            .checked_sub(base)
            .ok_or_else(|| std::io::Error::other("pages before the view"))?;
        Ok(vec![DmaBufChunk {
            file: Arc::new(udmabuf(&fd, offset as u64, request.length)?),
            offset: 0,
            length: request.length,
        }])
    })
}

/// A no-copy buffer over `ram`'s view from `at`, of [`LENGTH`] bytes, its pages' source `source`.
fn buffer(device: &Device, ram: &Ram, at: usize, source: DmaBufSource) -> Buffer {
    // SAFETY: the view outlives the buffer (the test drops its buffers first), and the source
    // makes dma-bufs of its pages.
    unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(ram.pointer(at), source),
            LENGTH as u64,
            ResourceOptions::shared(),
            None,
        )
    }
    .unwrap()
}

/// One case's texture over a buffer of its own: its source's calls for the texture, and whether
/// it lies in place (else it is a shadow, refused by the source).
fn case(device: &Device, ram: &Ram, at: usize, answer: Answer) -> (usize, bool) {
    let calls = Arc::new(AtomicUsize::new(0));
    let buffer = buffer(device, ram, at, source(ram, answer, &calls));
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, 0, SIDE * 4).unwrap();
    let after = device.work_counts();
    let moved = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let in_place = moved(|c| c.host_textures_in_place);
    let refused = moved(|c| c.host_texture_shadows_refused_source);
    let calls = calls.load(Ordering::Acquire);
    eprintln!(
        "{TEST}: on {}: {answer:?}: the source asked {calls} times for the texture; in place \
         {in_place}, refused by the source {refused}",
        device.name()
    );
    drop(texture);
    drop(buffer);
    (calls, in_place == 1)
}

#[test]
fn a_standing_refusal_is_asked_once_and_a_plain_one_for_each_plan() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0, None);
    if !lists_dma_buf_route(&device) {
        eprintln!("{TEST}: on {}: no dma-buf route: not run", device.name());
        return;
    }
    assert!(
        device.settle_in_place_checks(CHECKS),
        "the in-place checks never answered"
    );
    // The images in dma-buf imports of their own, which ask the source, whatever the check of
    // images in the buffer's import answered.
    own_dma_buf_images(&device);
    let ram = Ram::new(3 * LENGTH, 0, 3 * LENGTH);
    let (plans, plain_in_place) = case(&device, &ram, 0, Answer::Plain);
    assert!(
        !plain_in_place,
        "a texture whose every request was refused lies in place"
    );
    if plans == 0 {
        eprintln!(
            "{TEST}: on {}: the texture was asked no pages (no image in place over them): not run",
            device.name()
        );
        return;
    }
    let (standing, standing_in_place) = case(&device, &ram, LENGTH, Answer::Standing);
    assert!(!standing_in_place);
    assert_eq!(
        standing, 1,
        "a standing refusal is asked once, of {plans} plans"
    );
    let (asked, in_place) = case(&device, &ram, 2 * LENGTH, Answer::PlainThenPages);
    if plans == 1 {
        eprintln!(
            "{TEST}: on {}: the device tries one plan: the skip is not shown here",
            device.name()
        );
        assert_eq!((asked, in_place), (1, false));
        return;
    }
    assert_eq!(
        (asked, in_place),
        (2, true),
        "after a plain refusal the next plan is asked, and its pages taken"
    );
}
