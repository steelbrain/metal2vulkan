//! Frees while the GPU is busy (the consumer's Maps runs on MoltenVK 1.4.1 failed a command buffer
//! `Execution` code 9, `MTLCommandBufferErrorInvalidResource`). A resource the busy work never
//! used, released or made while that work runs, leaves the work whole, and the command buffers
//! committed meanwhile too: Metal fails no command buffer for a release of what it does not use,
//! and a release is legal in Vulkan once no submission in flight uses the object.
//!
//! On MoltenVK with a device-wide `MTLResidencySet` (macOS 15 or later, Metal argument buffers),
//! each memory, buffer and image the driver makes is added to that set, and each one freed or
//! destroyed is removed from it and the set committed at once, while the Metal command buffers
//! in flight keep no references of their own (`retainedReferences` is off where the set exists):
//! `MVKDevice::removeResidency`, `MVKDeviceMemory.mm`, `MVKImage.mm`, `MVKQueue.mm` in MoltenVK
//! 1.4.1. Each leg here runs such changes while the busy work's command buffers spin on the GPU,
//! then lets them end and checks their end and their errors, and every other command buffer's
//! (`FREE-IN-FLIGHT` lines). A host import's deallocator (no keep-alive) runs once its import is freed, so a leg
//! counts the imports freed while the busy work still ran.

use super::first_use::{read_mark, release, Busy, ENDED};
use super::host_memory::on_device_with;
use super::host_memory::{no_copy, not_importable, recording, Mapping, Seen, GUEST_PAGE};
use super::keep_alives::eventually;
use super::rebind_reach::{node, output, A, POINTER_WALKER};
use super::{
    at_plan, bound_buffers, encode_case, pipeline_state, public_case, read_buffer, CaseResources,
    WAIT,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue, Device,
    ResourceOptions, Texture, TextureDescriptor,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// The bytes of the legs' large buffer.
const LARGE: u64 = 64 << 20;

/// The bytes of each host import.
const IMPORTED: usize = 4 * GUEST_PAGE;

/// The imports a burst makes and releases, and the command buffers committed beside them.
const BURST: usize = 64;

/// A leg's changes while the busy work runs.
#[derive(Clone, Copy, Debug)]
enum Leg {
    /// A `Private` texture made before the busy work, released.
    TextureReleased,
    /// A `Private` buffer of [`LARGE`] bytes made before the busy work, released.
    LargeBufferReleased,
    /// A no-copy buffer over host pages, which a command buffer that completed before the busy
    /// work wrote, released.
    UsedImportReleased,
    /// A `Private` texture made.
    TextureMade,
    /// [`BURST`] no-copy buffers made and released on one thread while another commits as many
    /// command buffers on a second queue (the consumer's unmap bursts beside its submissions).
    ImportBurstBesideCommits,
}

/// What a leg holds until its change, and what counts its imports' frees.
enum Held {
    Texture(Texture),
    Buffer(Buffer),
    Import(Buffer, Arc<Seen>),
    Nothing,
}

/// A `Private` 256 x 256 `RGBA8Unorm` texture.
fn private_texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 256, 256, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// A no-copy buffer over [`IMPORTED`] fresh host bytes, whose deallocator (no keep-alive) counts
/// into `seen` and then unmaps them; `None` where the device refuses the import.
fn import(device: &Device, seen: &Arc<Seen>) -> Option<Buffer> {
    let mapping = Mapping::anonymous(IMPORTED);
    let pointer = mapping.pointer();
    let deallocator = recording(seen, mapping);
    // SAFETY: the mapping stays mapped until the deallocator, which owns it, runs.
    let made = unsafe {
        no_copy(
            device,
            pointer,
            IMPORTED,
            ResourceOptions::default(),
            Some(deallocator),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(error) => {
            assert!(not_importable(&error), "{error:?}");
            None
        }
    }
}

/// Commit `command_buffer` and wait for it, checking it completed without an error.
fn completed(command_buffer: &CommandBuffer, what: &str) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "{what}");
}

/// What `leg` holds, made before the busy work, its first work done by a submission that
/// completed: the busy work's submission does none of it, so nothing in flight uses it. `None`
/// where the device refuses the host import.
fn hold(device: &Device, queue: &CommandQueue, leg: Leg) -> Option<Held> {
    let held = match leg {
        Leg::TextureReleased => Held::Texture(private_texture(device)),
        Leg::LargeBufferReleased => Held::Buffer(
            device
                .new_buffer(LARGE, ResourceOptions::private())
                .unwrap(),
        ),
        Leg::UsedImportReleased => {
            let seen = Arc::default();
            let buffer = import(device, &seen)?;
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..IMPORTED as u64, 0x5a).unwrap();
            blit.end_encoding().unwrap();
            completed(&command_buffer, "the import's fill");
            Held::Import(buffer, seen)
        }
        Leg::TextureMade | Leg::ImportBurstBesideCommits => Held::Nothing,
    };
    completed(
        &queue.command_buffer().unwrap(),
        "the first work's submission",
    );
    Some(held)
}

/// What a leg's changes did while the busy work ran: the imports freed then, and the command
/// buffers committed beside them.
#[derive(Default)]
struct Changed {
    imports_freed: usize,
    committed: Vec<CommandBuffer>,
}

/// `leg`'s changes, on other threads: `held` released ([`release_held`]), or a [`burst`].
fn change(device: &Device, beside: &CommandQueue, leg: Leg, held: Held) -> Changed {
    match leg {
        Leg::ImportBurstBesideCommits => burst(device, beside),
        _ => release_held(device, held),
    }
}

/// [`BURST`] imports made and released on one thread while another commits as many command
/// buffers on `beside`.
fn burst(device: &Device, beside: &CommandQueue) -> Changed {
    let target = device.new_buffer(4, ResourceOptions::shared()).unwrap();
    let seen: Arc<Seen> = Arc::default();
    std::thread::scope(|scope| {
        let commits = scope.spawn(|| {
            (0..BURST)
                .map(|round| {
                    let command_buffer = beside.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(&target, 0..4, round as u8).unwrap();
                    blit.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer
                })
                .collect::<Vec<_>>()
        });
        let frees = scope.spawn(|| {
            for _ in 0..BURST {
                let Some(buffer) = import(device, &seen) else {
                    return;
                };
                drop(buffer);
            }
        });
        frees.join().unwrap();
        Changed {
            imports_freed: seen.runs.load(Ordering::SeqCst),
            committed: commits.join().unwrap(),
        }
    })
}

/// `held` released on another thread, or, for [`Held::Nothing`], a texture made and released.
fn release_held(device: &Device, held: Held) -> Changed {
    let seen = std::thread::scope(|scope| {
        scope
            .spawn(move || match held {
                Held::Texture(texture) => {
                    drop(texture);
                    None
                }
                Held::Buffer(buffer) => {
                    drop(buffer);
                    None
                }
                Held::Import(buffer, seen) => {
                    drop(buffer);
                    Some(seen)
                }
                Held::Nothing => {
                    drop(private_texture(device));
                    None
                }
            })
            .join()
            .unwrap()
    });
    Changed {
        imports_freed: seen.map_or(0, |seen| seen.runs.load(Ordering::SeqCst)),
        committed: Vec::new(),
    }
}

/// Each [`Leg`]'s changes run on other threads while the busy work spins on the GPU, its command
/// buffers committed and submitted: once released, the busy work completes, without an error,
/// its end mark written, and so does every command buffer committed beside the changes. Prints
/// each leg's outcome, with the host imports freed while the busy work ran (`FREE-IN-FLIGHT`).
#[test]
fn frees_while_the_gpu_is_busy_fail_no_command_buffer() {
    on_device_with(at_plan, |device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let beside = device.new_command_queue().unwrap();
        let busy = Busy::new(device, &queue);
        let legs = [
            Leg::TextureReleased,
            Leg::LargeBufferReleased,
            Leg::UsedImportReleased,
            Leg::TextureMade,
            Leg::ImportBurstBesideCommits,
        ];
        let mut failed = Vec::new();
        for leg in legs {
            let Some(held) = hold(device, &queue, leg) else {
                println!("FREE-IN-FLIGHT on {name}: {leg:?}: the host import is refused");
                continue;
            };
            let (busy_work, words) = busy.commit(device, &queue);
            busy_work.scheduled();
            let changed = change(device, &beside, leg, held);
            let running = read_mark(&words) == 0;
            release(&words);
            busy_work.wait_until_completed_for(WAIT).unwrap();
            let (error, status, mark) = (busy_work.error(), busy_work.status(), read_mark(&words));
            let beside_errors: Vec<(usize, CommandBufferError)> = (changed.committed.iter())
                .enumerate()
                .filter_map(|(at, command_buffer)| {
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    command_buffer.error().map(|error| (at, error))
                })
                .collect();
            println!(
                "FREE-IN-FLIGHT on {name}: {leg:?} while the busy work ran ({}; {} host imports \
                 freed): {status:?}, error {error:?}, end mark {mark:#x}; {} command buffers \
                 beside, failed {beside_errors:?}",
                match running {
                    true => "still running after the changes",
                    false => "ended before the changes returned",
                },
                changed.imports_freed,
                changed.committed.len()
            );
            if error.is_some()
                || status != CommandBufferStatus::Completed
                || mark != ENDED
                || !beside_errors.is_empty()
            {
                failed.push((leg, error, status, beside_errors));
            }
        }
        addressed_import_released(device, &queue, &busy);
        assert!(
            failed.is_empty(),
            "{name}: command buffers failed as resources they never used changed: {failed:?}"
        );
    });
}

/// An import reached only through the address an argument buffer holds, read by the walker
/// ([`POINTER_WALKER`]) committed behind the busy work on its queue, its encoder naming no use
/// of it, released while the busy work spins (the consumer's guest command buffers reach guest
/// memory through such addresses, and its core frees an import as the guest unmaps a surface).
/// Neither Metal nor Vulkan keeps such an import for the walk: the program released memory its
/// queued work reads. So the leg asserts nothing: it prints the busy work's and the walk's errors
/// and the walk's output (`FREE-IN-FLIGHT ... AddressedImportReleased`). The pages stay mapped
/// throughout, as guest memory does: the import's free alone is measured. They go once the walk
/// completed and the import's deallocator ran, which `HostMemory::new`'s contract asks: the
/// deallocator, which reads them, may run on a handler thread after the waits return, where the
/// device delays it (`host-deallocator-delayed`).
fn addressed_import_released(device: &Device, queue: &CommandQueue, busy: &Busy) {
    let name = device.name();
    let seen: Arc<Seen> = Arc::default();
    // The pages stay mapped until the walk completed and the deallocator ran (the mapping goes
    // last), as the consumer's guest memory does: only the import is freed.
    let mapping = Mapping::anonymous(IMPORTED);
    // SAFETY: the mapping stays mapped until the end of this function, after every command
    // buffer over it completed and the deallocator ran.
    let made = unsafe {
        no_copy(
            device,
            mapping.pointer(),
            IMPORTED,
            ResourceOptions::default(),
            Some(recording(&seen, ())),
        )
    };
    let member = match made {
        Ok(member) => member,
        Err(error) => {
            assert!(not_importable(&error), "{error:?}");
            println!(
                "FREE-IN-FLIGHT on {name}: AddressedImportReleased: the host import is refused"
            );
            return;
        }
    };
    let bytes = node(A);
    let contents = member.contents().expect("a host import's contents");
    // SAFETY: the import's first bytes, which no command reads yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    let walker = public_case(POINTER_WALKER).unwrap();
    let (binding, offset, length) = output(&walker);
    let state = pipeline_state(device, &walker).unwrap();
    let mut holding = CaseResources::new(device, &walker).unwrap();
    let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
        panic!("{POINTER_WALKER}: one argument-buffer buffer");
    };
    holding.argument_buffers.insert(field, member.clone());
    let buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
    // The encoder names no use of the import: the walk reaches it through the address alone.
    drop(holding.argument_buffers.remove(&field));
    let (busy_work, words) = busy.commit(device, queue);
    busy_work.scheduled();
    let walk = queue.command_buffer().unwrap();
    let mut encoder = walk.compute_command_encoder().unwrap();
    encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
    encoder.end_encoding().unwrap();
    walk.commit().unwrap();
    std::thread::scope(|scope| scope.spawn(move || drop(member)).join().unwrap());
    let freed = seen.runs.load(Ordering::SeqCst);
    let running = read_mark(&words) == 0;
    release(&words);
    busy_work.wait_until_completed_for(WAIT).unwrap();
    walk.wait_until_completed_for(WAIT).unwrap();
    let produced =
        (walk.error().is_none()).then(|| read_buffer(&buffers[&binding], offset, length).unwrap());
    println!(
        "FREE-IN-FLIGHT on {name}: AddressedImportReleased while the busy work ran ({}; {freed} \
         host imports freed): the busy work {:?}, error {:?}; the walk queued behind it {:?}, \
         error {:?}, output {produced:02x?}",
        match running {
            true => "still running after the release",
            false => "ended before the release returned",
        },
        busy_work.status(),
        busy_work.error(),
        walk.status(),
        walk.error()
    );
    let ran = eventually(|| seen.runs.load(Ordering::SeqCst) == 1);
    assert!(
        ran,
        "{name}: AddressedImportReleased: the import's deallocator never ran"
    );
    drop(mapping);
}
