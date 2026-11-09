//! A blit's buffers and textures held until its command buffer completed: a no-copy buffer that
//! a committed blit writes or reads, released by the program while the blit still waits behind
//! other work, keeps its import (its deallocator unrun) until the command buffer completed, and
//! the blit's bytes land in its pages. The consumer's `SynchronizeAndDiscardResources` on an RTX
//! 5080 copied a multisampled `Depth32Float` texture into a no-copy transfer buffer over its guest
//! pages, committed, waited until scheduled, released the transfer buffer and set the texture
//! `Volatile`: the per-sample copy (`sample_interleave.rs`) held neither the buffer nor the
//! texture, the import was freed 3 ms later with the copy in flight
//! (`VUID-vkDestroyBuffer-buffer-00922`), and the copy engine's write into it faulted (Xid 31),
//! where the next import reusing the address would have taken the write silently.
//!
//! Each leg encodes its blit before the work it waits behind is committed, and commits it after:
//! an encoding that outlasted that work (a per-sample depth copy's, on a Radeon 780M) would leave
//! nothing pending to hold. The harness (the host pages, the no-copy buffer over them, the busy
//! work) serves the render and compute encoders' legs too (`encoder_holds.rs`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::error::RuntimeGap;
use crate::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue,
    Device, DeviceKind, DeviceOptions, Error, HostMemory, Origin, PurgeableState, ResourceOptions,
    ShaderCacheOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::device::{HostPageTexturePolicy, PipelineWaitPolicy};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The longest a command buffer, or a release after it, may take.
pub(super) const WAIT: Duration = Duration::from_secs(60);

/// The no-copy buffer's bytes: whole pages of any host.
pub(super) const BYTES: usize = 4 << 20;

/// The fills that keep the queue busy before the blit: [`FILLS`] of [`FILLED`] bytes each.
const FILLS: u32 = 16;
pub(super) const FILLED: u64 = 64 << 20;

/// The word a source holds, the depth a depth source holds (0.5), and the byte a fill writes.
pub(super) const SOURCE: u32 = 0x5a5a_5a5a;
const DEPTH: u32 = 0x3f00_0000;
const FILL: u8 = 0xc3;

/// A scratch directory for the device's shader cache, removed when dropped.
pub(super) struct Scratch(pub(super) PathBuf);

impl Scratch {
    /// The directory of the leg `leg`.
    pub(super) fn new(leg: &str) -> Scratch {
        let name = format!("m2v-holds-{leg}-{}", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Page-aligned host memory of [`BYTES`], zeroed, freed when dropped: declared before the device
/// that uses it, so dropped after it.
pub(super) struct Host(NonNull<u8>);

impl Host {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, 64 << 10).unwrap()
    }

    pub(super) fn new() -> Host {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Host::layout()) };
        Host(NonNull::new(pointer).unwrap())
    }

    /// Its words.
    pub(super) fn words(&self) -> &[u32] {
        // SAFETY: `BYTES` bytes, allocated in `new` aligned past a word, which no command writes
        // any more.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast(), BYTES / 4) }
    }

    /// Whether every word is `value`.
    pub(super) fn holds(&self, value: u32) -> bool {
        self.words().iter().all(|&word| word == value)
    }

    /// Its pages, for a no-copy buffer or a rebind.
    ///
    /// # Safety
    ///
    /// They outlive every buffer over them, which alone use them while it lives.
    pub(super) unsafe fn memory(&self) -> HostMemory {
        // SAFETY: the caller's contract; `BYTES` bytes, allocated in `new` aligned to a page.
        unsafe { HostMemory::new(self.0) }
    }

    /// Set every word to `value`.
    pub(super) fn fill(&self, value: u32) {
        // SAFETY: as above; no command uses them now.
        let words = unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr().cast(), BYTES / 4) };
        words.fill(value);
    }

    /// Set the words from word `at` on to `words`.
    pub(super) fn set(&self, at: usize, words: &[u32]) {
        assert!(at + words.len() <= BYTES / 4);
        // SAFETY: the words are within the allocation (above), which no command uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr(),
                self.0.as_ptr().cast::<u32>().add(at),
                words.len(),
            )
        };
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Host::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Host::layout()) };
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
pub(super) fn device(cache: &Path) -> Device {
    Device::system_default_with(&options(cache)).unwrap()
}

/// As [`device`], its recordings waiting for a pipeline still being made until it is made
/// (`PipelineWaitPolicy::Exact`): for a test that holds a making in the device's pool and watches
/// the command buffer that awaits it not complete. Under the default `Bounded` wait, a recording
/// whose bound passed (its command buffer's commit plus `pipeline_commit_wait`, 4.5 s) drops the
/// draw or dispatch (`render/pending.rs`, `pending_dispatch.rs`) and its command buffer completes,
/// which a loaded host reaches while such a test still holds the making.
pub(super) fn exact_waits_device(cache: &Path) -> Device {
    let mut options = options(cache);
    options.policy.pipeline_waits = PipelineWaitPolicy::Exact;
    Device::system_default_with(&options).unwrap()
}

/// As [`device`], every texture over host pages a shadow (`HostPageTexturePolicy::Shadow`): for a
/// test whose subject is such a shadow (its copies, merges and runs), which under the default
/// policy a render target over host pages is only where the device's check did not pass for it.
pub(super) fn shadow_device(cache: &Path) -> Device {
    let mut options = options(cache);
    options.policy.host_page_textures = HostPageTexturePolicy::Shadow;
    Device::system_default_with(&options).unwrap()
}

/// The options of [`device`].
pub(super) fn options(cache: &Path) -> DeviceOptions {
    DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    }
}

/// A no-copy buffer over `host`, whose deallocator counts its runs in `runs`; `None`, printed,
/// where the device imports no host memory.
pub(super) fn over_host(device: &Device, host: &Host, runs: &Arc<AtomicUsize>) -> Option<Buffer> {
    let runs = Arc::clone(runs);
    // SAFETY: the memory outlives the buffer (dropped after the device), and only the buffer's
    // commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            Some(Box::new(move |_: NonNull<u8>, _: u64| {
                runs.fetch_add(1, Ordering::SeqCst);
            })),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("BLIT-HOLDS: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A command buffer of `queue`'s that fills `target` [`FILLS`] times, committed, once its batch's
/// `vkQueueSubmit` was made (a kernel submission counted, `WorkCounts::queue_submissions`; its
/// kernel times are no such sign, as `freed_memory.rs`'s `filling` says): the work the blit waits
/// behind.
pub(super) fn busy(queue: &CommandQueue, target: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for value in 0..FILLS {
        blit.fill_buffer(target, 0..FILLED, value as u8).unwrap();
    }
    blit.end_encoding().unwrap();
    let before = queue.device().work_counts().queue_submissions;
    command_buffer.commit().unwrap();
    let deadline = Instant::now() + WAIT;
    while queue.device().work_counts().queue_submissions == before {
        assert!(Instant::now() < deadline, "the batch was never submitted");
        std::thread::yield_now();
    }
    command_buffer
}

/// Wait until `done`, failing with `what` past [`WAIT`].
pub(super) fn until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// What a leg's blit does with the no-copy buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    /// A copy from a `Private` buffer into it.
    CopyInto,
    /// A copy from it into a `Private` buffer.
    CopyFrom,
    /// A fill of it.
    Fill,
    /// A copy from a multisampled `R32Uint` texture into it, per sample (`sample_interleave.rs`).
    OutOfMultisampled,
    /// A copy from it into a multisampled `R32Uint` texture, per sample.
    IntoMultisampled,
    /// A copy from a multisampled `Depth32Float` texture into it, per sample, the texture set
    /// `Volatile` after the release: the consumer's synchronize and discard.
    OutOfMultisampledDepth,
}

impl Leg {
    /// The multisampled texture's format, for a leg that copies with one.
    fn multisampled(self) -> Option<PixelFormat> {
        match self {
            Leg::OutOfMultisampled | Leg::IntoMultisampled => Some(PixelFormat::R32Uint),
            Leg::OutOfMultisampledDepth => Some(PixelFormat::Depth32Float),
            Leg::CopyInto | Leg::CopyFrom | Leg::Fill => None,
        }
    }

    /// The word its source holds.
    fn word(self) -> u32 {
        match self {
            Leg::OutOfMultisampledDepth => DEPTH,
            _ => SOURCE,
        }
    }
}

/// The multisampled texture's height and texel bytes; its width is what makes its samples' bytes
/// [`BYTES`] at the first sample count of [`MS_SAMPLES`] the device makes it at.
const MS_HEIGHT: u64 = 512;
const MS_TEXEL: u64 = 4;
const MS_SAMPLES: [u64; 2] = [4, 2];

/// A `Private` multisampled texture of `pixel_format` whose samples' bytes are [`BYTES`];
/// `None`, printed, where the device makes none.
fn multisampled(device: &Device, pixel_format: PixelFormat) -> Option<Texture> {
    for samples in MS_SAMPLES {
        let width = BYTES as u64 / (MS_HEIGHT * samples * MS_TEXEL);
        let mut descriptor = TextureDescriptor::texture_2d(pixel_format, width, MS_HEIGHT, false);
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = samples;
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        match device.new_texture(&descriptor) {
            Ok(texture) => return Some(texture),
            Err(error) => eprintln!("BLIT-HOLDS: no {samples}-sample {pixel_format:?}: {error}"),
        }
    }
    None
}

/// Copy `buffer`'s [`BYTES`] into `texture`'s samples (`into`), or its samples into `buffer`:
/// whether the device ran it (`false`, printed, where it refuses the emulation).
fn per_sample(
    blit: &mut BlitCommandEncoder,
    buffer: &Buffer,
    texture: &Texture,
    into: bool,
) -> bool {
    let (width, height) = (texture.width(), texture.height());
    let row = width * texture.sample_count() * MS_TEXEL;
    let (size, origin) = (Size::new(width, height, 1), Origin::new(0, 0, 0));
    let length = BYTES as u64;
    let copied = match into {
        true => blit.copy_from_buffer_to_texture(
            buffer,
            0,
            row,
            length,
            size,
            texture,
            0,
            0,
            origin,
            BlitOption::None,
        ),
        false => blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            origin,
            size,
            buffer,
            0,
            row,
            length,
            BlitOption::None,
        ),
    };
    match copied {
        Ok(()) => true,
        Err(Error::NotImplemented(RuntimeGap::MultisampledBufferCopy)) => {
            eprintln!("BLIT-HOLDS: the device refuses the per-sample copy");
            false
        }
        Err(error) => panic!("the per-sample copy: {error}"),
    }
}

/// Run `encode` in a command buffer of `queue`'s and wait for it: whether it encoded.
fn run_now(queue: &CommandQueue, encode: impl FnOnce(&mut BlitCommandEncoder) -> bool) -> bool {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    if !encode(&mut blit) {
        return false;
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    true
}

/// Run `leg`: the blit committed behind the busy work and scheduled, the no-copy buffer released
/// at once; its deallocator must not run before the blit's command buffer completed, and the
/// blit's bytes must land.
fn run(leg: Leg) {
    let scratch = Scratch::new(&format!("blit-{leg:?}"));
    let host = Host::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let length = BYTES as u64;
    let private = device
        .new_buffer(length, ResourceOptions::private())
        .unwrap();
    let texture = match leg.multisampled() {
        Some(pixel_format) => match multisampled(&device, pixel_format) {
            Some(texture) => Some(texture),
            None => return,
        },
        None => None,
    };
    // The source: the `Private` buffer, or the multisampled texture's every sample, holds the
    // leg's word, copied in from the host pages through a no-copy buffer of its own.
    if matches!(
        leg,
        Leg::CopyInto | Leg::OutOfMultisampled | Leg::OutOfMultisampledDepth
    ) {
        host.fill(leg.word());
        let Some(staged) = over_host(&device, &host, &Arc::new(AtomicUsize::new(0))) else {
            return;
        };
        let encoded = run_now(&queue, |blit| match &texture {
            Some(texture) => per_sample(blit, &staged, texture, true),
            None => blit
                .copy_from_buffer_to_buffer(&staged, 0, &private, 0, length)
                .is_ok(),
        });
        if !encoded {
            return;
        }
        drop(staged);
        host.fill(0);
    } else if matches!(leg, Leg::CopyFrom | Leg::IntoMultisampled) {
        host.fill(leg.word());
    }
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(no_copy) = over_host(&device, &host, &runs) else {
        return;
    };
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let ran = match (&texture, leg) {
        (None, Leg::CopyInto) => blit
            .copy_from_buffer_to_buffer(&private, 0, &no_copy, 0, length)
            .is_ok(),
        (None, Leg::CopyFrom) => blit
            .copy_from_buffer_to_buffer(&no_copy, 0, &private, 0, length)
            .is_ok(),
        (None, _) => blit.fill_buffer(&no_copy, 0..length, FILL).is_ok(),
        (Some(texture), Leg::IntoMultisampled) => per_sample(&mut blit, &no_copy, texture, true),
        (Some(texture), _) => per_sample(&mut blit, &no_copy, texture, false),
    };
    if !ran {
        return;
    }
    blit.end_encoding().unwrap();
    let busy = busy(&queue, &target);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_scheduled().unwrap();
    let pending = command_buffer.status() != CommandBufferStatus::Completed;
    drop(no_copy);
    if let (Some(texture), Leg::OutOfMultisampledDepth) = (&texture, leg) {
        texture
            .set_purgeable_state(PurgeableState::Volatile)
            .unwrap();
    }
    let at_release = runs.load(Ordering::SeqCst);
    let still = command_buffer.status() != CommandBufferStatus::Completed;
    eprintln!(
        "BLIT-HOLDS {leg:?} on {}: pending at the release {pending}, still after it {still}, \
         deallocator runs at the release {at_release}",
        device.name()
    );
    if pending && still {
        assert_eq!(
            at_release, 0,
            "{leg:?}: the no-copy buffer's import went while its blit's command buffer ran"
        );
    } else {
        eprintln!("BLIT-HOLDS {leg:?}: the blit completed before the release: nothing to hold");
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    busy.wait_until_completed_for(WAIT).unwrap();
    until("the deallocator never ran", || {
        runs.load(Ordering::SeqCst) == 1
    });
    match leg {
        Leg::CopyInto | Leg::OutOfMultisampled | Leg::OutOfMultisampledDepth => assert!(
            host.holds(leg.word()),
            "{leg:?}: the copy's bytes are not in the pages"
        ),
        Leg::Fill => assert!(
            host.holds(u32::from_ne_bytes([FILL; 4])),
            "the fill's bytes are not in the pages"
        ),
        Leg::CopyFrom | Leg::IntoMultisampled => {
            // Read the copy back through a second no-copy buffer over the same pages, cleared
            // first.
            host.fill(0);
            let back = over_host(&device, &host, &Arc::new(AtomicUsize::new(0))).unwrap();
            assert!(run_now(&queue, |blit| match &texture {
                Some(texture) => per_sample(blit, &back, texture, false),
                None => blit
                    .copy_from_buffer_to_buffer(&private, 0, &back, 0, length)
                    .is_ok(),
            }));
            drop(back);
            assert!(
                host.holds(leg.word()),
                "{leg:?}: the copy did not read the pages"
            );
        }
    }
}

#[test]
fn a_no_copy_buffer_a_committed_copy_writes_keeps_its_import_until_completion() {
    run(Leg::CopyInto);
}

#[test]
fn a_no_copy_buffer_a_committed_copy_reads_keeps_its_import_until_completion() {
    run(Leg::CopyFrom);
}

#[test]
fn a_no_copy_buffer_a_committed_fill_writes_keeps_its_import_until_completion() {
    run(Leg::Fill);
}

#[test]
fn a_no_copy_buffer_a_committed_per_sample_copy_writes_keeps_its_import_until_completion() {
    run(Leg::OutOfMultisampled);
}

#[test]
fn a_no_copy_buffer_a_committed_per_sample_copy_reads_keeps_its_import_until_completion() {
    run(Leg::IntoMultisampled);
}

#[test]
fn a_no_copy_buffer_a_committed_depth_copy_writes_keeps_its_import_past_a_volatile_texture() {
    run(Leg::OutOfMultisampledDepth);
}
