//! The device's check of what its GPU's linear images cost against its optimal ones, on a thread
//! of its own, once per driver, release and runtime (`Device::settle_tiling_cost_check`): a
//! measured answer for a consumer that keeps a GPU copy of host pages where the device's own
//! numbers say a texture in place over them costs it more than the copy saves (an x86 Mac's GPU
//! keeps such a copy of each IOSurface). The runtime makes no choice from it. Metal answers no
//! such question; Vulkan's queries do not either: a format's linear and optimal features say what
//! each tiling holds, not what it costs, and `VK_IMAGE_TILING_OPTIMAL` is only "for more
//! efficient memory access".
//!
//! - **What it measures.** Two arms of 2D `BGRA8Unorm` images of [`WIDTH`] x [`HEIGHT`] texels, a
//!   source and a target each: the **linear** arm's are the runtime's own textures over a no-copy
//!   buffer of host pages, imported by their pointer, in place (the image the product binds
//!   there: the driver's linear image, or `DRM_FORMAT_MOD_LINEAR`), the **optimal** arm's
//!   `Private` textures, as `Device::new_texture` makes them (the image's create flags, format
//!   list and usage the product's). Each arm's frame is one command buffer between two
//!   timestamps: a render pass over the whole target that loads and stores it, each fragment
//!   fetching the source's texel at its place, then a second such pass, each fragment writing
//!   zero (`tiling_check/draw.rs`). The lock costs are the optimal target copied into the host
//!   pages (`vkCmdCopyImageToBuffer`, the **download**) and the host pages copied into the optimal
//!   source (`vkCmdCopyBufferToImage`, the **upload**), each a command buffer between two
//!   timestamps. A round is the two frames, in turn first, and the two copies; [`ROUNDS`] rounds
//!   after one that warms the device, and each answer is the median of its rounds. A timestamp
//!   is `timestampPeriod` nanoseconds a tick; each command buffer starts behind a barrier of every
//!   stage, so its first timestamp follows the work before it.
//! - **Its verdict**: [`TilingVerdict::LinearCostly`] where the linear frame passes the optimal
//!   one by more than [`COSTLY_EXCESS`] and more than [`COSTLY_RATIO`] times
//!   ([`verdict_of`]), else [`TilingVerdict::LinearCheap`]; [`TilingVerdict::Unchecked`] where
//!   the device writes no timestamps the host reads, or answers GPU times from the host, imports
//!   no host pointer, makes no image in place over the pages or no optimal one, refuses any
//!   object or command, or its loader word was found overwritten. No driver, vendor or device
//!   identity enters it. Measured (medians of a run, linear against optimal frames, then the
//!   download and the upload, 2026-11-16): ANV on the Iris Xe of RPL-P (Mesa 26.2.6) 4.685
//!   against 0.890 ms (5.26 times), 2.109 against 0.402 (5.25) and 2.343 against 0.448 (5.23)
//!   in three runs, the first at a lower clock, downloads 0.53 to 1.18 ms and uploads 0.21 to
//!   0.25 ms, `LinearCostly` in each; lavapipe (llvmpipe, LLVM 23.1.1) 3.199 against 2.938 ms
//!   (1.09 times), a download of 1.448 ms and an upload of 1.435 ms, `LinearCheap`. The bounds
//!   lie between the two: the ratio's at 2, about the middle of 1.09 and 5.23 by ratio, and the
//!   excess at 500 us, below the Iris Xe's least (1.7 ms) and above lavapipe's (0.26 ms).
//! - **Its cost**: 24 MiB of host pages and 24 MiB of the device's memory while it runs, their
//!   textures, the draws' modules, pass and two pipelines, and a two-timestamp query pool; the
//!   GPU time of [`ROUNDS`] + 1 rounds (on the Iris Xe about 30 to 65 ms, the four medians'
//!   sum nine times); and the host's wait for each of the round's four internal submissions.
//!   Measured from its start to its answer: 78.2, 84.2 and 99.3 ms on the Iris Xe, 158.1 ms on
//!   lavapipe.
//! - **Kept.** A measured answer is written as one record in the shader cache's content directory
//!   (`shader_cache/store.rs`), named by the driver's identity, the check's version and the
//!   runtime's ([`record_name`]): its four medians, from which the verdict is worked out again as
//!   it is read. The device of the same driver and release that asks next reads it and runs no
//!   check. An unchecked answer is kept nowhere.
//! - **Asked, never run unasked.** The check starts at the first `settle_tiling_cost_check`, and
//!   no call waits for it but that one, for at most its timeout; `Device::tiling_cost` answers
//!   what is known now (pending until it answers).
//! - **Out of the device's way, and its thread**, as the in-place check's (`in_place_check.rs`):
//!   its thread is marked as a device check's, so its textures over host pages take the image in
//!   place whatever the in-place check's verdicts, neither counted nor announced; before each
//!   round it steps aside for the device's allocations and frees on other threads
//!   (`step_aside.rs`), at once while a call waits for its answer; each round's Vulkan work runs
//!   inside a section of the exit guard; and it holds its device through the registry of the
//!   runtime's holds (`device/holds.rs`), reading the stop before each round: the program's last
//!   drop waits for its round in progress and its releases, and the check ends unrecorded.

mod draw;

use crate::device::holds::Holder;
use crate::device::{lock, Device, WeakDevice};
use crate::error::{vulkan, Error};
use crate::gpu_time::GpuClock;
use crate::in_place::DriverIdentity;
use crate::shader_cache::store::{hex, Lookup};
use crate::{exit_guard, release, IdPlacement, ResourceOptions, Texture, TextureDescriptor};
use ash::vk;
use draw::TilingDraw;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// The width of the check's images, in texels: a generic size, no consumer's.
pub(crate) const WIDTH: u32 = 2048;

/// The height of the check's images, in texels.
pub(crate) const HEIGHT: u32 = 1536;

/// The bytes of one of the check's `BGRA8Unorm` images, rows tight.
const IMAGE_BYTES: u64 = WIDTH as u64 * HEIGHT as u64 * 4;

/// The measured rounds; one more warms the device first.
pub(crate) const ROUNDS: usize = 8;

/// How much longer the linear frame must take than the optimal one for
/// [`TilingVerdict::LinearCostly`], with [`COSTLY_RATIO`].
pub(crate) const COSTLY_EXCESS: Duration = Duration::from_micros(500);

/// How many times the optimal frame the linear one must take for
/// [`TilingVerdict::LinearCostly`], with [`COSTLY_EXCESS`].
pub(crate) const COSTLY_RATIO: f64 = 2.0;

/// What the device's tiling-cost check answered ([`TilingCost::verdict`]). No Metal counterpart:
/// Metal answers no such question (the check's own, `Device::settle_tiling_cost_check`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TilingVerdict {
    /// Not known yet: the check has not been asked, or has not answered.
    Pending,
    /// The linear frame took longer than the optimal one by more than 500 us and more than
    /// twice as long.
    LinearCostly,
    /// It did not.
    LinearCheap,
    /// No measurement: the device writes no timestamps the host reads, imports no host pointer,
    /// makes no image in place over host pages or no optimal one, refused an object or command,
    /// or the run did not end. [`Device::tiling_cost_check_report`] says why.
    Unchecked,
}

/// The device's tiling-cost check's answer ([`Device::tiling_cost`]): its verdict and the medians
/// it measured. No Metal counterpart: Metal answers no such question.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilingCost {
    /// The verdict.
    pub verdict: TilingVerdict,
    /// The linear arm's frame: a fetching pass and a zeroing pass over a 2048x1536 `BGRA8Unorm`
    /// target in place over host pages, from a source there. `None` where nothing was measured.
    pub linear: Option<Duration>,
    /// The optimal arm's frame: the same passes over `Private` textures.
    pub optimal: Option<Duration>,
    /// The optimal target copied into the host pages, whole (a download at a CPU lock).
    pub download: Option<Duration>,
    /// The host pages copied into the optimal source, whole (an upload after a CPU write).
    pub upload: Option<Duration>,
}

impl TilingCost {
    /// An answer of `verdict` with nothing measured.
    const fn unmeasured(verdict: TilingVerdict) -> TilingCost {
        TilingCost {
            verdict,
            linear: None,
            optimal: None,
            download: None,
            upload: None,
        }
    }
}

/// The medians one run measured, in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Medians {
    pub(crate) linear: u64,
    pub(crate) optimal: u64,
    pub(crate) download: u64,
    pub(crate) upload: u64,
}

impl Medians {
    /// The answer they give.
    pub(crate) fn cost(&self) -> TilingCost {
        TilingCost {
            verdict: verdict_of(self.linear, self.optimal),
            linear: Some(Duration::from_nanos(self.linear)),
            optimal: Some(Duration::from_nanos(self.optimal)),
            download: Some(Duration::from_nanos(self.download)),
            upload: Some(Duration::from_nanos(self.upload)),
        }
    }
}

/// The verdict of a linear frame of `linear` nanoseconds against an optimal one of `optimal`:
/// [`TilingVerdict::LinearCostly`] where the linear one passes the optimal one by more than
/// [`COSTLY_EXCESS`] and more than [`COSTLY_RATIO`] times, else [`TilingVerdict::LinearCheap`].
pub(crate) fn verdict_of(linear: u64, optimal: u64) -> TilingVerdict {
    let excess = Duration::from_nanos(linear.saturating_sub(optimal));
    let costly = excess > COSTLY_EXCESS && linear as f64 > optimal as f64 * COSTLY_RATIO;
    match costly {
        true => TilingVerdict::LinearCostly,
        false => TilingVerdict::LinearCheap,
    }
}

/// A device's tiling-cost check (`DeviceShared::tiling`): whether it started, its answer once
/// known with how it came, and the calls waiting for it.
pub(crate) struct TilingCheck {
    started: AtomicBool,
    settled: Mutex<Option<(TilingCost, String)>>,
    wake: Condvar,
    /// The calls waiting for the answer ([`TilingCheck::wait`]), for which no round steps aside
    /// (`step_aside.rs`).
    waiters: AtomicU32,
}

impl TilingCheck {
    /// A device's, not started.
    pub(crate) fn new() -> TilingCheck {
        TilingCheck {
            started: AtomicBool::new(false),
            settled: Mutex::new(None),
            wake: Condvar::new(),
            waiters: AtomicU32::new(0),
        }
    }

    /// Note the check started; whether it had not been before.
    fn start(&self) -> bool {
        !self.started.swap(true, Ordering::AcqRel)
    }

    /// The answer now: pending until it is known.
    fn cost(&self) -> TilingCost {
        (lock(&self.settled).as_ref()).map_or(
            TilingCost::unmeasured(TilingVerdict::Pending),
            |(cost, _)| *cost,
        )
    }

    /// Settle the answer, `how` saying how it came.
    fn settle(&self, cost: TilingCost, how: String) {
        *lock(&self.settled) = Some((cost, how));
        self.wake.notify_all();
    }

    /// Whether a call waits for the answer.
    fn awaited(&self) -> bool {
        self.waiters.load(Ordering::Acquire) > 0
    }

    /// Wait until the answer is known, or `timeout` passed: whether it is.
    fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        self.waiters.fetch_add(1, Ordering::AcqRel);
        let _waiting = Waiting(&self.waiters);
        let mut settled = lock(&self.settled);
        while settled.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            settled = match self.wake.wait_timeout(settled, left) {
                Ok((settled, _)) => settled,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        true
    }

    /// The answer in words.
    fn report(&self) -> String {
        let settled = lock(&self.settled);
        let Some((cost, how)) = settled.as_ref() else {
            return "verdict Pending".to_owned();
        };
        let ms = |d: Option<Duration>| d.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3);
        match (cost.linear, cost.optimal) {
            (Some(linear), Some(optimal)) => format!(
                "verdict {:?}: linear {:.3} ms, optimal {:.3} ms ({:.2}x), download {:.3} ms, \
                 upload {:.3} ms (medians of {ROUNDS} rounds, {WIDTH}x{HEIGHT} BGRA8Unorm); {how}",
                cost.verdict,
                ms(Some(linear)),
                ms(Some(optimal)),
                linear.as_secs_f64() / optimal.as_secs_f64().max(1e-9),
                ms(cost.download),
                ms(cost.upload),
            ),
            _ => format!("verdict {:?}: {how}", cost.verdict),
        }
    }
}

/// A call waiting for the answer ([`TilingCheck::wait`]), counted until it returns.
struct Waiting<'a>(&'a AtomicU32);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Device {
    /// No Metal counterpart (Metal answers no such question): start the device's tiling-cost
    /// check where it has not started, and wait until its answer is known or `timeout` passed:
    /// whether it is. The check measures, on the device's GPU, a frame of two full-target passes
    /// over 2048x1536 `BGRA8Unorm` linear images in place over host pages against the same frame
    /// over `Private` (optimal) ones, and the copies between the optimal image and the host pages
    /// both ways, and answers whether the linear frame costs the GPU more
    /// ([`TilingVerdict::LinearCostly`]: more than 500 us and twice the optimal one), with the
    /// medians ([`Device::tiling_cost`]). It runs on a thread of its own, about 50 MiB of host
    /// and device memory while it runs, and its measured answer is kept in the shader cache's
    /// directory for the driver, its release and the runtime's, so a later device of the same
    /// reads it at once. Only this call starts it, and only this call waits for it. The runtime
    /// makes no choice from the answer.
    pub fn settle_tiling_cost_check(&self, timeout: Duration) -> bool {
        self.current().settle_tiling_cost_check_now(timeout)
    }

    /// [`Device::settle_tiling_cost_check`] on the generation that serves (`device/line.rs`).
    pub(crate) fn settle_tiling_cost_check_now(&self, timeout: Duration) -> bool {
        self.start_tiling_cost_check();
        self.shared().tiling.wait(timeout)
    }

    /// No Metal counterpart: the device's tiling-cost check's answer now
    /// ([`Device::settle_tiling_cost_check`]): [`TilingVerdict::Pending`] with nothing measured
    /// until it is known.
    pub fn tiling_cost(&self) -> TilingCost {
        self.current().shared().tiling.cost()
    }

    /// No Metal counterpart: the device's tiling-cost check's answer in words: its verdict, the
    /// four medians where it measured them (the linear and optimal frames, their ratio, the
    /// download and the upload), and whether it ran in this process (how long) or was read from
    /// the shader cache's record; why, where it is unchecked.
    pub fn tiling_cost_check_report(&self) -> String {
        self.current().shared().tiling.report()
    }

    /// Start the check where it has not started: its answer from the record where one holds it,
    /// else its thread. Where the thread cannot start, the answer is unchecked.
    fn start_tiling_cost_check(&self) {
        let shared = self.shared();
        let check = &shared.tiling;
        if !check.start() {
            return;
        }
        let identity = self.driver_identity();
        let looked = (shared.shader_cache.content_store())
            .load_content(&record_name(&identity), |bytes| decode(&identity, &bytes));
        if let Lookup::Hit(medians) = looked {
            check.settle(
                medians.cost(),
                "read from the shader cache's record".to_owned(),
            );
            return;
        }
        let weak = self.downgrade();
        let builder = std::thread::Builder::new().name("m2v-tiling-check".to_owned());
        if exit_guard::spawn(builder, move || run(weak)).is_err() {
            let unchecked = TilingCost::unmeasured(TilingVerdict::Unchecked);
            check.settle(unchecked, "its thread did not start".to_owned());
        }
    }
}

/// The check's thread: the run, then its answer settled and, where measured, recorded.
fn run(weak: WeakDevice) {
    crate::in_place::mark_checking();
    crate::step_aside::mark_a_check_s_thread(crate::step_aside::Counted::AsTheDevice);
    let began = Instant::now();
    let Some(device) = weak.hold() else {
        return;
    };
    let Some(measured) = measure(&device) else {
        // The process is exiting, or the program let the device go: nothing is kept.
        return;
    };
    let check = &device.shared().tiling;
    match measured {
        Ok(medians) => {
            // A failed write, or a store with no directory, leaves the answer to the device's
            // life alone.
            let store = device.shared().shader_cache.content_store();
            if store.has_directory() {
                let identity = device.driver_identity();
                let _ =
                    store.write_record_soon(&record_name(&identity), encode(&identity, &medians));
            }
            let took = began.elapsed().as_secs_f64() * 1e3;
            check.settle(medians.cost(), format!("ran {took:.1} ms in this process"));
        }
        Err(why) => {
            let unchecked = TilingCost::unmeasured(TilingVerdict::Unchecked);
            check.settle(unchecked, why);
        }
    }
}

/// What one command buffer of a round measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Linear,
    Optimal,
    Download,
    Upload,
}

/// The run's medians; why not, where it measured none; `None` where the process is exiting or the
/// program's last drop stopped the device's holds.
fn measure(device: &Device) -> Option<Result<Medians, String>> {
    let shared = device.shared();
    let clock = match (shared.timing.clock(), shared.timing.answers_from_host()) {
        (Some(clock), false) => *clock,
        _ => {
            let why = "the device writes no timestamps the host reads, or answers GPU times from \
                       the host";
            return Some(Err(why.to_owned()));
        }
    };
    if word_overwritten(device) {
        let why = "the device's loader word was found overwritten (the driver names the device in \
                   its messages)";
        return Some(Err(why.to_owned()));
    }
    let made = {
        let _busy = exit_guard::enter()?;
        Arms::make(device)
    };
    let arms = match made {
        Ok(arms) => arms,
        Err(why) => return Some(Err(why)),
    };
    let measured = rounds(device, &clock, &arms);
    // The arms' textures and pages go inside a section of their own.
    let busy = exit_guard::enter();
    drop(arms);
    drop(busy);
    measured
}

/// The rounds over `arms`, as [`measure`] answers them; their draws and query pool made for them
/// and released after them.
fn rounds(device: &Device, clock: &GpuClock, arms: &Arms) -> Option<Result<Medians, String>> {
    let made = {
        let _busy = exit_guard::enter()?;
        TilingDraw::new(device, arms.format)
            .and_then(|draw| Stamps::new(device).map(|stamps| (draw, stamps)))
    };
    let (draw, stamps) = match made {
        Ok(made) => made,
        Err(error) => return Some(Err(format!("its draws were not made: {error}"))),
    };
    let extent = [WIDTH, HEIGHT];
    let targets = {
        let _busy = exit_guard::enter()?;
        (draw.target(arms.linear_images, arms.format, extent)).and_then(|linear| {
            (draw.target(arms.optimal_images, arms.format, extent)).map(|optimal| (linear, optimal))
        })
    };
    let (linear, optimal) = match targets {
        Ok(targets) => targets,
        Err(error) => return Some(Err(format!("its views were not made: {error}"))),
    };
    let (pages, at) = arms.pages_buffer;
    let region = vk::BufferImageCopy {
        buffer_offset: at,
        buffer_row_length: WIDTH,
        buffer_image_height: HEIGHT,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width: WIDTH,
            height: HEIGHT,
            depth: 1,
        },
    };
    let (optimal_source, optimal_target) = arms.optimal_images;
    let general = vk::ImageLayout::GENERAL;
    let mut times: [Vec<u64>; 4] = Default::default();
    for round in 0..=ROUNDS {
        if !device.holds_go_on(Holder::TilingCheck) {
            return None;
        }
        step_aside(device);
        if word_overwritten(device) {
            let why = "the device's loader word was found overwritten during its rounds";
            return Some(Err(why.to_owned()));
        }
        let frames = match round % 2 {
            0 => [Step::Linear, Step::Optimal],
            _ => [Step::Optimal, Step::Linear],
        };
        for step in frames.into_iter().chain([Step::Download, Step::Upload]) {
            let _busy = exit_guard::enter()?;
            let timed = stamps.time(device, clock, |raw, command_buffer| {
                // SAFETY: the internal command buffer is recording outside a render pass and only
                // this records it; the images and the pages' buffer are the arms', live until the
                // submission, waited for, completed, and rest in `GENERAL` (each image's first
                // move made); the arms' objects live until they are dropped, after the rounds.
                unsafe {
                    match step {
                        Step::Linear => linear.record(raw, command_buffer),
                        Step::Optimal => optimal.record(raw, command_buffer),
                        Step::Download => raw.cmd_copy_image_to_buffer(
                            command_buffer,
                            optimal_target,
                            general,
                            pages,
                            &[region],
                        ),
                        Step::Upload => raw.cmd_copy_buffer_to_image(
                            command_buffer,
                            pages,
                            optimal_source,
                            general,
                            &[region],
                        ),
                    }
                }
            });
            match timed {
                Ok(nanos) if round > 0 => times[step as usize].push(nanos),
                Ok(_) => {}
                Err(error) => return Some(Err(format!("its {step:?} round failed: {error}"))),
            }
        }
    }
    let [linear, optimal, download, upload] = times.map(|mut times| median(&mut times));
    Some(Ok(Medians {
        linear,
        optimal,
        download,
        upload,
    }))
}

/// The median of `times`: the upper of the two middle ones where they are even; 0 where there is
/// none.
pub(crate) fn median(times: &mut [u64]) -> u64 {
    times.sort_unstable();
    times.get(times.len() / 2).copied().unwrap_or(0)
}

/// Before a round, step aside for the device's allocations and frees on other threads
/// (`step_aside.rs`): not while the program's last drop stops the check or a call waits for its
/// answer. Counted where it waited.
fn step_aside(device: &Device) {
    let shared = device.shared();
    let activity = shared.allocations.allocations.activity();
    let stop = || shared.holds.stopping() || shared.tiling.awaited();
    if let Some(waited) = activity.step_aside(stop) {
        shared.counts.check_stepped_aside(waited);
    }
}

/// Whether the runtime found `device`'s loader word overwritten (`loader_word.rs`): the check
/// then runs no round, as the in-place check runs none.
fn word_overwritten(device: &Device) -> bool {
    // SAFETY: the device is live while `device` is.
    unsafe { device.shared().loader_word.overwritten() }
}

/// The two arms' textures and their images, the host pages the linear arm's lie over, and where
/// the pages' first byte lies in their `VkBuffer`.
struct Arms {
    /// The linear arm's source and target, over `pages`; dropped before them.
    _linear: [Texture; 2],
    /// The optimal arm's source and target.
    _optimal: [Texture; 2],
    _pages: crate::in_place_check::ClassPages,
    format: vk::Format,
    linear_images: (vk::Image, vk::Image),
    optimal_images: (vk::Image, vk::Image),
    pages_buffer: (vk::Buffer, u64),
}

impl Arms {
    /// The arms on `device`, inside a section of the exit guard; why not, where the device makes
    /// no such image or pages.
    fn make(device: &Device) -> Result<Arms, String> {
        let pages = crate::in_place_check::pointer_pages(device, 2 * IMAGE_BYTES)
            .map_err(|unmade| format!("no host pages for its linear images: {unmade}"))?;
        let descriptor = |usage: TextureUsage, options: ResourceOptions| {
            let mut descriptor = TextureDescriptor::texture_2d(
                PixelFormat::BGRA8Unorm,
                u64::from(WIDTH),
                u64::from(HEIGHT),
                false,
            );
            descriptor.usage = usage;
            descriptor.resource_options = options;
            descriptor
        };
        let source = TextureUsage::SHADER_READ;
        let target = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let row = u64::from(WIDTH) * 4;
        let over = |usage, offset| {
            let descriptor = descriptor(usage, ResourceOptions::shared());
            (pages.buffer)
                .new_runtime_texture("new_texture", &descriptor, offset, row)
                .map_err(|error| format!("no linear image in place over host pages: {error}"))
        };
        let linear = [over(source, 0)?, over(target, IMAGE_BYTES)?];
        // The runtime's own: no resource ID, so the program's IDs are Metal's whether or when
        // the check runs (a texture over a buffer the runtime makes takes none either).
        let private = |usage| {
            let descriptor = TextureDescriptor {
                placement: IdPlacement::Outside,
                ..descriptor(usage, ResourceOptions::private())
            };
            (device.new_texture(&descriptor)).map_err(|error| format!("no optimal image: {error}"))
        };
        let optimal = [private(source)?, private(target)?];
        let image = |texture: &Texture, in_place: bool| {
            let (image, format, aliases) = match texture.tiling_check_image() {
                Some(Ok(made)) => made,
                Some(Err(error)) => return Err(format!("its image's first move failed: {error}")),
                None => return Err("its texture is a shadow, not an image".to_owned()),
            };
            match aliases == in_place {
                true => Ok((image, format)),
                false if in_place => Err("its texture over host pages lies in no image in \
                                          place"
                    .to_owned()),
                false => Err("its Private texture lies over bytes, not in an image".to_owned()),
            }
        };
        let (linear_source, format) = image(&linear[0], true)?;
        let (linear_target, target_format) = image(&linear[1], true)?;
        let (optimal_source, source_format) = image(&optimal[0], false)?;
        let (optimal_target, last_format) = image(&optimal[1], false)?;
        if [target_format, source_format, last_format] != [format; 3] {
            return Err(format!(
                "its images hold other formats: VkFormat {}, {}, {} and {}",
                format.as_raw(),
                target_format.as_raw(),
                source_format.as_raw(),
                last_format.as_raw()
            ));
        }
        let pages_buffer = pages.buffer.at(0);
        if !pages_buffer.1.is_multiple_of(4) {
            return Err("its pages' first byte lies off a texel of their VkBuffer".to_owned());
        }
        Ok(Arms {
            _linear: linear,
            _optimal: optimal,
            _pages: pages,
            format,
            linear_images: (linear_source, linear_target),
            optimal_images: (optimal_source, optimal_target),
            pages_buffer,
        })
    }
}

/// The check's query pool of two timestamps, which each timed command buffer resets and writes.
pub(crate) struct Stamps {
    device: Device,
    pool: vk::QueryPool,
}

impl Stamps {
    /// The pool on `device`; inside a section of the exit guard.
    pub(crate) fn new(device: &Device) -> Result<Stamps, Error> {
        let raw: &ash::Device = device.shared().vk();
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);
        // SAFETY: `info` is live for the call.
        let pool = unsafe { raw.create_query_pool(&info, None) }.map_err(vulkan)?;
        Ok(Stamps {
            device: device.clone(),
            pool,
        })
    }

    /// The nanoseconds between the timestamps around what `record` records, in an internal
    /// submission of `device`'s waited for ([`record_around`](Self::record_around)); inside a
    /// section of the exit guard.
    pub(crate) fn time(
        &self,
        device: &Device,
        clock: &GpuClock,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<u64, Error> {
        device.submit_internal(|commands, command_buffer| {
            self.record_around(commands, command_buffer, record);
        })?;
        self.read(device, clock)
    }

    /// [`time`](Self::time), its submission made only where no batch of `device`'s is in flight
    /// (`Device::submit_internal_if_idle`): `None` where one was, and nothing was submitted.
    pub(crate) fn time_if_idle(
        &self,
        device: &Device,
        clock: &GpuClock,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<Option<u64>, Error> {
        let made = device.submit_internal_if_idle(|commands, command_buffer| {
            self.record_around(commands, command_buffer, record);
        })?;
        match made {
            true => self.read(device, clock).map(Some),
            false => Ok(None),
        }
    }

    /// Record into `command_buffer`, an internal command buffer recording outside a render pass:
    /// the pool's queries reset, a barrier of every stage behind the work before, the first
    /// timestamp at the top of the pipe, what `record` records, the second at its bottom.
    fn record_around(
        &self,
        raw: &ash::Device,
        command_buffer: vk::CommandBuffer,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) {
        let pool = self.pool;
        let all = vk::PipelineStageFlags::ALL_COMMANDS;
        let before = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        // SAFETY: the internal command buffer is recording outside a render pass; the pool is
        // live, of two timestamp queries, which nothing pending uses (each submission is waited
        // for).
        unsafe {
            raw.cmd_reset_query_pool(command_buffer, pool, 0, 2);
            let flags = vk::DependencyFlags::empty();
            raw.cmd_pipeline_barrier(command_buffer, all, all, flags, &before, &[], &[]);
            let top = vk::PipelineStageFlags::TOP_OF_PIPE;
            raw.cmd_write_timestamp(command_buffer, top, pool, 0);
        }
        record(raw, command_buffer);
        // SAFETY: as above.
        unsafe {
            let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
            raw.cmd_write_timestamp(command_buffer, bottom, pool, 1);
        }
    }

    /// The nanoseconds between the pool's two timestamps, written by a submission that completed.
    fn read(&self, device: &Device, clock: &GpuClock) -> Result<u64, Error> {
        let raw: &ash::Device = device.shared().vk();
        let mut stamps = [0u64; 2];
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        // SAFETY: the pool is live, its two queries written by the submission, which completed.
        unsafe { raw.get_query_pool_results(self.pool, 0, &mut stamps, flags) }.map_err(vulkan)?;
        Ok(clock.duration(stamps[0], stamps[1]))
    }
}

impl Drop for Stamps {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard.
        let Some(_busy) = exit_guard::enter() else {
            // The process is exiting: the pool is left to its end.
            return;
        };
        let raw: &ash::Device = self.device.shared().vk();
        // SAFETY: every submission that used the pool completed (each was waited for).
        unsafe { release::object(raw, self.pool) };
    }
}

/// The version of what the check measures and of how its record is named and framed: a change to
/// either bumps it, and a record of another version is never read.
pub(crate) const CHECK_VERSION: u32 = 1;

/// A record's first bytes.
const MAGIC: [u8; 8] = *b"M2VTILE\0";

/// The domain tag of a record's name.
const NAME_TAG: &[u8] = b"metal2vulkan tiling-cost check record\0";

/// What the check measures, in a record's name.
const CHECK_KIND: &[u8] = b"linear against optimal\0";

/// A record's length: magic, version, runtime, identity, four medians, checksum.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 + 4 * 8 + 32;

/// The digest of `identity` that names its record and that the record holds again.
fn digest(identity: &DriverIdentity) -> [u8; 32] {
    identity.digest_for(NAME_TAG, CHECK_KIND, CHECK_VERSION)
}

/// The name of the record of `identity`'s answer in the shader cache's content directory.
pub(crate) fn record_name(identity: &DriverIdentity) -> String {
    hex(&digest(identity))
}

/// The bytes of a record of `medians` for `identity`:
///
/// ```text
/// magic          8 bytes   "M2VTILE\0"
/// version        u32       CHECK_VERSION
/// runtime        u32       RUNTIME_VERSION
/// identity       32 bytes  the identity's digest for this check
/// linear         u64       the linear frame's median, in nanoseconds
/// optimal        u64       the optimal frame's
/// download       u64       the download's
/// upload         u64       the upload's
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian.
pub(crate) fn encode(identity: &DriverIdentity, medians: &Medians) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&CHECK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&metal2vulkan::cache::RUNTIME_VERSION.to_le_bytes());
    bytes.extend_from_slice(&digest(identity));
    for nanos in [
        medians.linear,
        medians.optimal,
        medians.download,
        medians.upload,
    ] {
        bytes.extend_from_slice(&nanos.to_le_bytes());
    }
    let checksum: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The medians `bytes` record for `identity`: `None` for anything but a whole record of this
/// version and runtime, for this identity, whose checksum holds and whose frames are not zero.
pub(crate) fn decode(identity: &DriverIdentity, bytes: &[u8]) -> Option<Medians> {
    if bytes.len() != RECORD_BYTES {
        return None;
    }
    let (body, checksum) = bytes.split_at(RECORD_BYTES - 32);
    let computed: [u8; 32] = Sha256::digest(body).into();
    if computed[..] != *checksum {
        return None;
    }
    let (magic, rest) = body.split_at(8);
    let (version, rest) = rest.split_at(4);
    let (runtime, rest) = rest.split_at(4);
    let (identity_digest, rest) = rest.split_at(32);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC
        || word(version) != CHECK_VERSION
        || word(runtime) != metal2vulkan::cache::RUNTIME_VERSION
        || identity_digest != digest(identity)
    {
        return None;
    }
    let nanos = |at: usize| Some(u64::from_le_bytes(rest.get(at..at + 8)?.try_into().ok()?));
    let medians = Medians {
        linear: nanos(0)?,
        optimal: nanos(8)?,
        download: nanos(16)?,
        upload: nanos(24)?,
    };
    (medians.linear > 0 && medians.optimal > 0).then_some(medians)
}
