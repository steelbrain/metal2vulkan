//! Textures over host pages, in place: a 2D texture (or array) whose declared usage only samples
//! or reads it, over a no-copy buffer's host pages or over a buffer the runtime placed in fresh
//! host pages, is a linear image aliasing those pages, where the device's own check shows that the
//! one operation Vulkan lets touch bytes beside the texels leaves every byte alone.
//!
//! - **What Vulkan says.** An image bound to imported memory starts `UNDEFINED`
//!   (`VUID-VkImageCreateInfo-pNext-01443`: "If the pNext chain includes a
//!   VkExternalMemoryImageCreateInfo ... structure whose handleTypes member is not 0, initialLayout
//!   must be VK_IMAGE_LAYOUT_UNDEFINED"), and every image leaves the layout it is made in before
//!   any use: its first move, to `GENERAL` (`texture/first_move.rs`), an acquire from
//!   `VK_QUEUE_FAMILY_FOREIGN_EXT` for one over dma-buf pages (`dma_buf.rs`). Of that move the
//!   Synchronization chapter says "Image layout transitions may perform read and write accesses on
//!   all memory bound to the image subresource range", and "If the old layout is
//!   VK_IMAGE_LAYOUT_UNDEFINED, the contents of that range may be discarded". Metal keeps every
//!   CPU write to bytes no command writes, at every moment. No later operation of such an image
//!   touches a byte beside its texels: a shader's read touches its texels alone (the Memory Model:
//!   "A memory operation must not affect memory at a memory location not within its set of memory
//!   locations"), every command uses it in `GENERAL` with memory barriers, which move no layout
//!   (`tracker.rs`), and nothing renders into it or writes it through a shader (below).
//! - **Relying on the driver, checked.** The owner's ruling of 2026-10-26 ("In place, trusting the
//!   driver") takes the image where the device's own check shows the move leaving every byte
//!   alone: a deliberate, scoped exception to never relying on undocumented Vulkan behaviour,
//!   recorded in `docs/FAST-PATHS.md`. The check (`in_place_check.rs`) runs once per driver and
//!   release, per memory class ([`PagesClass`]), and its verdict is kept in the shader cache's
//!   content records ([`record_name`]); a class it did not pass keeps the shadow.
//! - **Which textures** ([`in_place`]): those whose declared usage (as Metal reports it, an
//!   `Unknown` one every usage) has none of `RenderTarget`, `ShaderWrite` and `ShaderAtomic`, under
//!   `HostPageTexturePolicy::InPlace`, over a class whose check passed. Plain Metal also writes
//!   through a texture beyond its declared usage, and renders into one made without
//!   `RenderTarget`: such a write binds the texture's usage shadow (`texture/usage_shadow.rs`), and
//!   such a pass renders through an attachment stand-in (`attachment_stand_in.rs`), since the
//!   image has neither storage nor attachment usage; the copies of both touch texels alone.
//!   Elsewhere the texture is a shadow (`shadow.rs`), its notice naming why ([`ShadowCause`]) and
//!   counted by cause (`WorkCounts::host_texture_shadows_*`).
//! - **Before the verdict.** A texture made before its class's verdict is known waits for nothing:
//!   it is a shadow ([`ShadowCause::CheckPending`], counted apart). The first such texture starts
//!   the check on a thread of its own; a device whose record holds the verdict knows it as it is
//!   made, and starts none.
//! - **Visibility.** A command reads the image's texels in the memory a buffer over the same pages
//!   reads: the pages' host-visible, host-coherent import (a dma-buf's own import is of the same
//!   memory type). So a CPU write reaches a command through the image as it reaches the buffer's
//!   commands: one the program made before the scheduler submits the command buffer's batch is in
//!   that submission's first access scope ("The first access scope includes all host writes to
//!   mappable device memory that are available to the host memory domain", Host Write Ordering
//!   Guarantees), and the commit submits nothing itself (`scheduler/mod.rs`), so a write made after
//!   the commit and before that submission is too.

use crate::device::lock;
use crate::shader_cache::store::{hex, Lookup};
use crate::Device;
use ash::vk;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::TextureUsage;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// The memory a texture over host pages lies in, each checked apart: the import and the image a
/// texture there takes differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PagesClass {
    /// A no-copy buffer's pages (or a no-copy heap's buffer's), imported by their pointer
    /// (`VK_EXT_external_memory_host`): a linear image made for host allocations, bound in the
    /// pages' import.
    Pointer,
    /// A no-copy buffer's pages imported as dma-bufs (`dma_buf.rs`): a `DRM_FORMAT_MOD_LINEAR`
    /// image bound to a dedicated import of a dma-buf of its own pages, its first move an acquire
    /// from `VK_QUEUE_FAMILY_FOREIGN_EXT`.
    DmaBuf,
    /// A buffer of the runtime's own in fresh anonymous pages it imported by their pointer
    /// (`fresh_pages.rs`): as [`Pointer`](Self::Pointer), in the memory type of the buffer's class.
    Fresh,
}

impl PagesClass {
    /// Every class, in the order the check runs them and the record holds their verdicts.
    pub(crate) const ALL: [PagesClass; 3] =
        [PagesClass::Pointer, PagesClass::Fresh, PagesClass::DmaBuf];

    /// Its place in [`ALL`](Self::ALL).
    pub(crate) fn index(self) -> usize {
        match self {
            PagesClass::Pointer => 0,
            PagesClass::Fresh => 1,
            PagesClass::DmaBuf => 2,
        }
    }

    /// The class in words, for a notice.
    pub(crate) fn words(self) -> &'static str {
        match self {
            PagesClass::Pointer => "host pages imported by their pointer",
            PagesClass::DmaBuf => "host pages imported as dma-bufs",
            PagesClass::Fresh => "fresh host pages the runtime imported for the buffer",
        }
    }
}

/// What the device's check answered for a class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Not known yet: no record held it, and the check has not answered.
    Pending,
    /// The first move of every image the check made left every byte alone.
    Passed,
    /// The first move of an image the check made changed a byte.
    Failed,
    /// The check made no image of the class: the device makes no such import or image.
    Unchecked,
}

impl Verdict {
    fn byte(self) -> u8 {
        match self {
            Verdict::Pending => 0,
            Verdict::Passed => 1,
            Verdict::Failed => 2,
            Verdict::Unchecked => 3,
        }
    }

    fn of_byte(byte: u8) -> Option<Verdict> {
        match byte {
            0 => Some(Verdict::Pending),
            1 => Some(Verdict::Passed),
            2 => Some(Verdict::Failed),
            3 => Some(Verdict::Unchecked),
            _ => None,
        }
    }
}

/// Why a texture over host pages is a shadow and not an image in place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShadowCause {
    /// The device's policy asks the shadow (`HostPageTexturePolicy::Shadow`).
    Exact,
    /// Its usage renders into it (`RenderTarget`, whatever else it holds; an `Unknown` usage,
    /// every usage, among them).
    RenderTarget,
    /// Its usage writes it through a shader (`ShaderWrite` or `ShaderAtomic`) and does not render
    /// into it.
    ShaderWrite,
    /// Its class's check has not answered yet.
    CheckPending(PagesClass),
    /// Its class's check found a byte changed.
    CheckFailed(PagesClass),
    /// Its class's check made no image.
    Unchecked(PagesClass),
    /// The device makes no image over its texels as Metal lays them out: why, and the words.
    Refused(Refusal, String),
}

/// Why the device makes no image in place over a texture's texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Its rows share bytes (an IOSurface plane's GPU rows): no image holds a texel twice.
    Rows,
    /// Its bytes were detached from their pages (`detach.rs`): no image is bound to them.
    Detached,
    /// The device makes no such image of the plan: its format, features, usage or limits, or an
    /// import only a dedicated allocation takes.
    Image,
    /// No image the device makes lays the rows (or slices) at the buffer's pitch.
    Pitch,
    /// The image's alignment, size or memory type misses the bytes' place.
    Placement,
    /// Over pages imported as dma-bufs: the caller's source made no one dma-buf of the texture's
    /// own pages (it failed, as a source of `udmabuf`s does where the pages are more runs of its
    /// file than one `udmabuf` takes, or answered with several chunks).
    Source,
    /// Over pages imported as dma-bufs: no `DRM_FORMAT_MOD_LINEAR` image over one dma-buf of the
    /// texture's own pages was made for another reason (the modifier's features or limits, the
    /// image's memory past the pages, or the driver's import).
    DmaBuf,
}

impl Refusal {
    /// Its place among `WorkCounts::host_texture_shadows_refused_*`.
    pub(crate) fn index(self) -> usize {
        match self {
            Refusal::Rows => 0,
            Refusal::Detached => 1,
            Refusal::Image => 2,
            Refusal::Pitch => 3,
            Refusal::Placement => 4,
            Refusal::Source => 5,
            Refusal::DmaBuf => 6,
        }
    }
}

thread_local! {
    /// This thread is the device's check's (`in_place_check.rs`): the textures it makes over host
    /// pages take the image in place whatever the verdict, and are neither counted nor announced.
    static CHECKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread is the device's check's ([`CHECKING`]).
pub(crate) fn checking() -> bool {
    CHECKING.with(std::cell::Cell::get)
}

/// Mark this thread as the device's check's, for good: it is a thread of its own.
pub(crate) fn mark_checking() {
    CHECKING.with(|checking| checking.set(true));
}

impl ShadowCause {
    /// The cause in words, for the `shadowed-texture` notice.
    pub(crate) fn words(&self) -> String {
        match self {
            ShadowCause::Exact => "the device's policy asks a shadow of every texture over host \
                                   pages (HostPageTexturePolicy::Shadow)"
                .to_owned(),
            ShadowCause::RenderTarget | ShadowCause::ShaderWrite => "its usage renders into it or \
                                     writes it (RenderTarget, ShaderWrite or ShaderAtomic), and \
                                     an image bound to host pages starts UNDEFINED \
                                     (VUID-VkImageCreateInfo-pNext-01443), its layout \
                                     transitions and a render pass's load and store operations \
                                     free to read and write every byte bound to it"
                .to_owned(),
            ShadowCause::CheckPending(class) => format!(
                "the device's check of the first move of an image over {} has not answered yet",
                class.words()
            ),
            ShadowCause::CheckFailed(class) => format!(
                "the device's check found the first move of an image over {} changing a byte \
                 the CPU wrote beside it",
                class.words()
            ),
            ShadowCause::Unchecked(class) => {
                format!("the device's check made no image over {}", class.words())
            }
            ShadowCause::Refused(_, why) => why.clone(),
        }
    }
}

/// Whether a texture over host pages of `class`, of Metal `usage` (as Metal reports it), on a
/// device of `policy`, is tried as an image in place; else why it is a shadow (the module docs).
/// `verdict` answers the device's check for the class, asked only of a texture the policy and
/// usage let through (a pending one starts the check).
pub(crate) fn in_place(
    policy: HostPageTexturePolicy,
    usage: TextureUsage,
    class: PagesClass,
    verdict: impl FnOnce(PagesClass) -> Verdict,
) -> Result<(), ShadowCause> {
    if policy == HostPageTexturePolicy::Shadow {
        return Err(ShadowCause::Exact);
    }
    if usage.contains(TextureUsage::RENDER_TARGET) {
        return Err(ShadowCause::RenderTarget);
    }
    if usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC) {
        return Err(ShadowCause::ShaderWrite);
    }
    match verdict(class) {
        Verdict::Passed => Ok(()),
        Verdict::Pending => Err(ShadowCause::CheckPending(class)),
        Verdict::Failed => Err(ShadowCause::CheckFailed(class)),
        Verdict::Unchecked => Err(ShadowCause::Unchecked(class)),
    }
}

/// A device's check verdicts (`DeviceShared::in_place`): one per class, and whether its check
/// started and settled.
pub(crate) struct InPlaceChecks {
    verdicts: [AtomicU8; 3],
    /// The check was started on its thread (at most once per device).
    started: AtomicBool,
    /// Every class's verdict is known: from the record, or the check ended.
    settled: Mutex<bool>,
    wake: Condvar,
    /// The device fails its check whatever the moves did (`DeviceOptions::fail_in_place_check`),
    /// and keeps no record of it.
    fail: bool,
    /// The check's run, from its start to its last verdict, in nanoseconds (0 before it ends).
    took: AtomicU64,
    /// What the check found changed, where a class failed: a line for each.
    notes: Mutex<Vec<String>>,
}

impl InPlaceChecks {
    /// No verdict yet; `fail` as [`InPlaceChecks::fail`] says.
    pub(crate) fn new(fail: bool) -> InPlaceChecks {
        InPlaceChecks {
            verdicts: [0, 0, 0].map(AtomicU8::new),
            started: AtomicBool::new(false),
            settled: Mutex::new(false),
            wake: Condvar::new(),
            fail,
            took: AtomicU64::new(0),
            notes: Mutex::new(Vec::new()),
        }
    }

    /// Note what the check found changed.
    pub(crate) fn note(&self, line: String) {
        lock(&self.notes).push(line);
    }

    /// `class`'s verdict now.
    pub(crate) fn verdict(&self, class: PagesClass) -> Verdict {
        let byte = self.verdicts[class.index()].load(Ordering::Acquire);
        Verdict::of_byte(byte).unwrap_or(Verdict::Pending)
    }

    /// Every class's verdict now.
    pub(crate) fn verdicts(&self) -> [Verdict; 3] {
        PagesClass::ALL.map(|class| self.verdict(class))
    }

    /// Give `class` its verdict.
    pub(crate) fn set(&self, class: PagesClass, verdict: Verdict) {
        self.verdicts[class.index()].store(verdict.byte(), Ordering::Release);
    }

    /// Whether the check is to fail whatever the moves did.
    pub(crate) fn forced_failure(&self) -> bool {
        self.fail
    }

    /// Note the check started; whether it had not been before.
    pub(crate) fn start(&self) -> bool {
        !self.started.swap(true, Ordering::AcqRel)
    }

    /// Every verdict is known now; the check took `took` where it ran.
    pub(crate) fn settle(&self, took: Option<Duration>) {
        if let Some(took) = took {
            let nanos = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
            self.took.store(nanos.max(1), Ordering::Release);
        }
        *lock(&self.settled) = true;
        self.wake.notify_all();
    }

    /// Wait until every verdict is known, or `timeout` passed: whether they are.
    pub(crate) fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut settled = lock(&self.settled);
        while !*settled {
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

    /// How long the check ran in this process, where it ran and ended.
    pub(crate) fn took(&self) -> Option<Duration> {
        match self.took.load(Ordering::Acquire) {
            0 => None,
            nanos => Some(Duration::from_nanos(nanos)),
        }
    }
}

/// The version of what the check does and of how a record of it is named and framed: a change to
/// either bumps it, and a record of another version is never read (its name differs, and so does
/// the version it holds).
pub(crate) const CHECK_VERSION: u32 = 1;

/// A record's first bytes.
const MAGIC: [u8; 8] = *b"M2VINPL\0";

/// The domain tag of a record's name.
const NAME_TAG: &[u8] = b"metal2vulkan in-place check record\0";

/// What the check checks, in a record's name: the first move of an image over host pages. A check
/// of another operation keeps its own record beside this one's, which it leaves valid.
const CHECK_KIND: &[u8] = b"first move\0";

/// The driver identity a verdict holds for: `VkPhysicalDeviceProperties`' vendor, device and
/// driver version, `VkPhysicalDeviceDriverProperties`' driver ID, name and info, and
/// `VkPhysicalDeviceIDProperties`' device UUID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DriverIdentity {
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    pub(crate) driver_version: u32,
    pub(crate) driver_id: i32,
    pub(crate) driver_name: String,
    pub(crate) driver_info: String,
    pub(crate) device_uuid: [u8; 16],
}

impl DriverIdentity {
    /// The SHA-256 of [`NAME_TAG`], [`CHECK_KIND`], [`CHECK_VERSION`], the runtime's
    /// `RUNTIME_VERSION` and every field: what names a record and what the record holds again.
    pub(crate) fn digest(&self) -> [u8; 32] {
        self.digest_for(NAME_TAG, CHECK_KIND, CHECK_VERSION)
    }

    /// The SHA-256 of `tag`, `kind`, `version`, the runtime's `RUNTIME_VERSION` and every field:
    /// what names the record of the check of `kind` at `version` (another device check's, as
    /// [`digest`](Self::digest) is this one's), and what that record holds again.
    pub(crate) fn digest_for(&self, tag: &[u8], kind: &[u8], version: u32) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(tag);
        hash.update(kind);
        hash.update(version.to_le_bytes());
        hash.update(metal2vulkan::cache::RUNTIME_VERSION.to_le_bytes());
        hash.update(self.vendor_id.to_le_bytes());
        hash.update(self.device_id.to_le_bytes());
        hash.update(self.driver_version.to_le_bytes());
        hash.update(self.driver_id.to_le_bytes());
        for text in [&self.driver_name, &self.driver_info] {
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        hash.update(self.device_uuid);
        hash.finalize().into()
    }
}

/// The name of the record of `identity`'s verdicts in the shader cache's content directory.
pub(crate) fn record_name(identity: &DriverIdentity) -> String {
    hex(&identity.digest())
}

/// The bytes of a record of `verdicts` for `identity`:
///
/// ```text
/// magic          8 bytes   "M2VINPL\0"
/// version        u32       CHECK_VERSION
/// runtime        u32       RUNTIME_VERSION
/// identity       32 bytes  DriverIdentity::digest
/// verdicts       3 bytes   one per class, in PagesClass::ALL's order
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian.
pub(crate) fn encode(identity: &DriverIdentity, verdicts: [Verdict; 3]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&CHECK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&metal2vulkan::cache::RUNTIME_VERSION.to_le_bytes());
    bytes.extend_from_slice(&identity.digest());
    bytes.extend(verdicts.map(Verdict::byte));
    let checksum: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// A record's length.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 + 3 + 32;

/// The verdicts `bytes` record for `identity`: `None` for anything but a whole record of this
/// version and runtime, for this identity, whose checksum holds and whose every verdict is an
/// answer (a pending one is never written).
pub(crate) fn decode(identity: &DriverIdentity, bytes: &[u8]) -> Option<[Verdict; 3]> {
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
    let (digest, rest) = rest.split_at(32);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC
        || word(version) != CHECK_VERSION
        || word(runtime) != metal2vulkan::cache::RUNTIME_VERSION
        || digest != identity.digest()
    {
        return None;
    }
    let verdicts = [rest[0], rest[1], rest[2]].map(Verdict::of_byte);
    match verdicts {
        [Some(a), Some(b), Some(c)] if ![a, b, c].contains(&Verdict::Pending) => Some([a, b, c]),
        _ => None,
    }
}

impl Device {
    /// The identity of the device's driver (`DriverIdentity`), read from its physical device.
    pub(crate) fn driver_identity(&self) -> DriverIdentity {
        let shared = self.shared();
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut ids = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut driver)
            .push_next(&mut ids);
        // SAFETY: the physical device is the instance's, at Vulkan 1.2 or later, where both
        // structures are core; the chain lives for the call.
        unsafe {
            shared
                .instance
                .raw
                .get_physical_device_properties2(shared.physical, &mut properties)
        };
        let base = properties.properties;
        let text = |chars: &[std::ffi::c_char]| {
            let bytes: Vec<u8> = chars
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        DriverIdentity {
            vendor_id: base.vendor_id,
            device_id: base.device_id,
            driver_version: base.driver_version,
            driver_id: driver.driver_id.as_raw(),
            driver_name: text(&driver.driver_name),
            driver_info: text(&driver.driver_info),
            device_uuid: ids.device_uuid,
        }
    }

    /// As the device is made: where its policy takes textures over host pages in place, its
    /// verdicts from the record of its driver's identity where one holds them (counted as
    /// recalled), settling them; else they wait for the check, which the first texture that asks
    /// starts. A record that does not decode is no record: the check runs, and its verdicts
    /// replace it.
    pub(crate) fn recall_in_place_checks(&self) {
        let shared = self.shared();
        if self.compile_device().policy().host_page_textures != HostPageTexturePolicy::InPlace {
            return;
        }
        if shared.in_place.forced_failure() {
            return;
        }
        let identity = self.driver_identity();
        let name = record_name(&identity);
        let looked = shared
            .shader_cache
            .content_store()
            .load_content(&name, |bytes| decode(&identity, &bytes));
        if let Lookup::Hit(verdicts) = looked {
            for (class, verdict) in PagesClass::ALL.into_iter().zip(verdicts) {
                shared.in_place.set(class, verdict);
                shared.counts.in_place_check_recalled();
            }
            shared.in_place.settle(None);
        }
    }

    /// `class`'s verdict for a texture asking it now: a pending one starts the check, unless it
    /// started already. Never waits.
    pub(crate) fn in_place_verdict(&self, class: PagesClass) -> Verdict {
        let verdict = self.shared().in_place.verdict(class);
        if verdict == Verdict::Pending {
            self.start_in_place_checks();
        }
        verdict
    }

    /// Start the device's check where its verdicts are pending and no check started, and wait
    /// until every verdict is known or `timeout` passed: whether they are. Under
    /// `HostPageTexturePolicy::Shadow` no check runs and no verdict is known: false at once. A
    /// runtime extension for the lanes that hold a texture over host pages to the path it takes.
    #[doc(hidden)]
    pub fn settle_in_place_checks(&self, timeout: Duration) -> bool {
        if self.compile_device().policy().host_page_textures != HostPageTexturePolicy::InPlace {
            return false;
        }
        let checks = &self.shared().in_place;
        if checks.verdicts().contains(&Verdict::Pending) {
            self.start_in_place_checks();
        }
        checks.wait(timeout)
    }

    /// How long the device's check ran in this process, where it ran and ended: `None` where its
    /// verdicts came from a record, or the check has not ended. A runtime extension for the lanes
    /// that measure it.
    #[doc(hidden)]
    pub fn in_place_check_took(&self) -> Option<Duration> {
        self.shared().in_place.took()
    }

    /// The device's check's verdicts now, one line per class, and what it found changed where a
    /// class failed. A runtime extension for the lanes that report it.
    #[doc(hidden)]
    pub fn in_place_check_report(&self) -> Vec<String> {
        let checks = &self.shared().in_place;
        let verdicts = PagesClass::ALL
            .into_iter()
            .map(|class| format!("{class:?}: {:?}", checks.verdict(class)));
        verdicts
            .chain(lock(&checks.notes).iter().cloned())
            .collect()
    }
}
