//! The runtime's typed errors.
//!
//! Where Metal raises, aborts, or answers nil, the runtime returns one of these, and a failed
//! command records nothing. Execution status words are not errors: they are reported on the command
//! buffer ([`CommandBuffer::execution_report`](super::CommandBuffer::execution_report)).
//!
//! **A misuse ends Metal's process, or Metal answers nil**. Two kinds tell a program's misuses
//! apart by what plain Metal does with them, and both name the call and its [`Problem`]:
//!
//! - [`Error::Invalid`]: Metal's validation ends the process (its descriptor validation, which
//!   plain Metal runs too, or its API validation layer where plain Metal's answer is not one the
//!   runtime gives). A program that would run on Metal never meets one.
//! - [`Error::Nil`]: plain Metal answers nil (no object, no error), and the process goes on;
//!   its API validation layer may stop the call too. A program may meet one and carry on, as on
//!   Metal.
//!
//! The same problem can be either, by the call and the descriptor: a 2D multisample texture of a
//! two-plane YCbCr format ends the process, and the 2D one answers nil, both
//! [`TextureProblem`]s. Where Metal answers nil through a kind of its own the runtime keeps that
//! kind: no memory ([`Error::OutOfMemory`], an automatic heap without room among it), a pinned
//! address that cannot be honoured ([`Error::PlacementConflict`]), no device
//! ([`Error::NoDevice`]), a library or pipeline Metal answers nil with an `NSError` for
//! ([`Error::Compile`]).

use super::address::PlacementConflict;
use super::address_map::{AddressModeMap, MappedSampling};
use super::apple_only::AppleOnly;
use super::argument_encoder::ArgumentEncoderProblem;
use super::copy_plan::CopyProblem;
use super::function_table::FunctionTableProblem;
use super::heap::HeapProblem;
use super::indirect_command_buffer::IndirectCommandBufferProblem;
use super::interop::SemaphoreProblem;
use super::linear_plan::BufferTextureProblem;
use super::mipmap_plan::MipmapProblem;
use super::pipeline::{BinaryFunctionProblem, RequiredThreadsProblem};
use super::rate_map::RateMapProblem;
use super::rate_mapped_pass::RegionRefusal;
use super::rebind::RebindProblem;
use super::render::DrawProblem;
use super::render_pass::{Attachment, RenderPassProblem};
use super::render_state::RenderStateProblem;
use super::sparse::SparseProblem;
use super::texture::WrappedImageProblem;
use super::texture_descriptor::TextureProblem;
use super::view_plan::ViewProblem;
use super::visibility::VisibilityProblem;
use super::wrapped_buffer::WrappedBufferProblem;
use metal2vulkan::bind::UnpinnedCompile;
use metal2vulkan::device::{ApiVersion, VkExtension, VkFeature};
use metal2vulkan::error::{BindError, Error as CompileError, Unsupported};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::Stage;
use metal2vulkan::reflection::LinkedUseGap;
use std::fmt;
use std::path::PathBuf;

/// Why a runtime call failed: where Metal raises, ends the process, answers nil or an `NSError`,
/// the runtime returns one of these.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// What Metal's validation stops, ending the process: the call and the problem (the
    /// module's docs).
    Invalid(Invalid),
    /// A device or translator gap, naming what is missing.
    Unsupported(Unsupported),
    /// The planner refused the encoder's bind state for this command.
    Bind(BindError),
    /// Library or pipeline creation failed: refused, over budget, or cancelled.
    Compile(CompileError),
    /// A runtime piece a later build step adds: the command needs it, so it is refused rather than
    /// run without it.
    NotImplemented(RuntimeGap),
    /// A pinned GPU address the device's Metal address space cannot honour (a runtime extension):
    /// its range meets bytes another resource holds, or leaves the space. Metal answers nil for
    /// both, and aliases nothing.
    PlacementConflict(PlacementConflict),
    /// The device ran out of memory of this kind.
    OutOfMemory(MemoryKind),
    /// The Vulkan device was lost.
    DeviceLost,
    /// Any other Vulkan failure, raw.
    Vulkan(VkResultCode),
    /// No Vulkan device the runtime can use: the instance found no driver, or no physical device
    /// qualified (`MTLCreateSystemDefaultDevice` answering nil). A Vulkan loader that did not
    /// open is [`Error::LoaderNotFound`].
    NoDevice,
    /// An Apple-only call off Metal's default that the runtime refuses, never a misuse:
    /// [`AppleOnly::gap`](crate::AppleOnly::gap) says why, whether Metal's answer is not measured
    /// or the runtime has not built it. Each is the runtime's own gap, not a device's.
    AppleOnly(AppleOnly),
    /// The shader cache directory cannot be used: only device creation and adoption fail so; later,
    /// a failed read or write is a counted miss.
    ShaderCache(ShaderCacheError),
    /// A binary archive that cannot be loaded or serialized (`MTLBinaryArchiveError`).
    BinaryArchive(BinaryArchiveError),
    /// A creation under `PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS` whose pipeline no listed
    /// binary archive holds.
    BinaryArchiveMiss(BinaryArchiveMiss),
    /// A rasterization-rate map Metal's API validation takes that the compile layer does not
    /// compile: this crate's bound, where Apple's compile is unmeasured or crashes.
    RateMap(UnpinnedCompile),
    /// A command buffer wait that would deadlock (`waitUntilCompleted`, `waitUntilScheduled`, or
    /// `commandBuffer` on a queue whose command buffers are all held): the runtime refuses it at
    /// once rather than hang. The cause says which of Metal's answers the refusal stands for: where
    /// Metal never returns ([`DeadlockCause::OwnQueue`]), where it would hang too
    /// ([`DeadlockCause::Cycle`]), where Metal's handler threads run out and it hangs
    /// ([`DeadlockCause::HandlerThreadCap`]), once the process is exiting
    /// ([`DeadlockCause::Exiting`]), and from a submission handler, which Metal has not
    /// ([`DeadlockCause::SubmissionHandler`]).
    WaitWouldDeadlock(DeadlockCause),
    /// An instance API version device creation cannot request
    /// (`DeviceOptions::instance_api_version`): below Vulkan 1.2, the runtime's floor, or above
    /// the version the Vulkan loader reports.
    InstanceApiVersion(InstanceApiVersion),
    /// A Vulkan sampler the device cannot make: `maxSamplerAllocationCount` of the runtime's are
    /// live (its sampler states' samplers, one per use, LOD clamps and view mapping; its
    /// pipelines' immutable samplers; its neutral ones). A device's lack: the command that needs
    /// another is refused rather than recorded with a sampler Vulkan does not allow.
    SamplerLimit(SamplerLimit),
    /// A dynamic library's file that cannot be loaded or written (`MTLDynamicLibraryError`).
    DynamicLibrary(DynamicLibraryError),
    /// The Vulkan loader library opened from none of the places tried (no Metal counterpart:
    /// Metal has no loader). The GPU may be there: install the Vulkan loader, or name it with
    /// [`DeviceOptions::loader`](crate::DeviceOptions::loader).
    LoaderNotFound(LoaderNotFound),
    /// A driver setting the runtime cannot run under, read from the device when it is made or
    /// adopted: the device is refused, naming the setting ([`DriverSetting`]). Turn it back on.
    DriverSetting(DriverSetting),
    /// What plain Metal answers nil for, the process going on: the call and the problem (the
    /// module's docs).
    Nil(Nil),
    /// What plain Metal answers nil for with an `NSError`, the process going on: the call, the
    /// problem, and Metal's error ([`MetalError`]), which a consumer answering as Metal does can
    /// hand on as it is.
    MetalError(MetalError),
    /// What the runtime cannot prove plain Metal's answer for: the call and the problem
    /// ([`Unproven`]). Metal may make the object, or refuse it; the runtime does neither, so a
    /// consumer answering its guest as Metal does stops, naming the problem, rather than guess.
    Unproven(Unproven),
    /// A device extension the options list (`DeviceOptions::device_extensions`) whose feature
    /// structure the runtime knows, on a device that lacks one of its features: device creation
    /// refuses it rather than enable the extension without the feature (no Metal counterpart).
    ExtensionFeature(ExtensionFeature),
    /// A device extension the options list (`DeviceOptions::device_extensions`) whose registry
    /// dependency names an instance extension the instance would not enable: the options are
    /// refused before the instance is made, where a driver need not refuse the device (no Metal
    /// counterpart).
    InstanceExtensionRequired(InstanceExtensionRequired),
    /// A render encoder's call whose lowering, after the call returned, met a degradation the
    /// device's options refuse there (`DeviceOptions::lowering_refuses`): its notice, which was
    /// not announced. The call fails its command buffer (`CommandBufferError::CallRefused`), and
    /// this is its refusal (`CommandBuffer::refusal`). No Metal counterpart.
    Degraded(Box<crate::degradation::DegradeNotice>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(invalid) => {
                write!(f, "invalid use of {}: {:?}", invalid.call, invalid.problem)
            }
            Error::Unsupported(unsupported) => write!(f, "unsupported: {unsupported:?}"),
            Error::Bind(error) => write!(f, "the bind state was refused: {error:?}"),
            Error::Compile(error) => write!(f, "compile failed: {error:?}"),
            Error::NotImplemented(gap) => write!(f, "not implemented in the runtime yet: {gap:?}"),
            Error::PlacementConflict(conflict) => {
                write!(f, "the pinned address cannot be honoured: {conflict:?}")
            }
            Error::OutOfMemory(kind) => write!(f, "out of {kind:?} memory"),
            Error::DeviceLost => f.write_str("the Vulkan device was lost"),
            Error::Vulkan(code) => write!(f, "Vulkan failure {}", code.0),
            Error::NoDevice => f.write_str("no Vulkan device the runtime can use"),
            Error::AppleOnly(feature) => {
                write!(
                    f,
                    "an Apple-only feature the runtime refuses ({:?}): {feature:?}",
                    feature.gap()
                )
            }
            Error::ShaderCache(error) => write!(f, "{error}"),
            Error::BinaryArchive(error) => write!(f, "{error}"),
            Error::BinaryArchiveMiss(miss) => write!(f, "{miss}"),
            Error::RateMap(reason) => write!(f, "the rate map is not compiled: {reason}"),
            Error::WaitWouldDeadlock(cause) => {
                write!(f, "a command buffer wait would deadlock: {cause}")
            }
            Error::InstanceApiVersion(refused) => {
                write!(
                    f,
                    "the instance API version cannot be requested: {refused:?}"
                )
            }
            Error::SamplerLimit(SamplerLimit { limit }) => write!(
                f,
                "the device's {limit} Vulkan samplers (maxSamplerAllocationCount) are live"
            ),
            Error::DynamicLibrary(error) => write!(f, "{error}"),
            Error::LoaderNotFound(missing) => write!(f, "{missing}"),
            Error::DriverSetting(setting) => write!(f, "{setting}"),
            Error::Nil(nil) => {
                write!(f, "Metal answers nil for {}: {:?}", nil.call, nil.problem)
            }
            Error::MetalError(error) => write!(
                f,
                "Metal answers nil for {} with error {} {} \"{}\": {:?}",
                error.call, error.domain, error.code, error.localized_description, error.problem
            ),
            Error::Unproven(unproven) => write!(
                f,
                "Metal's answer to {} is unmeasured: {:?}",
                unproven.call, unproven.problem
            ),
            Error::ExtensionFeature(missing) => write!(
                f,
                "the device lists {} but lacks its feature {}",
                missing.extension, missing.feature
            ),
            Error::InstanceExtensionRequired(missing) => write!(
                f,
                "the device extension {} requires the instance extension {}, which the instance \
                 does not enable",
                missing.extension, missing.instance_extension
            ),
            Error::Degraded(notice) => write!(
                f,
                "{}: {}, refused as the device's options ask ({})",
                notice.object,
                notice.reason,
                notice.degradation.code()
            ),
        }
    }
}

/// A listed device extension's feature the device lacks ([`Error::ExtensionFeature`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtensionFeature {
    /// The extension, as the options name it (`VK_EXT_swapchain_maintenance1`).
    pub extension: &'static str,
    /// The feature, as the registry names its member (`swapchainMaintenance1`).
    pub feature: &'static str,
}

/// A listed device extension's instance extension the instance would not enable
/// ([`Error::InstanceExtensionRequired`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceExtensionRequired {
    /// The device extension, as the options name it (`VK_KHR_swapchain`).
    pub extension: &'static str,
    /// The instance extension it requires (`VK_KHR_surface`).
    pub instance_extension: &'static str,
}

/// Why a command buffer wait would deadlock ([`Error::WaitWouldDeadlock`]). Metal has no such
/// error: where the runtime refuses, Metal's wait hangs, or would hang, or (past the runtime's
/// cap) runs. A consumer that copies Metal's behaviour hangs the thread on
/// [`OwnQueue`](Self::OwnQueue), and decides the others itself.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeadlockCause {
    /// From one of the device's handler threads (a scheduled or completed handler, or a
    /// deallocator run as a command buffer releases what it retained), a wait for a transition
    /// of the handler's own queue still to come: its own command buffer's, or a later one's; or
    /// a new command buffer of that queue while it holds all of its `maxCommandBufferCount`.
    /// Metal never returns from that wait (measured on an M4, macOS 27.0.1).
    OwnQueue,
    /// From a handler thread, a wait for another queue whose handlers wait, through the queues
    /// they wait for, for this one: a cycle of handlers' waits, across every device. Metal would
    /// hang too (not measured).
    Cycle,
    /// From a handler thread, a wait for another queue that would leave the device's handler
    /// threads with none that is not blocked in such a wait (or a marked wait off those threads,
    /// which none can count): the 512th. Metal ran 511 such waits at once, which returned once
    /// the command buffers they waited for completed; with 512 its process's dispatch threads ran
    /// out, and no handler ran nor command buffer completed after: it hangs (the M3 Max under
    /// macOS 27.0.1). Metal's threads are the process's, the runtime's each device's.
    HandlerThreadCap,
    /// From any thread, once the process is exiting ([`process_exiting`](crate::process_exiting)):
    /// the runtime's threads that schedule and finish command buffers have ended, and it makes no
    /// Vulkan call, so a command buffer not yet scheduled or finished never will be. A wait
    /// already blocked answers within 50 ms.
    Exiting,
    /// From a submission handler ([`CommandBuffer::add_submission_handler`](crate::CommandBuffer::add_submission_handler),
    /// a runtime extension Metal has no counterpart of), a wait for what has not happened yet: a
    /// command buffer's scheduling or completion, or a slot of a full queue that no uncommitted
    /// command buffer holds. The handler runs on the thread that submits command buffers, and its
    /// own command buffer's scheduling, with every later transition of its queue, waits for it.
    SubmissionHandler,
}

impl fmt::Display for DeadlockCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DeadlockCause::OwnQueue => "it waits for a later transition of the handler's own queue",
            DeadlockCause::Cycle => "it would close a cycle of handlers' waits",
            DeadlockCause::HandlerThreadCap => "it would leave the device no free handler thread",
            DeadlockCause::Exiting => "the process is exiting",
            DeadlockCause::SubmissionHandler => {
                "it waits in a submission handler, which scheduling waits for"
            }
        })
    }
}

impl From<DeadlockCause> for Error {
    fn from(cause: DeadlockCause) -> Error {
        Error::WaitWouldDeadlock(cause)
    }
}

/// The Vulkan loader libraries that did not open, in the order the runtime tried them
/// ([`Error::LoaderNotFound`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoaderNotFound {
    /// Each library tried: [`DeviceOptions::loader`](crate::DeviceOptions::loader) where the
    /// options name one, else the platform's own by name (a bare name, which the dynamic linker
    /// searches for) and then, on macOS, the install directories no default search path holds.
    pub tried: Vec<PathBuf>,
}

impl fmt::Display for LoaderNotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the Vulkan loader did not open from any of")?;
        for (i, path) in self.tried.iter().enumerate() {
            let separator = if i == 0 { " " } else { ", " };
            write!(f, "{separator}{}", path.display())?;
        }
        Ok(())
    }
}

/// A driver setting a device is refused under ([`Error::DriverSetting`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DriverSetting {
    /// MoltenVK's `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` is off (an environment variable, a
    /// layer setting of the consumer's, or the consumer's own instance): Metal always binds
    /// through argument buffers, and without them MoltenVK passes every resource as a discrete
    /// Metal argument, whose form Metal's compiler crashes on for large fragment functions, and
    /// its bind limits fall to the per-stage ones. An instance the runtime creates asks for them.
    MoltenVkArgumentBuffersOff,
}

impl fmt::Display for DriverSetting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DriverSetting::MoltenVkArgumentBuffersOff => {
                "MoltenVK runs with MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS off: Metal always binds \
                 through argument buffers, and without them Metal's compiler crashes on large \
                 fragment functions and the bind limits shrink; turn the setting back on"
            }
        })
    }
}

impl std::error::Error for Error {}

/// A call Metal's validation stops, ending the process (the module's docs). One plain Metal answers
/// nil for is a [`Nil`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invalid {
    /// The method, by its Metal selector's snake_case name.
    pub call: &'static str,
    /// What it did wrong.
    pub problem: Problem,
}

impl Invalid {
    /// `problem` in `call`.
    pub fn new(call: &'static str, problem: Problem) -> Invalid {
        Invalid { call, problem }
    }
}

/// A call plain Metal answers nil for, the process going on (the module's docs): its API
/// validation layer may stop it too. One Metal's validation ends the process for is an
/// [`Invalid`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nil {
    /// The method, by its Metal selector's snake_case name.
    pub call: &'static str,
    /// Why Metal answers nil.
    pub problem: Problem,
}

impl Nil {
    /// `problem` in `call`.
    pub fn new(call: &'static str, problem: Problem) -> Nil {
        Nil { call, problem }
    }
}

/// A call whose answer on plain Metal the runtime cannot prove ([`Error::Unproven`]): Metal may
/// make what it asks for, or refuse it. Every problem the runtime answers so:
/// [`Problem::IndirectCommandsDirectArgumentUnproven`],
/// [`Problem::UnwrittenFragmentInputsUnproven`], a device's [`Problem::HostReleaseUnmeasured`], and
/// [`Problem::IndirectCommandsDirectArgumentUndecided`].
///
/// The first refusal in Metal's order decides, proven or not: an unproven answer stops the call's
/// checks even where a later one would be a proven refusal. A fragment function's unproven direct
/// argument hides the vertex function's proven one, and the indirect command buffers' unproven
/// answer comes before a proven [`Problem::UnwrittenFragmentInputs`]. There Metal answers nil,
/// while the runtime answers `Unproven`, since the `NSError` text Metal would give is uncertain.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unproven {
    /// The method, by its Metal selector's snake_case name.
    pub call: &'static str,
    /// What Metal's answer is unmeasured for.
    pub problem: Problem,
}

impl Unproven {
    /// `problem` in `call`.
    pub fn new(call: &'static str, problem: Problem) -> Unproven {
        Unproven { call, problem }
    }
}

/// A call plain Metal answers nil for with an `NSError`, the process going on: the call, the
/// problem, and Metal's error as measured ([`Error::MetalError`]). A consumer that answers its
/// guest as Metal does makes the `NSError` of `domain`, `code` and a `userInfo` holding
/// `NSLocalizedDescription` = `localized_description`, its only key where Metal's holds no
/// other (every refusal the runtime answers so: [`Problem::IndirectCommandsDirectArgument`],
/// [`Problem::UnrasterizedVertexReturn`], [`Problem::DepthWrittenWithoutAttachment`],
/// [`Problem::UnwrittenFragmentInputs`], [`Problem::RasterSampleCount`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalError {
    /// The method, by its Metal selector's snake_case name.
    pub call: &'static str,
    /// Why Metal refuses.
    pub problem: Problem,
    /// `NSError.domain`. Where Metal's is its GPU driver's class, it follows the device: the
    /// runtime answers the M3 Max's under macOS 26.7 (each problem's docs name the devices
    /// measured), and a consumer may substitute its own device's.
    pub domain: &'static str,
    /// `NSError.code`.
    pub code: i64,
    /// `NSError.localizedDescription`: `userInfo`'s `NSLocalizedDescription` (from `&'static str`:
    /// Metal's text may name the program's own identifiers; a refusal is rare, so its text is
    /// allocated).
    pub localized_description: Box<str>,
}

impl MetalError {
    /// `problem` in `call`, Metal's error of `domain`, `code` and `localized_description`.
    pub fn new(
        call: &'static str,
        problem: Problem,
        domain: &'static str,
        code: i64,
        localized_description: &'static str,
    ) -> MetalError {
        MetalError {
            call,
            problem,
            domain,
            code,
            localized_description: localized_description.into(),
        }
    }

    /// [`new`](Self::new), of a `localized_description` made for the call.
    pub fn with_description(
        call: &'static str,
        problem: Problem,
        domain: &'static str,
        code: i64,
        localized_description: String,
    ) -> MetalError {
        MetalError {
            call,
            problem,
            domain,
            code,
            localized_description: localized_description.into_boxed_str(),
        }
    }
}

/// What a call did wrong.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    /// An encoder of this command buffer is open: one encoder at a time. Plain Metal asserts too
    /// ("A command encoder is already encoding to this command buffer", "commit command buffer with
    /// uncommitted encoder", "encodeSignalEvent:value: with uncommitted encoder"; an M4).
    EncoderOpen,
    /// A compute or render encoder made, or a blit encoder's first command (`call` names the
    /// blit encoder), under protection options 1, 3 or 7 (`CommandBuffer::set_protection_options`),
    /// where Metal's process ends (SIGABRT, measured by a consumer on an M3 Max, macOS 26.7).
    ProtectedEncoder {
        /// The protection options set.
        options: u64,
    },
    /// An encoder of this command buffer was dropped without `end_encoding`: its commands are
    /// lost, so the command buffer can never be committed. Plain Metal asserts as the encoder
    /// goes ("Command encoder released without endEncoding").
    EncoderAbandoned,
    /// The command buffer was already committed. Plain Metal asserts too ("commit an already
    /// committed command buffer", and a new encoder's `_status <
    /// MTLCommandBufferStatusCommitted`).
    AlreadyCommitted,
    /// A scheduled signal encoded once the command buffer completed, or failed (status `Error`)
    /// (`CommandBuffer::encode_signal_event_scheduled`): Metal's process ends there (a
    /// segmentation fault, as the consumer measured; in 10 of 15 runs on an M3 Max under macOS
    /// 27.0.1, the others setting nothing). Encoded after the commit and before the completion,
    /// it does nothing, as Metal's.
    ScheduledSignalAfterCompletion,
    /// An object made by another device.
    ForeignObject,
    /// Dma-bufs a `DmaBufSource` returned (a runtime extension: Metal has none,
    /// `HostMemory::with_dma_buf`) that break its contract: chunks that do not cover the pages
    /// asked for, split off the granule, or of lengths off the host page, and a descriptor that
    /// is no dma-buf or holds fewer bytes than its chunk.
    DmaBuf,
    /// An image the caller made that cannot be wrapped as a texture of the descriptor
    /// (`new_texture_with_vk_image`).
    WrappedImage(WrappedImageProblem),
    /// A handle of a texture that is not shareable, for which Metal answers nil, an [`Error::Nil`].
    NotShareable,
    /// A semaphore that cannot be one of the caller's binary semaphores.
    Semaphore(SemaphoreProblem),
    /// No pipeline state is set (plain Metal faults at a compute dispatch without one).
    NoPipelineState,
    /// A run of compute encoders' first indirect dispatch (`dispatch_threadgroups_indirect`,
    /// `dispatch_threads_indirect`) in an encoder whose dispatch type is not the run's first
    /// encoder's, on a device whose Metal level ends the process there (a segmentation fault):
    /// [`MetalLevel::MacOS26_7`](crate::MetalLevel::MacOS26_7), as the consumer measured on an M3
    /// Max. A run is a command buffer's compute encoders one after another with no other encoder
    /// and no event signal or wait between; its first encoder's type counts though it dispatches
    /// nothing, and once one indirect dispatch of the run went through, every later one does.
    /// macOS 27.0.1 runs it.
    IndirectDispatchOfAnotherType,
    /// An argument-table index past Metal's table (plain Metal writes past its own table: a
    /// compute texture at index 128 faulted the dispatch on an M4).
    IndexOutOfRange {
        /// The index given.
        index: u32,
        /// The table's entries: indices below it are taken.
        limit: u32,
    },
    /// An offset past the end of the buffer: at the bind in a render stage's table, and at a
    /// compute dispatch whose kernel reads a slot bound so, where Metal's read is past the buffer.
    OffsetOutOfRange {
        /// The offset given.
        offset: u64,
        /// The buffer's length.
        length: u64,
    },
    /// `set_buffer_offset` at an index with no buffer bound: a render stage's form (a compute
    /// form at an index with nothing bound changes nothing, as plain Metal's; any form at an index
    /// holding a visible-function table makes it read as the null table).
    NoBufferBound {
        /// The argument-table index.
        index: u32,
    },
    /// `set_bytes` (or a stage's form) of more bytes than Metal's limit for inline bytes.
    InlineBytesLength {
        /// The bytes given.
        length: u64,
        /// The most bytes one call takes.
        limit: u64,
    },
    /// An imageblock size of more than 1024 cells over more than one row and column, or with a
    /// side past 65536, at a compute encoder's `set_imageblock_size` (plain Metal runs 1024 by
    /// 1 there with its cells reading 0, or is unmeasured); and a width or height that is not a
    /// power of two, or more than 1024 cells, at an indirect command's execution (as Metal's API
    /// validation refuses it).
    ImageblockSize {
        /// The width given, in pixels.
        width: u64,
        /// The height given, in pixels.
        height: u64,
    },
    /// A tile dispatch's threadgroup with no threads along an axis (a compute dispatch of one
    /// runs nothing, as plain Metal's).
    EmptyThreadgroup,
    /// A size past what the call can state: a tile dispatch's or mesh draw's axis past 32 bits
    /// (compute sizes and per-bind attribute strides past 32 bits are taken modulo `2^32`, as
    /// plain Metal takes them).
    SizeOutOfRange,
    /// A buffer of no bytes, or more than the device can hold in one: plain Metal answers nil
    /// (`Nil`), from the device, with bytes or over the caller's, and from either type of heap.
    BufferLength {
        /// The length asked for, in bytes.
        length: u64,
    },
    /// A raw value Metal does not define for this type.
    UndefinedValue {
        /// The Metal type's name (`MTLPurgeableState`, for one).
        ty: &'static str,
        /// The raw value given.
        raw: u64,
    },
    /// A storage mode this call does not take (`Memoryless` for a buffer, for one): Metal stops
    /// it (`Invalid`), but a shared texture's other than `Private`, which plain Metal answers nil
    /// for (`Nil`).
    StorageMode,
    /// A handler added after the command buffer was committed. Plain Metal asserts too
    /// ("Completed handler provided after commit call").
    HandlerAfterCommit,
    /// The given slices' lengths differ.
    LengthMismatch,
    /// A texture fill given no pattern bytes (`fillTexture:...:bytes:length:` of length 0).
    /// Metal's validation stops it ("buffer length for format ... must be at least" a unit's
    /// bytes); plain Metal reads a unit's bytes from the pattern's address all the same, bytes the
    /// call does not give (an Apple M3 Max under macOS 27.0.1).
    EmptyFillPattern,
    /// A texture buffer's fill given a pattern of more than 32752 bytes
    /// (`fillTexture:...:bytes:length:`): Metal's process ends in the call, in the blit's
    /// dispatch, whatever the region, of no texels or past the buffer too (an Apple M3 Max under
    /// macOS 27.0.1: `R8Uint`, `R32Uint` and `RGBA32Uint` texture buffers; 32752 bytes are
    /// filled). A fill of any other texture takes a pattern of any length, of which it reads 16
    /// bytes at most.
    LongTextureBufferFillPattern {
        /// The pattern's bytes.
        length: u64,
    },
    /// `offset..offset + size` runs past a resource of `length` bytes: Metal stops it
    /// (`Invalid`). A resource placed past its heap's end is [`HeapProblem::PastEnd`].
    RangeOutOfBounds {
        /// The range's first byte.
        offset: u64,
        /// The range's bytes.
        size: u64,
        /// The resource's length, in bytes.
        length: u64,
    },
    /// A shader cache capacity, on disk or in memory, of zero: neither cache can be turned
    /// off.
    ShaderCacheCapacity,
    /// A command queue's `maxCommandBufferCount` of zero: Metal's command queue creation
    /// validation stops it, with or without API validation.
    MaxCommandBufferCount,
    /// A texture descriptor Metal stops (`Invalid`), or plain Metal answers nil for (`Nil`):
    /// [`TextureProblem`] says which.
    Texture(TextureProblem),
    /// A texture view Metal stops, or a use of a texture its usage does not allow.
    TextureView(ViewProblem),
    /// A texture copy Metal stops: `replace_region`, `get_bytes`, and the blit encoder's.
    Copy(CopyProblem),
    /// A `generate_mipmaps` Metal stops.
    Mipmap(MipmapProblem),
    /// A texture over a buffer Metal stops.
    BufferTexture(BufferTextureProblem),
    /// A render pass descriptor, or a render encoder's late store action, Metal stops.
    RenderPass(RenderPassProblem),
    /// A draw, or render encoder state, Metal stops.
    Draw(DrawProblem),
    /// `new_render_pipeline_state`: the vertex function writes `[[point_size]]` and the input
    /// primitive topology class is line or triangle. Metal's pipeline creation answers an error
    /// here, with or without API validation ("Vertex shader writes point size but
    /// inputPrimitiveTopology is ..."; measured on an M3 Max and an M4).
    PointSizeTopology,
    /// `new_render_pipeline_state`: a tessellation pipeline made with
    /// `supportIndirectCommandBuffers` whose factor scale is enabled. Metal's validation stops it
    /// ("pipelines with tessellationFactorScaleEnabled = YES are not compatible with indirect
    /// command buffers", measured on an M4).
    IndirectScaledTessellation,
    /// `new_compute_pipeline_state`: a kernel the compile layer already compiled
    /// (`StageSource::Compiled`). Metal's descriptor takes a function, and a state compiles its
    /// kernel again for each variant a dispatch needs.
    CompiledKernel,
    /// `new_compute_pipeline_state`: the kernel's linked `functions`, or its `binary_functions`,
    /// list this name twice, or both lists hold it for two different functions. Metal's API
    /// validation stops every repeat ("All linked functions must have unique names"); plain
    /// Metal's creation fails on a name `functions` repeats and takes one function listed in both
    /// lists, as the runtime takes it (measured on an M3 Max and an M4).
    DuplicateLinkedFunction {
        /// The function's name.
        name: String,
    },
    /// A viewport, scissor rectangle or other render encoder state Metal stops at its setter.
    RenderState(RenderStateProblem),
    /// A `set_visibility_result_mode` Metal stops.
    Visibility(VisibilityProblem),
    /// A heap, or a resource placed in one, Metal stops (`new_heap` and the heap's calls).
    Heap(HeapProblem),
    /// A resource ID, placed or reserved, of `limit` or more: Metal ends the process for one
    /// (`resource_ids.rs`). A pipeline state's is its placement's low 32 bits, which Metal keeps.
    ResourceId {
        /// The resource ID asked for.
        id: u64,
        /// The first ID past the kind's range.
        limit: u64,
    },
    /// An indirect command buffer ID of `limit` (2^31) to 2^32 - 1 in a reservation
    /// (`Device::reserve_resource_ids`): Metal's call never returns there, its pool spinning
    /// (an M3 Max, macOS 27.0.1), so nothing is reserved. A caller matching Metal treats this as
    /// that hang, not as an invalid argument.
    ReservationHangs {
        /// The first such ID in the call.
        id: u64,
        /// The first ID at which Metal's reservation hangs.
        limit: u64,
    },
    /// [`IdPlacement::Outside`](crate::IdPlacement::Outside) for anything but a texture
    /// `Device::new_texture` makes: a sampler state, a pipeline state, an indirect command
    /// buffer, a view, or a heap's, a buffer's, a shared or a caller's texture. A runtime
    /// extension's misuse: Metal has no such placement.
    OutsidePlacement,
    /// A visible-function table call, or a table or buffer binding a dispatch reads as the other,
    /// Metal stops.
    FunctionTable(FunctionTableProblem),
    /// An indirect command buffer call Metal stops or ends the process for.
    IndirectCommandBuffer(IndirectCommandBufferProblem),
    /// A rasterization-rate map's descriptor Metal stops (`new_rasterization_rate_map`).
    RateMap(RateMapProblem),
    /// An argument encoder call Metal stops, raises for, or answers nil for.
    ArgumentEncoder(ArgumentEncoderProblem),
    /// An indirect dispatch's `bytes` of arguments at `offset` that run past its buffer's
    /// `length` (Metal's API validation: "(indirectBufferOffset +
    /// sizeof(MTLDispatchThreadsIndirectArguments))(72) must be <= \[indirectBuffer length\](64)";
    /// what plain Metal reads there is memory). An offset that is not a multiple of 4 is read at
    /// the multiple below, as plain Metal reads it, and checked from there.
    IndirectDispatchRange {
        /// The offset the arguments are read at: the offset given, rounded down to a multiple
        /// of 4.
        offset: u64,
        /// The arguments' bytes.
        bytes: u64,
        /// The buffer's length.
        length: u64,
    },
    /// A sparse heap's or texture's call Metal stops, or answers nil for.
    Sparse(SparseProblem),
    /// A buffer the caller made that cannot be wrapped as a Metal buffer of the length and
    /// storage (`new_buffer_with_vk_buffer`).
    WrappedBuffer(WrappedBufferProblem),
    /// A tile function's threadgroup argument of `length` bytes at byte `offset` of its tile's
    /// threadgroup memory, which the pass makes `tile_length` bytes (its
    /// `threadgroupMemoryLength`): a length or offset that is not a multiple of 16, or an
    /// argument that ends past the tile's memory (`set_threadgroup_memory_length` of a render
    /// encoder, as Metal's validation refuses it).
    TileThreadgroupArgument {
        /// The argument's length, in bytes.
        length: u64,
        /// The argument's offset in the tile's threadgroup memory.
        offset: u64,
        /// The tile's threadgroup memory, in bytes.
        tile_length: u64,
    },
    /// A depth/stencil state's `gpu_resource_id` on a device that does not claim Apple10:
    /// Metal's API validation stops it ("Obtaining the resourceID of a DepthStencilState object
    /// requires support for MTLGPUFamilyApple10"), and the primary reference's plain call (macOS
    /// 26.7) raises; macOS 27.0.1's plain call answers 0.
    DepthStencilResourceId,
    /// A pipeline's `requiredThreadsPerThreadgroup` that Metal refuses at creation, or a dispatch
    /// whose threadgroup size is not it.
    RequiredThreads(RequiredThreadsProblem),
    /// A pipeline's binary functions that Metal refuses: not precompiled, of another stage, or
    /// added to a pipeline created without support for adding them.
    BinaryFunctions(BinaryFunctionProblem),
    /// A function made by name from a dynamic library (`newFunctionWithName:` and its forms):
    /// Metal stops at an assertion ("type is not a valid MTLFunctionType") for every name, a
    /// kernel the library holds among them (a probe, an M3 Max and an M4, plain and under API
    /// validation).
    DynamicLibraryFunction,
    /// `rebind_bytes_no_copy` of a buffer or heap the runtime cannot rebind as asked (a runtime
    /// extension; `rebind.rs`).
    Rebind(RebindProblem),
    /// `new_render_pipeline_state`, `new_mesh_render_pipeline_state` or
    /// `new_compute_pipeline_state` of a descriptor with `supportIndirectCommandBuffers` whose
    /// `stage` function uses a direct texture or sampler argument (`[[texture(n)]]`,
    /// `[[sampler(n)]]`; its reflection's used bindings of those types): plain Metal answers nil
    /// with an `NSError` ([`Error::MetalError`]), on an M3 Max under macOS 26.7 and an M4 under
    /// 27.0.1 alike. Unused ones, an argument buffer's textures and samplers, and constexpr
    /// samplers are made. Metal names the fragment function first, then
    /// the vertex, mesh and object functions. Its error: code 3, `localizedDescription` "Vertex
    /// shader cannot be used with indirect command buffers" (and "Fragment shader", "Mesh
    /// shader", "Object shader", "Compute function" alike), `userInfo` holding that alone, and
    /// the domain of Metal's GPU driver class, which follows the device: `AGXMetalG15X_M1` on the
    /// M3 Max, which the runtime answers, and `AGXMetalG16G_B0` on the M4.
    IndirectCommandsDirectArgument {
        /// The function's stage: `Vertex`, `Fragment`, `Mesh`, `Object`, or `Kernel` (a compute
        /// state's).
        stage: Stage,
    },
    /// `new_render_pipeline_state` of a descriptor with rasterization disabled whose vertex
    /// function's return type is not `void`: plain Metal answers nil with an `NSError`
    /// ([`Error::MetalError`]) with or without `supportIndirectCommandBuffers`, before the
    /// indirect command buffers' refusal (on an M4 under macOS 27.0.1), a post-tessellation vertex
    /// function's too. Its error: code 3, `localizedDescription` "RasterizationEnabled is false but
    /// the vertex shader's return type is not void", `userInfo` holding that alone, and the domain
    /// of Metal's GPU driver class, as [`Problem::IndirectCommandsDirectArgument`]'s.
    UnrasterizedVertexReturn,
    /// `new_render_pipeline_state` (a render or mesh state) of a descriptor with
    /// `supportIndirectCommandBuffers` whose function of `stage`, the first in the order Metal
    /// names them that uses a direct texture or sampler once linked, uses one only by a rule the
    /// runtime keeps beyond Metal where Metal's answer is unmeasured, `reason`
    /// ([`Error::Unproven`]). Metal may make the state, or refuse it as
    /// [`Problem::IndirectCommandsDirectArgument`]; every use a measured rule keeps is that problem
    /// instead.
    IndirectCommandsDirectArgumentUnproven {
        /// The function's stage: `Vertex`, `Mesh` or `Fragment`.
        stage: Stage,
        /// The rule that keeps the use.
        reason: LinkedUseGap,
    },
    /// `new_render_pipeline_state`, `new_mesh_render_pipeline_state` or
    /// `new_compute_pipeline_state` of a descriptor with `supportIndirectCommandBuffers` where the
    /// compile layer refuses to say whether its functions use a direct texture or sampler
    /// argument, `refusal` ([`Error::Unproven`]): a stage the compile layer already compiled that
    /// kept no reflection (`Unsupported` with `Feature::Reflection`), or the decode's or
    /// specialization's own refusal (The use is decided from the arguments' roles, so an argument
    /// whose reflection Metal's answer has not been measured for no longer refuses it). Metal may
    /// make the state, or refuse it as [`Problem::IndirectCommandsDirectArgument`]; the runtime
    /// makes no state it cannot decide that for.
    IndirectCommandsDirectArgumentUndecided {
        /// The compile layer's refusal.
        refusal: Box<CompileError>,
    },
    /// `new_render_pipeline_state` of a descriptor without a depth attachment whose fragment
    /// function writes depth (a `[[depth(...)]]` output): plain Metal answers nil with an
    /// `NSError` ([`Error::MetalError`]) with or without rasterization, a color attachment or
    /// `supportIndirectCommandBuffers`, before the indirect command buffers' refusal (on an M4
    /// under macOS 27.0.1); `new_mesh_render_pipeline_state` alike, first. Its error: code 3,
    /// `localizedDescription` "depthAttachmentPixelFormat is not valid and shader writes to depth",
    /// and the domain of Metal's GPU driver class, as
    /// [`Problem::IndirectCommandsDirectArgument`]'s.
    DepthWrittenWithoutAttachment,
    /// `new_render_pipeline_state` of a descriptor whose fragment function's stage inputs
    /// `inputs` (their MSL member names, in declaration order), which its linking keeps, are
    /// written by no vertex output, with or without rasterization: plain Metal answers nil with an
    /// `NSError` ([`Error::MetalError`]), after the indirect command buffers' refusal (on an M4
    /// under macOS 27.0.1). Its error: code 3, `localizedDescription` "Fragment input(s) `c,d`
    /// mismatching vertex shader output type(s) or not written by vertex shader" with `inputs`
    /// joined by commas, and the domain of Metal's GPU driver class. A post-tessellation vertex
    /// function's alike, and `new_mesh_render_pipeline_state`'s where the mesh function writes no
    /// per-vertex or per-primitive data of them, its text naming the "mesh shader".
    UnwrittenFragmentInputs {
        /// The inputs' names.
        inputs: Box<[String]>,
    },
    /// [`Problem::UnwrittenFragmentInputs`] where only a rule kept beyond Metal keeps such an
    /// input (`reason`): Metal may make the state ([`Error::Unproven`]).
    UnwrittenFragmentInputsUnproven {
        /// The inputs' names, as every rule keeps them.
        inputs: Box<[String]>,
        /// The rule that keeps one.
        reason: LinkedUseGap,
    },
    /// A device made, with no `metal_level` named, on a macOS host whose release (`macos`: major,
    /// minor, patch; `None` where it cannot be read) is below every measured
    /// [`MetalLevel`](crate::MetalLevel) ([`Error::Unproven`]): an older Metal's answers are
    /// unmeasured, and a newer one's would answer above the host. Naming a level in the options
    /// makes the device answer at it.
    HostReleaseUnmeasured {
        /// The host's macOS version, where it reads.
        macos: Option<[u32; 3]>,
    },
    /// `new_render_pipeline_state` or `new_mesh_render_pipeline_state` of a descriptor whose
    /// `attachment` is in a pixel format (`format`) the device's
    /// [`MetalLevel`](crate::MetalLevel) describes none of: under macOS 26.7 the values from 630
    /// to 650 and the three-channel ones, under 27.0.1 the three-channel ones and 646 to 650, and
    /// `Depth24Unorm_Stencil8` and `X24_Stencil8` at every level (an Apple GPU has neither).
    /// Plain Metal's descriptor validation stops it ("pixelFormat, for color render target(0), is
    /// not a valid MTLPixelFormat", "depthAttachmentPixelFormat is not a valid MTLPixelFormat";
    /// on an Apple M4 under 27.0.1, the consumer's run on an M3 Max under 26.7),
    /// blended or not, before anything else; the first such attachment is named (the colors by
    /// index, then depth, then stencil).
    AttachmentNotAFormat {
        /// The attachment.
        attachment: Attachment,
        /// Its pixel format.
        format: PixelFormat,
    },
    /// `new_render_pipeline_state` or `new_mesh_render_pipeline_state` of a `rasterSampleCount`
    /// other than 1, 2 and 4, and `new_tile_render_pipeline_state` of 0, on every device: plain
    /// Metal answers nil with an `NSError` (code 2, "rasterSampleCount (8) is not supported by
    /// device."; on an Apple M4 under 27.0.1, with and without API validation), an
    /// [`Error::MetalError`]. A device whose Vulkan driver rasterizes more
    /// samples makes no such state either.
    RasterSampleCount {
        /// The `rasterSampleCount` given.
        count: u32,
    },
}

/// Why `DeviceOptions::instance_api_version` cannot be requested. The runtime never clamps a
/// request to what it can make.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstanceApiVersion {
    /// Below Vulkan 1.2, the runtime's floor.
    BelowFloor {
        /// The version requested.
        requested: ApiVersion,
    },
    /// Above the version the Vulkan loader reports (`vkEnumerateInstanceVersion`; 1.0 where the
    /// loader has no such command).
    AboveLoader {
        /// The version requested.
        requested: ApiVersion,
        /// The version the loader reports.
        loader: ApiVersion,
    },
}

/// The device's sampler count at its limit (`Error::SamplerLimit`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerLimit {
    /// The device's `maxSamplerAllocationCount`.
    pub limit: u32,
}

/// Why a shader cache directory cannot be used.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderCacheError {
    /// The directory, or `None` when there is no default one (no home directory).
    pub path: Option<PathBuf>,
    /// The I/O error's kind.
    pub kind: std::io::ErrorKind,
    /// The I/O error's text.
    pub message: String,
}

impl ShaderCacheError {
    /// The failure `kind` (`message`) of `path`.
    pub fn new(path: Option<PathBuf>, kind: std::io::ErrorKind, message: &str) -> ShaderCacheError {
        ShaderCacheError {
            path,
            kind,
            message: message.to_string(),
        }
    }
}

impl fmt::Display for ShaderCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.path {
            Some(path) => write!(
                f,
                "the shader cache directory {} cannot be used: {}",
                path.display(),
                self.message
            ),
            None => write!(f, "no shader cache directory: {}", self.message),
        }
    }
}

/// Why a binary archive cannot be loaded or serialized (`MTLBinaryArchiveError`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryArchiveError {
    /// `MTLBinaryArchiveErrorInvalidFile`: the file to load is missing or not an archive, or
    /// the directory to serialize into is missing, as Metal answers (measured on an M3 Max and
    /// an M4).
    InvalidFile {
        /// The file or directory.
        path: PathBuf,
    },
    /// `MTLBinaryArchiveErrorCompilationFailure`, "Nothing to serialize": nothing was added to
    /// the archive or loaded into it.
    NothingToSerialize,
    /// The file cannot be written, where Metal answers the file system's error (a directory at
    /// the path, for one): the path, and the I/O error's kind and text.
    Write {
        /// The file.
        path: PathBuf,
        /// The I/O error's kind.
        kind: std::io::ErrorKind,
        /// The I/O error's text.
        message: String,
    },
}

impl fmt::Display for BinaryArchiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryArchiveError::InvalidFile { path } => {
                write!(f, "{} is not a binary archive", path.display())
            }
            BinaryArchiveError::NothingToSerialize => f.write_str("Nothing to serialize"),
            BinaryArchiveError::Write { path, message, .. } => {
                write!(
                    f,
                    "the binary archive {} cannot be written: {message}",
                    path.display()
                )
            }
        }
    }
}

/// Why a dynamic library's file cannot be loaded or written (`MTLDynamicLibraryError`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DynamicLibraryError {
    /// `MTLDynamicLibraryErrorInvalidFile`: the file to load holds no library ("its AIR slice is
    /// missing"), or the directory to serialize into is missing ("Invalid URL"), as Metal
    /// answers (a probe, an M3 Max and an M4).
    InvalidFile {
        /// The file.
        path: PathBuf,
    },
    /// The file to load cannot be read, where Metal answers the file system's error (a missing
    /// file is `NSCocoaErrorDomain` 260): the path, and the I/O error's kind and text.
    Read {
        /// The file.
        path: PathBuf,
        /// The I/O error's kind.
        kind: std::io::ErrorKind,
        /// The I/O error's text.
        message: String,
    },
    /// The file cannot be written, where Metal answers the file system's error (a directory at
    /// the path, for one): the path, and the I/O error's kind and text.
    Write {
        /// The file.
        path: PathBuf,
        /// The I/O error's kind.
        kind: std::io::ErrorKind,
        /// The I/O error's text.
        message: String,
    },
}

impl fmt::Display for DynamicLibraryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynamicLibraryError::InvalidFile { path } => {
                write!(f, "{} is not a dynamic library's file", path.display())
            }
            DynamicLibraryError::Read { path, message, .. } => {
                write!(
                    f,
                    "the dynamic library {} cannot be read: {message}",
                    path.display()
                )
            }
            DynamicLibraryError::Write { path, message, .. } => {
                write!(
                    f,
                    "the dynamic library {} cannot be written: {message}",
                    path.display()
                )
            }
        }
    }
}

/// The pipeline a creation under `PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS` found in no
/// listed binary archive. Metal answers its creation nil with an error of code 3 for each kind
/// of pipeline, compute, render and tile alike (the consumer's measurement, an M3 Max under
/// macOS 26.7), whose text [`Display`](fmt::Display) gives.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryArchiveMiss {
    /// The function Metal names: a compute pipeline's kernel; a tile pipeline's tile function;
    /// a render pipeline's first stage whose archive key no listed archive holds, its fragment
    /// function before its vertex function (the fragment function for a color change; the vertex
    /// function where the fragment's key is held, as for a topology change, or where the pipeline
    /// has no fragment function).
    pub function: String,
    /// The kind of pipeline.
    pub pipeline: ArchivedPipeline,
}

impl BinaryArchiveMiss {
    /// `pipeline`, of `function`, found in no archive.
    pub fn new(function: &str, pipeline: ArchivedPipeline) -> BinaryArchiveMiss {
        BinaryArchiveMiss {
            function: function.to_string(),
            pipeline,
        }
    }
}

impl fmt::Display for BinaryArchiveMiss {
    /// Metal's text (measured on an M3 Max and an M4): `Unable to find function "<function>"
    /// (compute pipeline) in binary archives.`, with `render pipeline` or `tile pipeline` for
    /// those kinds.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pipeline = match self.pipeline {
            ArchivedPipeline::Compute => "compute",
            ArchivedPipeline::Render => "render",
            ArchivedPipeline::Tile => "tile",
        };
        write!(
            f,
            "Unable to find function \"{}\" ({pipeline} pipeline) in binary archives.",
            self.function
        )
    }
}

/// A kind of pipeline a binary archive holds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArchivedPipeline {
    /// A compute pipeline (`newComputePipelineStateWithDescriptor:...`).
    Compute,
    /// A render pipeline (`newRenderPipelineStateWithDescriptor:...`).
    Render,
    /// A tile render pipeline (`Device::new_tile_render_pipeline_state`).
    Tile,
}

/// A runtime piece a later build step adds. Each is refused typed until then: Metal itself takes
/// these inputs.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeGap {
    /// Textures and texture buffers.
    Textures,
    /// The resource-ID heap, descriptor set 1.
    ResourceHeap,
    /// Sparse heaps and their textures.
    SparseHeaps,
    /// What the runtime's emulation of sparse textures does not build yet: a tile map of a cube or
    /// multisampled sparse texture, which the tile-map format has no layout for; a sparse texture
    /// whose unmapped texels the runtime cannot zero (a single-sampled one of a format the device
    /// copies nothing into, a multisampled one of a format it does not clear and render); and a
    /// sparse heap or texture on a device that binds no sparse memory (`sparseBinding`, or the
    /// queue family's bit) where functions read the image's own residency (it enables
    /// `shaderResourceResidency`), whose exact path is the tile-map arm.
    SparseTileMaps,
    /// A heap larger than the device's `maxBufferSize`, which the device can allocate: the
    /// runtime places a heap's buffers in one `VkBuffer` over the whole heap, which no `VkBuffer`
    /// that size can be. Windows of `VkBuffer`s over the one allocation are the exact path.
    HeapPastMaxBufferSize {
        /// The heap's size, in bytes.
        size: u64,
        /// The device's `maxBufferSize`.
        max_buffer_size: u64,
    },
    /// A heap texture whose image needs an alignment past the heap memory's base alignment
    /// (256 KiB): no offset in the heap is sure to meet it. The base is the largest image
    /// alignment measured (an RX 7900 XTX's multisample images under RADV).
    HeapTextureAlignment {
        /// The alignment the image needs, in bytes.
        alignment: u64,
    },
    /// A heap texture whose image the heap's memory type does not take: an optimal image whose
    /// memory types (a depth format's, which Vulkan lets differ from a color image's) leave out
    /// the heap's. Memory of its own would lose the heap's aliasing. (A texture whose linear image
    /// needs extended usage is checked format by format, not refused with it.) In a heap over host
    /// pages, also a texture no linear image over the pages serves and no shadow of the heap's
    /// bytes holds, and one with a depth twin. A multisample texture there is made in memory of
    /// its own.
    HeapTextureMemory,
    /// A visible-function table of more entries than one buffer of the device's
    /// `max_buffer_length` holds words for, or than its size word holds: the runtime keeps a
    /// table's words in one buffer. Metal makes such a table on an M3 Max and ends the process on
    /// an M4.
    FunctionTablePastMaxBufferLength {
        /// The table's entries.
        count: u64,
        /// The device's `max_buffer_length`.
        max_buffer_length: u64,
    },
    /// An indirect command buffer of more commands than one buffer of the device's
    /// `max_buffer_length` holds the records of: the runtime keeps them in one buffer.
    IndirectCommandBufferPastMaxBufferLength {
        /// The buffer's commands.
        count: u64,
        /// The device's `max_buffer_length`.
        max_buffer_length: u64,
    },
    /// What indirect command buffers lack of their render commands: a command whose mesh draw's
    /// object function has threadgroup memory arguments, whose lengths (the command's
    /// `setObjectThreadgroupMemoryLength:atIndex:` or the encoder's) are not measured; an
    /// execution of a range a draw of the encoder before
    /// it may write, or of commands that need a render pass instance of their own (a fetched
    /// attachment, the shader-side stencil, views, a fragment imageblock, a texture whose shadow
    /// the pass does not hold, or a memoryless attachment the pass backs and reads past its
    /// instances); an execution of render commands by a compute encoder, or of compute commands by
    /// a render encoder; and, on a device that claims Apple10, a depth/stencil state's ID, which
    /// names the state such a command sets.
    IndirectRenderCommands,
    /// An indirect compute command's imageblock side past 65535, which Metal keeps: the record
    /// holds 16 bits a side, as a kernel's `set_imageblock_size` writes it, and no dispatch runs
    /// an imageblock of more than 1024 cells.
    IndirectImageblockSize {
        /// The imageblock's width.
        width: u64,
        /// The imageblock's height.
        height: u64,
    },
    /// The device replay of indirect command buffers.
    CommandReplay,
    /// Render attachments and their binding forms.
    RenderAttachments,
    /// Acceleration structures.
    AccelerationStructures,
    /// A feature a created device would enable that the runtime's feature chain has no member
    /// for, or whose structure no enabled extension exposes.
    FeatureEnable(VkFeature),
    /// An extension a created device would enable that the device lists under none of the names
    /// the runtime knows for it.
    ExtensionEnable(VkExtension),
    /// Render pipelines, passes and draws.
    Render,
    /// A 2D array over a buffer whose texels run past the buffer's length, which Metal makes
    /// over bytes its buffer's allocation has past the length, or a two-plane YCbCr texture over
    /// a surface's plane whose luma codes run past the buffer, which Metal reads as zeros there:
    /// the runtime's buffers have no bytes there.
    TexelsPastBufferLength,
    /// A texture of an IOSurface plane (`Buffer::new_texture_with_iosurface_plane`, the stand-in
    /// for `newTextureWithDescriptor:iosurface:plane:`) whose rows overlap on Metal's GPU: more
    /// than one row of more than 16 bytes, at a `bytesPerRow` whose multiple of 16 below it is
    /// shorter than the row. Metal makes it; its GPU reads the rows that far apart, and where two
    /// rows' writes meet, which one wins depends on the order of its writes (a blit's and a
    /// kernel's differ).
    OverlappingSurfaceRows,
    /// A render pass whose custom sample positions Vulkan takes only through what the runtime
    /// does not build: a depth/stencil attachment of a caller's image not made with
    /// `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`, of a format the device does
    /// not copy both ways, as a stand-in made with it is copied, a stencil-only one (`S8_UINT`,
    /// which Vulkan does not let take it) on a device none of whose combined formats renders
    /// through a scratch image made with it, or a device without `variableSampleLocations`, which
    /// takes them only at the pass's begin (`VkRenderPassSampleLocationsBeginInfoEXT`)
    /// (`sample_positions.rs`).
    SamplePositionsAtBegin,
    /// A shared event handle opened on a device other than its event's: another physical
    /// device's, or one made with other options or adopted (equal options make one device, as
    /// Metal's calls do). Vulkan semaphores do not cross devices unless exported, which the runtime
    /// does not do (no venue has two GPUs in one process).
    SharedEventAcrossDevices,
    /// A viewport depth range past `[0, 1]`, which Metal takes, on a device without the features
    /// to take it natively or by the vertex stage's compensation (a driver that draws the
    /// compensated depths inexactly refuses it `Unsupported`, `Feature::ViewportDepthRange`), or
    /// in depth clamp mode where its one clamp differs between the draw's viewports, or holds one
    /// depth that a fragment function reading its position's depth would read, that lies past
    /// `[0, 1]` without `VK_EXT_depth_range_unrestricted`, or that bias factors move Metal's depth
    /// off where the vertex stage does not clamp each vertex's depth or the device holds no
    /// Vulkan range past 1. Or for a draw whose fragment function runs its depth test itself, where
    /// a compensated depth replaces the draw's range, its viewports have different ranges (the
    /// function reads one range), or in depth clamp mode the depth bias or a clamp of one depth
    /// moves the draw's range off the encoder's, which Metal clamps to.
    ViewportDepthRange,
    /// NaN or infinite depth bias factors whose effect differs between flat and sloped triangles,
    /// or a draw whose bias adds a clamp that moves a viewport's depth range past `[0, 1]` or that
    /// depth clamp mode bounds, where the vertex stage's compensation cannot add it after Metal's
    /// clamp of each vertex's depth: Metal takes them, and no static Vulkan state gives its answer.
    NonFiniteDepthBias,
    /// A draw under a depth compare of `Never` in a pass without a depth attachment, which fails
    /// every fragment, whose fragment function writes color attachments as storage images (a
    /// framebuffer fetch in the pixel interlock) or reaches the pass's imageblock: those writes
    /// are in the function's body, which Metal runs, and no write mask stops them. Also, under
    /// `[[early_fragment_tests]]`, a stencil face whose operation depends on the stored value, of a
    /// function that writes memory or such attachments, which Metal never runs. Every other such
    /// draw is drawn.
    DepthNeverWithoutAttachment,
    /// A draw whose depth/stencil state uses depth or the stencil that the pass lacks, which
    /// plain Metal tests against a hidden `float` depth of 1.0 and stencil of 0 (Metal's
    /// validation stops it), where the encoder gives the pass no hidden depth and stencil: a pass
    /// with a depth or a stencil attachment that lacks the other aspect (its texture is the
    /// pass's), a pass without attachments or over a rasterization-rate map, and a device that
    /// clears no `Depth32Float_Stencil8` image by a transfer. An indirect command buffer's
    /// command draws as the encoder's own draw at its execution's call would, and is refused so
    /// where that draw would be.
    HiddenDepthStencil,
    /// An indirect draw whose arguments the binding plan asks to read on the CPU
    /// (`DrawCommands::indirect_needs_arguments`): an indexed draw whose index fetch nothing
    /// bounds (no `robustBufferAccess2`, or an index buffer offset at or past its length), whose
    /// exact path is a GPU pass that writes the indices the draw fetches into a scratch, zero past
    /// the Metal buffer, before the draw; or a device that emulates base vertex and instance, draws
    /// multiview, or lacks `drawIndirectFirstInstance`, whose path is a prepass that writes what
    /// the draw needs. The runtime never reads arguments an earlier command may write. An index
    /// bounds pass whose arguments or indices a draw of the encoder wrote since its last memory
    /// barrier, which Metal does not order before the draw's fetch, is refused too. So is an
    /// indirect patch draw whose post-tessellation function reads `[[base_instance]]` through the
    /// device's tessellation stages, which take it from push constants; on a device without
    /// `tessellationShader` the draw's tessellation scan reads it on the GPU, and none is refused.
    IndirectDrawArguments,
    /// A depth/stencil resolve no Vulkan resolve mode of the device states: a filter or pair of
    /// filters the device's `supportedDepthResolveModes`, `supportedStencilResolveModes` or
    /// `independentResolve` lack, depth and stencil resolved into different textures, or the
    /// stencil filter `DepthResolvedSample` after the depth filter `Min` or `Max` of an
    /// attachment whose image is not sampled (the runtime draws it from the samples otherwise). A
    /// resolve draw is the exact path.
    DepthStencilResolve,
    /// The tile memory of a render pass of a sample count Apple's GPUs lack (theirs are 1, 2
    /// and 4), which a Vulkan device may have: a pass that sets its tile size, imageblock sample
    /// length or threadgroup memory length, or an encoder's tile size asked, where Metal's
    /// answers are unmeasured (`tile_pass.rs`).
    TileAtSampleCount,
    /// A draw or tile dispatch whose imageblock is of another size than the pass's, or a draw
    /// whose fragment function's layouts are of another alignment than the pass's earlier
    /// draws': Metal places the explicit imageblock at an offset its alignment sets, and what a
    /// layout of another size leaves in the bytes past it is read only for a draw over every
    /// pixel (`fragment_imageblock.rs`). Views of another master of the same size and alignment are
    /// drawn.
    ImageblockMasters,
    /// A render pass that reaches its imageblock storage, from a draw whose fragment function
    /// declares an explicit imageblock or a tile dispatch whose function's explicit imageblock
    /// stands alone, and dispatches a tile function whose explicit imageblock is aliased onto its
    /// implicit one past the bytes the attachments pack to: the runtime keeps the aliased layout
    /// in threadgroup memory of its dispatch, staged from and to the attachments, where Metal's
    /// reaches the explicit imageblock after them (`fragment_imageblock.rs`). A layout within the
    /// attachments' bytes is dispatched.
    TileAndFragmentImageblocks,
    // The pieces the GPU-family audit names as blocking rows of Apple's feature set tables
    // (`runtime::families`). Each says the step that builds it, or that none is scheduled.
    /// Metal 4's API (`MTL4CommandQueue`, `MTL4CommandBuffer`, `MTL4ArgumentTable`,
    /// `MTL4Compiler` and the rest), with the features only its programming model offers: the
    /// Metal4 family's one row. Not scheduled.
    Metal4Api,
    /// Counter sample buffers (`MTLCounterSampleBuffer`) and the commands that sample into them.
    /// Not scheduled: a command buffer's GPU times are built.
    CounterSampleBuffers,
    /// Residency sets (`MTLResidencySet`). Not scheduled.
    ResidencySets,
    /// A sparse texture's access counters (`getTextureAccessCounters:...` and
    /// `resetTextureAccessCounters:...` of a blit encoder). Not scheduled (orchestrator,
    /// 2026-10-06): Metal's counts are its texture hardware's units, not a count of accesses,
    /// nondeterministic for large dispatches and different on an M3 Max and an M4 (a Metal sparse
    /// probe on both), so no exact answer exists, and Vulkan counts nothing of the kind.
    TextureAccessCounters,
    /// Fast resource loading: `MTLIOCommandQueue`, its command buffers and file handles. Not
    /// scheduled.
    IoCommandQueues,
    /// Texture view pools (`MTLTextureViewPool`). Not scheduled.
    TextureViewPools,
    /// Color attachment mapping (`MTLLogicalToPhysicalColorAttachmentMap`). Not scheduled.
    ColorAttachmentMapping,
    /// Lossy texture compression (`MTLTextureCompressionTypeLossy`). Not scheduled.
    LossyTextureCompression,
    /// Per-pipeline shader validation (`MTLShaderValidation`). Not scheduled.
    ShaderValidation,
    /// Shader logging (`MTLLogState`, a shader's `os_log`). Not scheduled.
    ShaderLogging,
    /// Device notifications (`MTLCopyAllDevicesWithObserver`). Not scheduled.
    DeviceNotifications,
    /// A draw in a rate-mapped pass that the region path does not draw, for the compile layer's
    /// reason (`rate_mapped_pass.rs`).
    RateMapRegions(RegionRefusal),
    /// A layered render pass (`renderTargetArrayLength` 2 or more) that needs a step of one
    /// layer: a color resolve from the sample planes (`color_resolve.rs`). Resolves, separate
    /// depth and stencil textures (copied by draws once per layer) and stand-ins take every layer.
    LayeredPass,
    /// A blit copy between overlapping regions of one texture's texels (`copyFromTexture:...` with
    /// the source's region meeting the destination's, through the texture or views of it), other
    /// than a region copied onto itself, which keeps its texels. Not scheduled: no exact answer
    /// exists. Metal's copy races: small regions read as `memmove` would, but larger ones read
    /// texels another part of the copy has already written, differently from run to run, on an
    /// M3 Max and an M4, with and without API validation (probes).
    OverlappingTextureCopies,
    /// `Buffer::rebind_bytes_no_copy` of a no-copy buffer with a live texture over its bytes,
    /// given memory at another offset in its page than the old: the texture's copies and texel
    /// views are planned at the bytes' offset in their `VkBuffer` (`rebind.rs`). Also a rebind
    /// of a no-copy buffer or heap with a texture whose image is bound to its bytes, where the
    /// device makes no such image over the new memory: over memory it imports by its pointer, one
    /// that takes the import's type and lies where the texture's texels do, its rows (and slices)
    /// as far apart as the old image's, and never for a heap's texture whose image lay over
    /// dma-bufs, whose rows the driver's `DRM_FORMAT_MOD_LINEAR` layout spaces; over memory it
    /// imports as dma-bufs (`HostMemory::with_dma_buf`), an image of `DRM_FORMAT_MOD_LINEAR` over
    /// a dma-buf of the texture's own new pages, which needs `VK_EXT_image_drm_format_modifier`
    /// and `VK_EXT_queue_family_foreign`, a layout the driver takes and the source's dma-buf.
    RebindTextures,
    /// A PVRTC texture's blocks in a Metal block order the runtime does not lay out:
    /// `replaceRegion` and `getBytes` of a region other than a whole level, which Metal's API
    /// validation stops and plain Metal places by no rule the runtime has found, and of a whole
    /// level at a `bytesPerRow` other than 0 and the level's row, unmeasured. A whole level's take
    /// its blocks in Z order with the y bit lowest. A blit's copy at a nonzero `bytesPerRow` moves
    /// the blocks row by row, and one at 0 lays them out in the region's Morton order, a region off
    /// its blocks that of the blocks it is rounded to, with `MTLBlitOptionRowLinearPVRTC` or
    /// without it, as plain Metal's does (its API validation stops all but the first with the
    /// option).
    PvrtcBlockOrder,
    /// A heap's or a sparse heap's texture of an ASTC format on a device without that ASTC format
    /// (`astc.rs`), or a sparse heap's texture of an ETC2 or EAC format on a device without them
    /// (`etc.rs`) or of a 4:2:2 format (`chroma_pairs.rs`), which Metal makes: an ASTC texture's
    /// blocks would lie in the heap's bytes, which its decoded twin would decode as the texture is
    /// placed, and a sparse heap's tiles are mapped by GPU commands. A heap's size query answers
    /// an ASTC one Metal's model, which no image of the device decides. A texture of its own takes
    /// them
    /// (`Degradation::CarriedDecodedTwin`), and so does an ETC2, EAC or 4:2:2 texture of a heap.
    DecodedHeapTexture,
    /// A draw (`drawPrimitives:` and its forms, or an indirect command buffer's command) that
    /// may read a vertex attribute the pipeline fetches in its vertex function past `2^32` bytes
    /// of its buffer's bind offset, in a buffer that runs past them: Metal reads the attribute at
    /// `element · stride + offset` in 64 bits, where the fetch's displacement wraps at `2^32`
    /// (`render/vertex_reach.rs`). An indexed draw of 32-bit indices, or an indirect draw, may
    /// read any element its indices or arguments name. A draw past a buffer of at most `2^32`
    /// bytes past its bind offset is drawn, as Metal reads outside the buffer there.
    VertexFetchPast4GiB {
        /// The Metal vertex buffer index (`setVertexBuffer:offset:atIndex:`'s index).
        buffer_index: u32,
        /// The furthest byte past the bind offset the draw may read, saturating.
        reach: u64,
    },
    /// A dispatch (`dispatchThreads:threadsPerThreadgroup:` and its forms, or an indirect
    /// command buffer's command) whose kernel may read a `[[stage_in]]` attribute past `2^32`
    /// bytes of its buffer's bind offset, in a buffer that runs past them: Metal fetches it at
    /// `element · stride + offset` in 64 bits, where the fetch's displacement wraps at `2^32`
    /// (`fetch_reach.rs`). The furthest element is the stage-in region's origin plus the grid's
    /// last thread position over the step rate, clamped to the region's last element; an
    /// indexed step may read its index type's largest index, and a dispatch whose grid GPU
    /// memory states any 32-bit position. A dispatch past a buffer of at most `2^32` bytes past
    /// its bind offset runs, as Metal reads outside the buffer there.
    StageInFetchPast4GiB {
        /// The Metal buffer index (`setBuffer:offset:atIndex:`'s index) of the attribute's
        /// layout.
        buffer_index: u32,
        /// The furthest byte past the bind offset the dispatch may read, saturating.
        reach: u64,
    },
    /// A command that would sample a texture of an address-mode map
    /// (`TextureDescriptor::address_mode_map`) where the runtime cannot map it at bind
    /// (`address_map.rs`): through a constexpr sampler whose modes the map changes, through a
    /// sampler slot whose textures in one stage map it to different modes, or by a function
    /// sampling handles that can reach it.
    MappedTextureSampling {
        /// The texture's map.
        map: AddressModeMap,
        /// How it would be sampled.
        through: MappedSampling,
    },
    /// A library compiled from Metal Shading Language source. `newLibraryWithSource:options:error:`
    /// and its completion-handler form ([`Device::new_library_with_source`]) are refused: the
    /// runtime takes compiled shaders only, a metallib ([`Device::new_library_with_data`]) or AIR
    /// through the translator, and builds no MSL compiler. Precompile the source with Apple's
    /// `metal` tool and load the metallib.
    ///
    /// [`Device::new_library_with_source`]: crate::Device::new_library_with_source
    /// [`Device::new_library_with_data`]: crate::Device::new_library_with_data
    LibrarySource,
    /// An indexed strip after `set_primitive_restart_enabled` with an index Vulkan does not
    /// restart at, which the runtime draws as a list an index rewrite pass writes
    /// (`index_rewrite.rs`), where the pass cannot run as Metal draws: a direct draw whose list
    /// is past what one storage range binds or a `u32` counts, or whose base instance the
    /// device's indirect draws cannot carry (without `drawIndirectFirstInstance`); a draw whose
    /// indices or arguments a draw or tile dispatch of the encoder wrote since its last memory
    /// barrier, which Metal does not order before the draw's fetch, read in place; and an
    /// argument or index range the binding layer cannot bind.
    RestartedStrip,
    /// A draw of one of Metal's private primitive types (`DrawPrimitiveType`: a line loop, a
    /// triangle fan, quads, a quad strip), which the runtime draws as the list Metal draws, where
    /// that list cannot be drawn as Metal draws: a direct draw without indices whose list a `u32`
    /// cannot count; a draw over indices or from indirect arguments the index rewrite pass cannot
    /// draw, as for `RestartedStrip`; and a draw in the private list form (a vertex function that
    /// writes memory or computes `[[primitive_id]]` into a varying, or a direct draw whose last
    /// vertex ID passes `i32::MAX`) whose list has no device address.
    PrivatePrimitives,
    /// A draw under Apple's last provoking vertex (`setProvokingVertexMode:` 1) on a device
    /// without `VK_EXT_provoking_vertex`'s last vertex, which the runtime draws as the list of its
    /// primitives led by their last vertices (`render/provoking.rs`), where that list cannot be
    /// drawn as Metal draws, as for `PrivatePrimitives`: a direct draw without indices whose list
    /// a `u32` cannot count, or whose last vertex ID passes `i32::MAX` (drawn in the private list
    /// form) where its list has no device address; and a draw over indices or from indirect
    /// arguments the index rewrite pass cannot draw, as for `RestartedStrip`.
    ProvokingVertex,
    /// An indexed draw (`drawIndexedPrimitives:`) of a pipeline whose fragment function reads
    /// `[[primitive_id]]` on a device whose fragment stage cannot declare it (neither
    /// `geometryShader` nor `tessellationShader`), which draws its indices as vertices from 0 in
    /// the de-indexed form, its vertex function reading the indices in place at their device
    /// address (`VariantKey::primitive_id_indices`): indices a draw of the encoder wrote since
    /// its last memory barrier, which Metal does not order before the draw's fetch and the draw
    /// would read as a copy, and indices without a device address.
    PrimitiveIdIndices,
    /// A rotated texture (`TextureDescriptor::rotation`) where the runtime does not build what
    /// Metal does: CPU access or a blit of a quarter-turned one whose rows, as those uses address
    /// them (the texture's `height` rows of `width` texels at the plane's `bytesPerRow`), run
    /// past its buffer, whose bytes Metal reads past the stored frame; a blit between such a
    /// texture whose rows overlap and a texture of an image; a view of a rotated texture other
    /// than the whole texture as one 8-bit RGBA format of another (`Texture::new_texture_view`); a
    /// binding beyond its usage; a plane whose offset or `bytesPerRow` is off 16 bytes, or a format
    /// of a texel size other than 1, 2, 4, 8 or 16 bytes, whose rotation is not measured; a store
    /// past the device's largest storage buffer range or dispatch grid; and a command that writes a
    /// rotated texture's bytes both through a render pass and another way, which no merge serves.
    RotatedTexture,
    /// A texture over a no-copy buffer's or heap's bytes (a heap's texture, or a texture over
    /// the buffer or over one of the heap's buffers) that no stand-in of zero bytes presents
    /// exactly while they are detached (a texture of a multi-planar format, a two-plane YCbCr
    /// texture's converted texels, a texture with a depth twin, a multisample texture):
    /// `detach_backing` of the bytes, and its making over them while they are detached. Also made
    /// over detached bytes: a rotated texture, and one over a surface's two planes. And
    /// `detach_backing` while a texture is being made over the bytes, or `rebind_bytes_no_copy` of
    /// detached bytes while one is.
    DetachTextures,
    /// A texture buffer of more than one level (`TextureDescriptor::mipmap_level_count`, up to
    /// its width's full chain) from the device's, a heap's or the shared creator. Metal makes it
    /// (an M4 under macOS 27.0.1 and an M3 Max under 26.7), its levels separate with no buffer
    /// behind it (`buffer` nil), each replaced, read and viewed on its own (the M4); the
    /// runtime's texture buffers are one level over a buffer. A buffer's creator stops it, as Metal's does,
    /// and the heap size query answers Metal's size, its first level's.
    TextureBufferLevels,
    /// A texture of `TextureDescriptor::write_swizzle_enabled` whose swizzle is not the default,
    /// with `ShaderWrite` or `RenderTarget` usage, from the device's, a heap's or a buffer's
    /// creator. Metal makes it (an M4 under macOS 27.0.1 and an M3 Max under 26.7), and its
    /// writes go through the swizzle (the M4); the runtime's swizzles are its sampled views'
    /// alone. The heap size query answers
    /// its size.
    WriteSwizzle,
    /// `Texture::replace_region` or `get_bytes` of a texture whose bytes (a no-copy buffer's or
    /// heap's) are detached (`detach_backing`). Metal's CPU access reaches the pages the texture
    /// took as it was made (a heap's texture, the pages the heap was made over), measured on an
    /// M4 under macOS 27.0.1; the runtime keeps no pages of detached bytes.
    DetachedTexelAccess,
    /// A use of a texture of a depth or stencil format (`Depth16Unorm`, `Depth32Float`,
    /// `Depth32Float_Stencil8`, `Stencil8`, `X32_Stencil8`) over a plane of a surface with planes
    /// (`Buffer::new_texture_with_iosurface_plane` with `IoSurfacePlane::planar`) that the
    /// runtime does not make as Metal does: a blit of the stencil of a `Depth32Float_Stencil8` one
    /// or of its `X32_Stencil8` view (that stencil is not in the plane; unmeasured), a render pass
    /// on an `X32_Stencil8` one's view as `Depth32Float_Stencil8` (unmeasured), and a pass on a
    /// device its tiled copies do not fit. The texture, its reads, samples, blits and CPU access,
    /// its passes at any extent, its views' passes and a pass loading its stencil are made (an
    /// M4 under macOS 27.0.1).
    SurfaceDepthStencil {
        /// The descriptor's pixel format.
        pixel_format: PixelFormat,
    },
    /// A blit copy between a buffer and a multisampled texture (`copyFromBuffer:...toTexture:`
    /// into, or `copyFromTexture:...toBuffer:` out of, a texture of more than one sample).
    /// Vulkan copies no multisampled image to or from a buffer (its `vkCmdCopyBufferToImage` and
    /// `vkCmdCopyImageToBuffer` require a sample count of one,
    /// VUID-vkCmdCopyBufferToImage-dstImage-07973), so the single-sample buffer-image path cannot
    /// take it. Plain Metal copies it per sample, the pixel's samples consecutive in the buffer:
    /// texel `(x, y)` sample `s` of an `n`-sample texture of `bpp`-byte samples lies at
    /// `offset + y * bytesPerRow + (x * n + s) * bpp`, whole or into a sub-region, where the
    /// offset and `bytesPerRow` are multiples of the pixel's `n * bpp` bytes, rows overlapping
    /// into the texture where `bytesPerRow` is below a row; at others it rounds the first row's
    /// offset and the pitch down (measured on an Apple M3 Max under macOS 27.0.1, for 1- to
    /// 16-byte colour, depth and stencil samples at 2 and 4 samples; its API validation refuses an
    /// offset off the pixel). The runtime places the rows as Metal does, through a staging buffer
    /// where a row's offset in the `VkBuffer` is off a sample (`staged-texture-copy`), but for the
    /// layouts of `MultisampledBufferLayout`. The runtime runs that copy exactly
    /// (`sample_interleave.rs`) through a single-sample wide image and a multisampled uint scratch,
    /// by a compute split and a sample-rate merge draw. This is the refusal where the device cannot
    /// run that: no `sampleRateShading`, no integer multisample color attachment at the count, no
    /// storage of the copy format, or a texture without the transfer usage the copy needs. A depth
    /// or stencil texture's samples cross the same images in the unsigned format of the aspect's
    /// bits and move into or out of the aspect as the multisampled blit copies between
    /// depth/stencil and colour move them (maintenance8's copies, or sample-rate draws); this is
    /// also the refusal where that move refuses: on a device whose maintenance8 copies are missing
    /// or wrong, a 16-bit depth on a driver whose codes the fragment-depth draw is not measured to
    /// keep (`DriverFact::FragmentDepthKeepsDepth16Codes`), or a depth or stencil texture the
    /// draws cannot sample or render to (a 32-bit float depth on a driver whose fragment depth is
    /// not measured is drawn, announced as `drawn-depth-copy`; a 16-bit depth where its codes are
    /// kept crosses as its codes, exact). Nothing is recorded, where previously the runtime
    /// recorded invalid Vulkan and some drivers hung instead of failing (the consumer's NVIDIA
    /// Xid 32 stall).
    MultisampledBufferCopy,
    /// A blit copy between a buffer and a multisampled texture (as for `MultisampledBufferCopy`)
    /// in one of two buffer layouts that plain Metal copies by rules not yet reduced, and that
    /// Metal's API validation rejects (`bytesPerRow` must be at least a row and a multiple of the
    /// pixel, a texel's samples' bytes): a `bytesPerRow` of 0 from an offset that is a multiple of
    /// 16 bytes (but for a single row from offset 0), where plain Metal copies the texture's
    /// samples in a block order of its own; and, for a pixel of 32 bytes or more, a
    /// `bytesPerRow` below a row, or off the pixel for more than one row, where plain Metal places
    /// the rows by the offset's and the pitch's bits together (measured on an Apple M3 Max under
    /// macOS 27.0.1). The runtime copies every other layout as plain Metal does.
    MultisampledBufferLayout,
    /// What a render pass the device draws in color groups does not build: a pass of more color
    /// attachments with a texture than the lesser of the device's `maxColorAttachments` and
    /// `maxFragmentOutputAttachments` (four on a device at Vulkan's floor), or of one at an index
    /// at or past it, where Metal takes eight, draws each group's colors in a pass of its own
    /// (`render/color_groups.rs`). Refused there: a render pipeline state whose colors with a
    /// pixel format are not the pass's colors with a texture (or one the device draws in groups
    /// in a pass it draws in one), a tile pipeline's state, an execution of render commands in a
    /// pass of several groups, a depth or stencil the pass loads from a texture without the
    /// transfer usage its later groups' copy needs, and in a pass of several groups a draw that
    /// reads what another group writes (a color of another group, its own group's colors or the
    /// depth and stencil where it draws in a later group too, a resolve texture, the visibility
    /// result buffer, memory it reaches through a handle table, or a texture over a buffer's
    /// bytes it may reach through a root table), and a resolve target set on a texture a draw of
    /// the pass read.
    ColorAttachmentGroups,
}

/// The kind of memory that ran out.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryKind {
    /// Host memory (`VK_ERROR_OUT_OF_HOST_MEMORY`).
    Host,
    /// Device memory (`VK_ERROR_OUT_OF_DEVICE_MEMORY`), or no free range of the runtime's Metal
    /// address space.
    Device,
    /// An automatic heap with no free range the resource fits, for which Metal answers nil.
    Heap,
}

/// A raw `VkResult` of a failed call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VkResultCode(pub i32);

/// Why a command buffer failed on the device (`MTLCommandBufferError`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandBufferError {
    /// The device was lost while the command buffer ran.
    DeviceLost,
    /// The device ran out of memory for the submission.
    OutOfMemory,
    /// Any other Vulkan failure of the submission, raw.
    Internal(VkResultCode),
    /// An indexed indirect draw's index bounds or index rewrite pass found the rest of the
    /// command buffer's index pool too small for the draw's indices, and drew nothing of the draw.
    /// Metal draws it; the pool (a sixty-fourth of the device's largest device-local heap) is bound
    /// before the GPU reads the draw's count, so nothing is drawn short silently.
    IndexPoolOverflow,
    /// A function the command buffer ran sampled, gathered, or queried a texture through a
    /// sampler state an argument buffer named whose min or mag filter is `Cubic`
    /// (`MTLSamplerMinMagFilterCubic`). Metal filters through it bicubically; the runtime does not
    /// emulate that filter through an argument buffer, so the operation's result is not Metal's and
    /// the command buffer fails (`NotImplemented(SamplerBicubicFilter)`'s reach, decided where the
    /// function ran). A command buffer whose functions never reach such a sampler runs as Metal's.
    CubicArgumentSampler,
    /// An emulated mesh draw's records, or an emulated indirect draw's object slots (one per object
    /// threadgroup, the grid and payload its object function writes), did not fit the rest of the
    /// command buffer's mesh records pool, or its vertices passed a `u32`: such draws drew nothing. Metal draws them; the
    /// runtime's pool is the device's budget, so nothing is drawn short silently. The first such
    /// draw, as its scan saw it.
    MeshRecordsOverflow(MeshRecordsOverflow),
    /// An emulated tessellation draw's records (an indirect patch draw's, on a device without
    /// `tessellationShader`) did not fit the rest of the command buffer's records pool, the one
    /// emulated mesh draws share: such draws drew nothing. Metal draws them; the runtime's pool is
    /// the device's budget, so nothing is drawn short silently. The first records fault of the
    /// command buffer, where it is a tessellation draw's, as its scan saw it.
    TessellationRecordsOverflow(TessellationRecordsOverflow),
    /// The device cannot launch a mesh draw's grid that Metal draws, and such draws drew
    /// nothing: on a device that does not fold object grids (without device addresses,
    /// `shaderInt64`, `multiDrawIndirect` or `drawIndirectCount`), an indirect draw's object grid,
    /// as the GPU wrote it, past the task stage's limits (`maxTaskWorkGroupCount`,
    /// `maxTaskWorkGroupTotalCount`), or a mesh grid past the mesh stage's; on one that folds them,
    /// a grid of more pieces than `MeshArguments::OBJECT_PIECES` (only under task limits below
    /// Vulkan's required least); on a device without `VK_EXT_mesh_shader` (the emulation), an
    /// indirect draw's object grid past the device's `maxComputeWorkGroupCount`. Nothing is drawn
    /// short silently. A direct draw of such a grid on
    /// a device that does not fold is refused as it is encoded (`Unsupported{MeshShading}` naming
    /// the limit). Not Metal's failure: an object grid Metal's GPU discards is
    /// [`GpuRecoveryVictim`](Self::GpuRecoveryVictim).
    MeshGridPastLimits,
    /// A host step refused a command it recorded from an indirect command buffer's records
    /// (`Degradation::HostReplayedIndirectCommands`): what `commit` refuses where the CPU wrote
    /// the records, such as a command's pipeline state of other color formats than the pass's.
    /// The batch the step recorded for, and those after it, did not run.
    /// [`CommandBuffer::refusal`](crate::CommandBuffer::refusal) gives the refusal, typed as
    /// `commit`'s would be.
    Refused,
    /// The program committed the command buffer as failed, with this `MTLCommandBufferError`
    /// code ([`CommandBuffer::commit_failed`](crate::CommandBuffer::commit_failed)). Of Metal's
    /// codes of a page fault (3) and a hang (2), it counts against its queue as the error it
    /// stands for does ([`AccessRevoked`](Self::AccessRevoked)), and its completion is the GPU's
    /// recovery from it, which ends a predicated loop running on the device (as a victim after a
    /// fault, with no error after a hang); of any other code, neither.
    Injected(u64),
    /// The command buffer encoded a render pass without attachments whose
    /// `renderTargetWidth` or `renderTargetHeight` is past 16384, which Metal's encoder takes:
    /// Metal fails the command buffer from that pass on with `MTLCommandBufferErrorInternal`
    /// ("Internal Error (0000010d:Internal Error)" on an M4, `0000010c` on the M3 Max), the
    /// encoders before it run and the event signals encoded before it fire, while the pass,
    /// every encoder after it and the signals after it do not (with and without API validation,
    /// which lets the pass through). The queue runs on.
    RenderTargetSize,
    /// Metal's GPU faulted an encoder, which ran nothing, not even a render pass's load, clear or
    /// store actions: a render encoder's execution of an indirect command buffer that inherits
    /// its pipeline state, under the encoder's state made without
    /// `supportIndirectCommandBuffers` (a range of commands that runs, or any range read from a
    /// buffer, `{0, 0}` included), or an indexed draw of primitive type 9
    /// ([`DrawPrimitiveType::Nine`](crate::DrawPrimitiveType::Nine)), of any index count, of no
    /// index or instance too. Metal completes the command buffer with
    /// `MTLCommandBufferErrorPageFault` ("Caused GPU Address Fault Error
    /// (0000000b:kIOGPUCommandBufferCallbackErrorPageFault)" on an M4); the encoders before the
    /// faulted one run, every event signal fires, and the queue runs on. On the M3 Max (macOS
    /// 27.0.1; 120 command buffers of type-9 draws, this error's and
    /// [`GpuRecoveryVictim`](Self::GpuRecoveryVictim)'s, and 44 of such executions) it completed
    /// 6.86 to 10.3 ms after it read `Scheduled`, and the encoders after the faulted one were
    /// lost to the GPU's recovery where the GPU ran just before (127 of 142), ran where it was
    /// idle before (the first fault after 3 s idle), and in some shapes ran in part (an encoder
    /// after an event signal, a command buffer committed right after). The runtime answers the
    /// steady state: the faulted encoder and every one after it run nothing, and the command
    /// buffer completes no sooner than 6.86 ms after it read `Scheduled` and the work before the
    /// encoder ran, its queue's command buffers committed meanwhile reading `Scheduled` as they
    /// are committed, and running (`command_buffer/hang.rs`).
    ///
    /// Such a fault, as the runtime makes it, counts against the command buffer's queue, as a
    /// [`GpuHang`](Self::GpuHang) does: from the completion of the queue's second, its later
    /// command buffers fail with [`AccessRevoked`](Self::AccessRevoked). A relayed one is the
    /// driver's queue's.
    ///
    /// Also a page fault Metal answers for the command buffer's work under MoltenVK, which the
    /// runtime relays (`CommandBufferError::Execution`).
    PageFault,
    /// Metal's GPU hung at a direct draw of primitive type 9 of 3 vertices or more
    /// ([`DrawPrimitiveType::Nine`](crate::DrawPrimitiveType::Nine)) and failed the command buffer
    /// with `MTLCommandBufferErrorTimeout` (2, "Caused GPU Hang Error
    /// (00000003:kIOGPUCommandBufferCallbackErrorHang)"). On the M3 Max (macOS 27.0.1, 33 hangs)
    /// it read `Scheduled` 0.68 to 13.2 ms after its commit and failed 28.3 to 46.2 ms after it,
    /// mostly 27.5 to 28.9 ms after `Scheduled`: the encoders before the draw's ran; its render
    /// encoder, not even its pass's load, clear or store actions, and every encoder after it ran
    /// nothing; its event signals fired. Every command buffer of its queue scheduled while it was
    /// in flight completed with no error together with it, committed at once, after it read
    /// `Scheduled`, or 20 ms on, and with an event wait met during the hang, by the host or
    /// another queue's command buffer; their work ran or not by a race (mostly ran where the hang
    /// read `Scheduled` within 2 ms, mostly not where later). One whose event wait was met only
    /// after it completed, one of another queue and one committed later ran. The runtime's hang
    /// reads `Scheduled` 5.06 ms after its commit (later where the work before the draw is
    /// submitted later), completes no sooner than 27.5 ms after it read `Scheduled` and the work
    /// before the draw ran, and runs nothing of its queue's others: the outcome of Metal's
    /// slower submission (`command_buffer/hang.rs`). It counts against its queue as a
    /// [`PageFault`](Self::PageFault) does.
    ///
    /// Metal's GPU also hangs at a render pass of no draw call over a `Mutable`
    /// rasterization-rate map where no command buffer with a draw call (one of nothing too) over
    /// such a map was committed on the device before its own (the M3 Max under macOS 27.0.1, our
    /// probe `rs`: maps of layers and of none, reset or not; a pass with only a pipeline state
    /// set; one after a drawn pass in the same command buffer), failing it with the same error.
    /// The runtime fails it from that pass on as at the draw.
    GpuHang,
    /// Metal failed the command buffer as a victim of its GPU's recovery, at an indirect draw of
    /// primitive type 9, indexed or not, whatever its arguments
    /// ([`DrawPrimitiveType::Nine`](crate::DrawPrimitiveType::Nine)), with
    /// `MTLCommandBufferErrorInternal` (1, "Discarded (victim of GPU error/recovery)
    /// (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)", its `GPUStartTime` 0), 16.9 to
    /// 30.0 ms after its commit (the M3 Max, macOS 27.0.1, 5 draws): its render encoder ran
    /// nothing, every event signal fired, and the queue ran on. The encoders after it, its
    /// completion and the runtime's answer are a [`PageFault`](Self::PageFault)'s. It does not
    /// count against its queue.
    ///
    /// Also an object grid of more than 2^29 threadgroups (`bind::METAL_OBJECT_GRID_TOTAL`), past
    /// what Metal's GPU runs within its watchdog's 5 seconds (2^29 of an object function that does
    /// next to nothing ran in 4.98 s on the M3 Max, 2^30 did not), which Metal fails with the same
    /// error 5 seconds on: nothing of the command buffer ran from the draw's encoder on (its clear,
    /// its draws before the grid's, the encoders after it), and a command buffer of the queue
    /// committed while it ran failed too. The runtime fails a direct draw's command buffer from
    /// its encoder on as its work completes, and an indirect draw's as its arguments pass reports
    /// it, the rest of its work running; it runs the queue's other command buffers (docs/API.md,
    /// "Accepted differences").
    GpuRecoveryVictim,
    /// Metal ignored the command buffer, committed on a queue whose second command buffer with a
    /// GPU error it caused ([`PageFault`](Self::PageFault), [`GpuHang`](Self::GpuHang), or
    /// [`Injected`](Self::Injected) of Metal's code of either; not a
    /// [`GpuRecoveryVictim`](Self::GpuRecoveryVictim)) had completed:
    /// `MTLCommandBufferErrorAccessRevoked` (4, "Ignored (for causing prior/excessive GPU errors)
    /// (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)" on the M3 Max, macOS
    /// 27.0.1). It reads `Scheduled` and completes at its commit, nothing of it runs and none of
    /// its event signals fires; so does every later one of the queue, while the device's other
    /// queues, new ones too, run on (`command_buffer/revocation.rs`).
    AccessRevoked,
    /// Metal failed the command buffer's work with this `MTLCommandBufferError` code, which the
    /// driver relayed: MoltenVK, which runs each batch as a Metal command buffer, reports a
    /// failed one, and the runtime gives its code to the command buffer whose batch it was
    /// (`device/relay.rs`), the device kept, as Metal keeps it. Measured: a GPU write into
    /// read-only host pages fails with `Internal` (1, "Discarded (victim of GPU error/recovery)")
    /// on an M4 with macOS 27.0.1, as does a command buffer the GPU ran beside it, and with
    /// `PageFault` on an M3 Max with 26.7 (the consumer's). Code 3 is
    /// [`CommandBufferError::PageFault`], and 11 [`CommandBufferError::DeviceLost`].
    Execution(u64),
    /// The driver did not make the `VkPipeline` of a draw the command buffer recorded, which
    /// its submission waited for: the runtime makes a draw's pipeline while the draw goes on
    /// and the command buffer's batch waits for it, so the failure surfaces here, not at the
    /// draw. The batch that records the draw, and those after it, did not run.
    /// [`CommandBuffer::failed_pipeline`](crate::CommandBuffer::failed_pipeline) names the
    /// state and gives the driver's error. Metal makes a state's pipelines as it makes the state,
    /// and has no such failure.
    PipelineCreation(PipelineCreationCause),
    /// A command of the command buffer named a buffer over the caller's host pages (a no-copy
    /// buffer, a heap over the caller's memory, or a resource placed in one) that was rebound
    /// (`rebind_bytes_no_copy`) after the command was encoded and before the command buffer's
    /// first batch was submitted: a binding, a blit's operand, an attachment, a vertex, index or
    /// indirect buffer, a texture over the pages, an indirect command's resolved buffer. Metal
    /// resolves such a use as it submits the command buffer, to the pages the buffer lies in
    /// then; the runtime resolved it as the command was encoded. The rebind encodes the command
    /// buffer's compute, blit and render encoders that named it again, from their calls, so they
    /// reach the new pages as Metal's do (a resource state encoder takes no pages: an indirect
    /// mapping update reads its buffer as it runs; a call refused at its encoding is refused
    /// again, and the calls after it run); this fails only an encoder that cannot be encoded
    /// again: one of a command buffer that used command predication; a render pass recorded later
    /// (an indirect command buffer's execution, a draw whose pipeline was being made) or whose end
    /// captured its draws' tile reach. Each of Metal's submissions holding such an encoder ran
    /// nothing (the command buffer's split), every other ran, the event signals fired and the
    /// handlers ran. What the command buffer's encoders made resident
    /// (`use_resource`, `use_heap`) and what its commands reach unnamed follow the rebind as
    /// Metal's do. The device announces the first (`Degradation::EncodedUseRebound`). A command
    /// buffer submitted before the rebind keeps the old pages, as Metal's does.
    EncodedUseRebound,
    /// A command of the command buffer used a buffer over the caller's host pages (a no-copy
    /// buffer, a heap over the caller's memory, or a buffer placed in one) that was detached
    /// (`Buffer::detach_backing`, `Heap::detach_backing`) when the command buffer's submission
    /// holding the command was submitted (each event group's as its first batch goes): named (a
    /// binding, a blit's operand, a vertex, index or indirect buffer) or made resident
    /// (`use_resource`, `use_heap`, whether or not a command reaches it). Each of Metal's submissions holding such a command
    /// ran nothing (the command buffer's split), every other ran, the event signals fired and the
    /// handlers ran, as Metal fails a submission that uses a resource whose memory was taken away
    /// (`OutOfMemory`, code 8; the consumer's rebind probe, an M3 Max under macOS 26.7). A reach of
    /// the buffer by its address alone reads zero and fails nothing. An encoder that named a
    /// detached buffer recorded nothing from that command on, so it fails alike where the buffer
    /// was given pages again before the submission, with [`CommandBufferError::EncodedUseRebound`].
    DetachedBacking,
    /// An encoder of the command buffer was made under protection options other than 0
    /// (`CommandBuffer::set_protection_options`; the options set, here), or options from 0x10 up
    /// were set: as Metal's GPU fails it (measured by a consumer on an M3 Max, macOS 26.7), the
    /// first such encoder ran nothing, nor did any after it (with none, every encoder ran), and
    /// none of its event signals fired. Metal's code is 16 ("Protection Violation") below 0x10, 1
    /// ("Invalid Input") from 0x10 up.
    ProtectedEncoder(u64),
    /// A unit of command predication (`ComputeCommandEncoder::encode_start_if` and the others)
    /// ended with a block open that no misused record killed, where the unit's commands skipped
    /// as the GPU reached its end (a failed If or While, a held If's Else, a block inside one that
    /// skipped): Metal's GPU recovery fails the command buffer, `MTLCommandBufferErrorInternal`
    /// ("Discarded (victim of GPU error/recovery)
    /// (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)", which
    /// [`CommandBuffer::error_description`](crate::CommandBuffer::error_description) gives), on
    /// an M3 Max under macOS 27.0.1, `PageFault` in some runs. The runtime reads
    /// the unit's skip counter where it ends, on the GPU, and fails the command buffer as it
    /// completes, before its completed handlers run, and only so; a block whose commands ran there
    /// is harmless, as on Metal. The rest of the command buffer runs: how much of it Metal's
    /// recovery runs varies run to run.
    PredicationOpenAtUnitEnd {
        /// The index of the call that opened the outermost block left open in the first unit
        /// that failed so, among the command buffer's command-predication calls that returned
        /// `Ok` (`encode_start_if` and the other six), counting from 0. Metal's error names no
        /// record.
        record: u32,
    },
    /// The command buffer encoded a render pass without attachments whose
    /// `defaultRasterSampleCount`'s low 8 bits are 16 to 127, which Metal's encoder takes: Metal
    /// fails the command buffer with `MTLCommandBufferErrorInternal` ("Internal Error
    /// (0000010c:Internal Error)" on the M3 Max under macOS 26.7, the consumer's run of our probe,
    /// and `0000010d` on an M4 under macOS 27.0.1, plain Metal; its API
    /// validation stops the pass). The runtime fails it from that pass on, as for
    /// [`RenderTargetSize`](Self::RenderTargetSize), on every device,
    /// whatever samples its driver rasterizes; which encoders and signals Metal runs was
    /// measured for that size alone.
    DefaultRasterSampleCount,
    /// A render encoder's call that the runtime refused as it lowered the call, after the call
    /// returned (`RenderLoweringPolicy::Worker`): a refusal the call's own checks of its
    /// arguments and the encoder's state could not find, such as the binder's, or a failure to
    /// make what the lowering needs. The call, by its Metal selector and its place among the
    /// encoder's calls ([`EncoderCall`]); [`CommandBuffer::refusal`](crate::CommandBuffer::refusal)
    /// gives the error the call would have returned under `RenderLoweringPolicy::AtCall`. The
    /// encoder and every encoder after it run nothing, and the event signals encoded from it on do
    /// not fire. Metal's validation stops the process at such a call instead. A commit whose own
    /// work refused it after `commit` returned is named `commit`
    /// ([`CommandBuffer::commit`](crate::CommandBuffer::commit)): none of the encoders runs.
    CallRefused(EncoderCall),
}

/// One call of a render encoder, as a refusal found after it returned names it
/// ([`CommandBufferError::CallRefused`]), and as a notice announced while the runtime lowered it
/// does (`DegradeNotice::call`). No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EncoderCall {
    /// The Metal selector of the encoder's method called (`"drawPrimitives:vertexStart:vertexCount:"`
    /// and its kin name their full form, as the method's documentation does).
    pub selector: &'static str,
    /// The call's place among the encoder's calls, from 0: every call of the encoder's methods
    /// counts, `end_encoding` and the queries (`tile_width`, `tile_height`) but none. The end of an
    /// encoder that logged its calls names its notices `endEncoding`, at the place after its last
    /// call. A commit whose own work, run after it returned, refused it is `commit` at 0
    /// (`CommandBuffer::commit`).
    pub index: u32,
}

impl EncoderCall {
    /// Call `index` of its encoder, to the method of Metal's `selector`.
    pub(crate) fn new(selector: &'static str, index: u32) -> EncoderCall {
        EncoderCall { selector, index }
    }
}

/// Whose failure a command buffer's error is ([`CommandBufferError::origin`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FailureOrigin {
    /// Metal fails such work too, with the code [`CommandBufferError::metal_code`] gives: a
    /// failure Metal relayed through the driver ([`CommandBufferError::Execution`],
    /// [`CommandBufferError::PageFault`]), a lost device (Metal's `DeviceRemoved`, where Metal may
    /// report a hang or a fault of the work and keep the device), the device's memory, a failure
    /// measured on Metal ([`CommandBufferError::RenderTargetSize`],
    /// [`CommandBufferError::DetachedBacking`], [`CommandBufferError::DefaultRasterSampleCount`]),
    /// or the code the program committed the command
    /// buffer as failed with ([`CommandBufferError::Injected`]).
    Metal,
    /// The runtime's own: Metal runs the work and fails no command buffer so, where the runtime
    /// failed it rather than run it short or wrong (a fixed pool's overflow, a pipeline made at
    /// the draw, a read or sampler the runtime does not emulate there, a host step's refusal), or
    /// Vulkan failed the submission in a way Metal has no counterpart of
    /// ([`CommandBufferError::Internal`]). Its [`CommandBufferError::metal_code`] is the nearest of
    /// Metal's codes, not Metal's answer.
    Runtime,
}

/// Why the driver did not make a draw's `VkPipeline` ([`CommandBufferError::PipelineCreation`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineCreationCause {
    /// The device ran out of memory of this kind.
    OutOfMemory(MemoryKind),
    /// The Vulkan device was lost.
    DeviceLost,
    /// Any other Vulkan failure, raw.
    Vulkan(VkResultCode),
    /// The runtime's own broken invariant (`Error::Compile` of an internal error).
    Internal,
}

impl PipelineCreationCause {
    /// The cause of `error`, a pipeline's making's.
    pub(super) fn of(error: &Error) -> PipelineCreationCause {
        match error {
            Error::OutOfMemory(kind) => PipelineCreationCause::OutOfMemory(*kind),
            Error::DeviceLost => PipelineCreationCause::DeviceLost,
            Error::Vulkan(code) => PipelineCreationCause::Vulkan(*code),
            _ => PipelineCreationCause::Internal,
        }
    }
}

/// A draw's or a dispatch's `VkPipeline` the driver did not make
/// ([`CommandBufferError::PipelineCreation`]): the command, by its encoder and its order there;
/// the pipeline state it ran, and the variant of that state and the specialization its pipeline
/// took; and the error the making returned, which the command returned where the runtime made
/// the pipeline at the command. A command buffer keeps the first, in encoding order, as Metal
/// reports one error per command buffer.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct FailedPipeline {
    /// The command's render or compute encoder, by its place among the command buffer's ended
    /// encoders, in the order they ended, from 0: the index of
    /// [`CommandBuffer::ended_encoders`](crate::CommandBuffer::ended_encoders), as
    /// [`CommitFailure::encoders`](crate::CommitFailure::encoders) numbers them.
    pub encoder: usize,
    /// The command, by its order among its encoder's commands of its kind that returned `Ok`,
    /// from 0: a render encoder's draw calls (every `draw_*` call: of primitives, indexed,
    /// indirect, patches and meshes) or tile dispatch calls (every `dispatch_threads_per_tile*`
    /// call), a compute encoder's dispatch calls (every `dispatch_*` call, direct and indirect).
    pub draw: usize,
    /// The state the command ran.
    pub state: FailedState,
    /// The variant of the state the draw's bind shape asked for: the storage textures' formats, the
    /// buffers fetched in the shader, sRGB textures read raw, and the other choices the compile
    /// takes from what the draw binds.
    pub variant: metal2vulkan::cache::VariantKey,
    /// The value of each specialization constant of the draw's `VkPipeline`, by its meaning, in
    /// id order, each as its low bytes: among them the word of each bound sampler
    /// (`SpecKey::SamplerState`). The format class of each bound texture is not among them: its
    /// functions read it from a push field, and one `VkPipeline` serves every bound format.
    pub specialization: Vec<(metal2vulkan::pipeline::SpecKey, u64)>,
    /// The error the driver's creation returned.
    pub error: Error,
}

/// The pipeline state of a command whose `VkPipeline` the driver did not make
/// ([`FailedPipeline::state`]).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum FailedState {
    /// A draw's render pipeline state, or a tile dispatch's tile render pipeline state.
    Render(crate::RenderPipelineState),
    /// A dispatch's compute pipeline state.
    Compute(crate::ComputePipelineState),
}

/// The first emulated mesh draw of a command buffer whose records did not fit
/// ([`CommandBufferError::MeshRecordsOverflow`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshRecordsOverflow {
    /// The draw's order among the command buffer's emulated mesh and tessellation draws, from 0.
    pub draw: u32,
    /// Its object threadgroups (1 for a pipeline without an object function).
    pub object_threadgroups: u32,
    /// The mesh threadgroups it launched, saturated at `u32::MAX`; 0 where its object slots did
    /// not fit, before any launched.
    pub mesh_threadgroups: u32,
    /// The bytes of the command buffer's records pool, the device's budget.
    pub budget: u64,
}

/// The first emulated tessellation draw of a command buffer whose records did not fit
/// ([`CommandBufferError::TessellationRecordsOverflow`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TessellationRecordsOverflow {
    /// The draw's order among the command buffer's emulated mesh and tessellation draws, from 0.
    pub draw: u32,
    /// Its records, `patchCount · instanceCount`, saturated at `u32::MAX`.
    pub records: u32,
    /// The bytes of the command buffer's records pool, the device's budget.
    pub budget: u64,
}

impl CommandBufferError {
    /// Metal's `MTLCommandBufferError` code for this error: `MTLCommandBufferErrorDeviceRemoved`
    /// (11) for a lost device, `OutOfMemory` (8) for the device's memory, the index pool's and the
    /// records pool's (mesh and tessellation), and a detached buffer's, `AccessRevoked` (4) for a
    /// revoked queue's, `PageFault` (3), and
    /// `Internal` (1), also for the runtime's refusals (a cubic argument sampler's reach among
    /// them, a sampler state left out of an argument-sampler listing, a host step's object nothing
    /// made, a named use rebound before the submission, a predicated loop past its copies, and a
    /// mesh grid past the device's limits); a
    /// command buffer committed as failed, the code the program chose; a pipeline's failed making,
    /// its cause's; a failure the driver relayed, Metal's.
    ///
    /// The code is Metal's answer only where [`origin`](Self::origin) is
    /// [`FailureOrigin::Metal`]. A [`FailureOrigin::Runtime`] error's code is the nearest of
    /// Metal's codes, for a caller that must report one, where Metal fails no command buffer.
    pub fn metal_code(self) -> u64 {
        match self {
            CommandBufferError::Injected(code) => code,
            CommandBufferError::DeviceLost => 11,
            CommandBufferError::OutOfMemory
            | CommandBufferError::IndexPoolOverflow
            | CommandBufferError::MeshRecordsOverflow(_)
            | CommandBufferError::TessellationRecordsOverflow(_)
            | CommandBufferError::DetachedBacking => 8,
            CommandBufferError::Internal(_)
            | CommandBufferError::MeshGridPastLimits
            | CommandBufferError::Refused
            | CommandBufferError::RenderTargetSize
            | CommandBufferError::CubicArgumentSampler
            | CommandBufferError::EncodedUseRebound
            | CommandBufferError::PredicationOpenAtUnitEnd { .. }
            | CommandBufferError::DefaultRasterSampleCount
            | CommandBufferError::CallRefused(_) => 1,
            CommandBufferError::PageFault => 3,
            CommandBufferError::GpuHang => 2,
            CommandBufferError::GpuRecoveryVictim => 1,
            CommandBufferError::AccessRevoked => 4,
            CommandBufferError::PipelineCreation(cause) => match cause {
                PipelineCreationCause::OutOfMemory(_) => 8,
                PipelineCreationCause::DeviceLost => 11,
                PipelineCreationCause::Vulkan(_) | PipelineCreationCause::Internal => 1,
            },
            CommandBufferError::Execution(code) => code,
            CommandBufferError::ProtectedEncoder(options) => match options >= 0x10 {
                true => 1,
                false => 16,
            },
        }
    }

    /// Whose failure this is: [`FailureOrigin::Metal`] where Metal fails such work too, with the
    /// code [`metal_code`](Self::metal_code) gives, and [`FailureOrigin::Runtime`] where Metal
    /// runs the work and fails no command buffer, or Vulkan failed it in a way Metal has no
    /// counterpart of. A caller answering as Metal does hands on a `Metal` failure as Metal's, and
    /// tells a `Runtime` one apart.
    pub fn origin(self) -> FailureOrigin {
        match self {
            CommandBufferError::Injected(_)
            | CommandBufferError::DeviceLost
            | CommandBufferError::OutOfMemory
            | CommandBufferError::PageFault
            | CommandBufferError::GpuHang
            | CommandBufferError::GpuRecoveryVictim
            | CommandBufferError::AccessRevoked
            | CommandBufferError::Execution(_)
            | CommandBufferError::RenderTargetSize
            | CommandBufferError::DetachedBacking
            | CommandBufferError::PredicationOpenAtUnitEnd { .. }
            | CommandBufferError::ProtectedEncoder(_)
            | CommandBufferError::DefaultRasterSampleCount => FailureOrigin::Metal,
            CommandBufferError::Internal(_)
            | CommandBufferError::IndexPoolOverflow
            | CommandBufferError::CubicArgumentSampler
            | CommandBufferError::MeshRecordsOverflow(_)
            | CommandBufferError::TessellationRecordsOverflow(_)
            | CommandBufferError::MeshGridPastLimits
            | CommandBufferError::Refused
            | CommandBufferError::PipelineCreation(_)
            | CommandBufferError::EncodedUseRebound
            | CommandBufferError::CallRefused(_) => FailureOrigin::Runtime,
        }
    }

    /// Metal's `localizedDescription` of [`CommandBufferError::PredicationOpenAtUnitEnd`] (an M3
    /// Max under macOS 27.0.1).
    pub(crate) const PREDICATION_OPEN_AT_UNIT_END_DESCRIPTION: &'static str =
        "Discarded (victim of GPU error/recovery) \
         (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)";

    /// The error of Metal's `MTLCommandBufferError` code `code` (not 0), as the driver relayed it
    /// (`device/relay.rs`): `PageFault` for 3, `DeviceLost` for 11, else `Execution`.
    pub(crate) fn of_metal_code(code: u64) -> CommandBufferError {
        match code {
            3 => CommandBufferError::PageFault,
            11 => CommandBufferError::DeviceLost,
            code => CommandBufferError::Execution(code),
        }
    }
}

/// A bounded wait ended before the awaited state. No Metal counterpart: the runtime's bounded
/// forms of Metal's unbounded waits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimedOut;

/// The error of a failed Vulkan call: out-of-memory and device loss are typed, the rest raw.
pub(super) fn vulkan(result: ash::vk::Result) -> Error {
    use ash::vk::Result as R;
    match result {
        R::ERROR_OUT_OF_HOST_MEMORY => Error::OutOfMemory(MemoryKind::Host),
        R::ERROR_OUT_OF_DEVICE_MEMORY => Error::OutOfMemory(MemoryKind::Device),
        R::ERROR_DEVICE_LOST => Error::DeviceLost,
        other => Error::Vulkan(VkResultCode(other.as_raw())),
    }
}

/// A Metal validation error of `call`.
pub(super) fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// `problem` in `call`, for which plain Metal answers nil.
pub(super) fn nil(call: &'static str, problem: Problem) -> Error {
    Error::Nil(Nil::new(call, problem))
}

/// An internal error: the compile layer's output broke a contract the runtime relies on.
pub(super) fn internal(detail: &str) -> Error {
    Error::Compile(metal2vulkan::error::Error::Internal(
        metal2vulkan::error::InternalError::new(
            metal2vulkan::error::InternalKind::Invariant,
            detail,
        ),
    ))
}
